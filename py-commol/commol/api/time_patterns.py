"""
Time-varying rate patterns for use in model transitions.

Provides a composable, Pydantic-validated API for expressing time-dependent
rate schedules — discrete pulses, periodic events, sliding windows, seasonal
curves, and arbitrary user formulas — as mathematical expressions that the
Commol engine can evaluate at each simulation step.

The only public symbol from this module is :class:`TimePattern`. Multi-group
schedules are built by calling ``TimePattern.add_group(...)`` directly on
the class and chaining further ``add_group`` / ``set_default`` calls on the
returned instance. The user never instantiates or names any other class.
"""

from collections.abc import Iterable
from enum import StrEnum
from typing import Annotated, ClassVar, NotRequired, Protocol, Self, TypedDict

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

from commol.utils.security import (
    SecurityConfig,
    validate_expression_security,
)


class ConditionDict(TypedDict):
    """Shape of a single stratification condition used by ``TimePattern``.

    Also re-exported by :mod:`commol.api.model_builder` as
    ``StratificationConditionDict``.
    """

    stratification: str
    category: str
    to: NotRequired[str]


class StratifiedRateDict(TypedDict):
    """Shape of one entry in ``ModelBuilder.add_transition(stratified_rates=...)``.

    ``rate`` may be either a formula string or a numeric literal; the builder
    converts numeric values to strings before storing them.
    """

    conditions: list[ConditionDict]
    rate: str | float


class AddGroupFn(Protocol):
    """Signature exposed by the ``TimePattern.add_group`` descriptor."""

    def __call__(
        self,
        conditions: list[ConditionDict],
        schedule: "TimePattern",
        *,
        source_compartment: str | None = None,
    ) -> "TimePattern": ...


class CombineMode(StrEnum):
    """Combination modes for ``TimePattern.combine``."""

    SUM = "sum"
    MAX = "max"
    MIN = "min"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_amount_string(amount: float | str) -> None:
    """Reject string amounts that would break the surrounding ``if(..., AMOUNT, 0)``."""
    if isinstance(amount, str) and "," in amount:
        raise ValueError(
            f"'amount' string must not contain commas (got {amount!r}); "
            "commas would break the surrounding if(...) expression"
        )


def _require_consistent_group(
    patterns: "tuple[TimePattern, ...]",
) -> "tuple[list[ConditionDict] | None, str | None]":
    """Verify shared conditions/source_compartment; return the shared pair."""
    first = patterns[0]
    for p in patterns[1:]:
        if p.conditions != first.conditions:
            raise ValueError(
                "combine() requires all input patterns to share the same "
                "conditions (use for_group on the combined result instead)"
            )
        if p.source_compartment != first.source_compartment:
            raise ValueError(
                "combine() requires all input patterns to share the same "
                "source_compartment"
            )
    return first.conditions, first.source_compartment


def _compose_formulas(sub_formulas: list[str], mode: "CombineMode") -> str:
    if mode is CombineMode.SUM:
        body = " + ".join(sub_formulas)
    elif mode is CombineMode.MAX:
        body = sub_formulas[0]
        for f in sub_formulas[1:]:
            body = f"max({body}, {f})"
    elif mode is CombineMode.MIN:
        body = sub_formulas[0]
        for f in sub_formulas[1:]:
            body = f"min({body}, {f})"
    else:
        raise ValueError(f"Unknown combine mode: {mode!r}")
    return f"({body})"


def _enforce_length_cap(formula: str, config: SecurityConfig | None) -> None:
    cfg = config or SecurityConfig()
    if len(formula) > cfg.max_expression_length:
        raise ValueError(
            f"Combined formula length {len(formula)} exceeds maximum "
            f"{cfg.max_expression_length} characters"
        )


def _copy_condition(c: ConditionDict) -> ConditionDict:
    """Shallow-copy a ConditionDict while preserving its TypedDict type."""
    copy: ConditionDict = {
        "stratification": c["stratification"],
        "category": c["category"],
    }
    if "to" in c:
        copy["to"] = c["to"]
    return copy


_REQUIRED_CONDITION_KEYS = ("stratification", "category")


def _condition_key(
    conditions: list[ConditionDict] | None,
) -> frozenset[tuple[str, str, str]]:
    """Return a hashable key for a conditions list, used for duplicate detection."""
    if not conditions:
        return frozenset()
    keys: list[tuple[str, str, str]] = []
    for c in conditions:
        for required in _REQUIRED_CONDITION_KEYS:
            if required not in c:
                raise ValueError(
                    f"Condition is missing required key {required!r}: {c!r}"
                )
        keys.append((c["stratification"], c["category"], c.get("to", "")))
    return frozenset(keys)


# ---------------------------------------------------------------------------
# Descriptor that lets `add_group` act as both class-level factory and
# instance-level mutator without exposing a separate constructor.
# ---------------------------------------------------------------------------


class _AddGroupDescriptor:
    """
    Hybrid descriptor: when accessed via the class it creates a fresh schedule
    and registers the first group; when accessed via a schedule instance it
    appends to that schedule.
    """

    _owner: type["TimePattern"]

    def __set_name__(self, owner: type["TimePattern"], name: str) -> None:
        self._owner = owner

    def __get__(
        self,
        instance: "TimePattern | None",
        owner: type["TimePattern"],
    ) -> AddGroupFn:
        if instance is None:

            def factory(
                conditions: list[ConditionDict],
                schedule: "TimePattern",
                *,
                source_compartment: str | None = None,
            ) -> "TimePattern":
                empty = _ScheduleTimePattern()
                return empty._append_group(
                    conditions, schedule, source_compartment=source_compartment
                )

            return factory

        if not isinstance(instance, _ScheduleTimePattern):

            def reject(
                conditions: list[ConditionDict],  # noqa: ARG001
                schedule: "TimePattern",  # noqa: ARG001
                *,
                source_compartment: str | None = None,  # noqa: ARG001
            ) -> "TimePattern":
                raise TypeError(
                    "add_group on a single TimePattern is only valid as a "
                    "class-level factory (TimePattern.add_group(...)); call "
                    "it on the class to start a new schedule."
                )

            return reject

        schedule_instance: "_ScheduleTimePattern" = instance

        def bound(
            conditions: list[ConditionDict],
            schedule: "TimePattern",
            *,
            source_compartment: str | None = None,
        ) -> "TimePattern":
            return schedule_instance._append_group(
                conditions, schedule, source_compartment=source_compartment
            )

        return bound


# ---------------------------------------------------------------------------
# Public base class
# ---------------------------------------------------------------------------


class TimePattern(BaseModel):
    """
    A time-varying expression usable as (or within) a transition rate.

    Instances are created exclusively via the factory classmethods (``pulse``,
    ``periodic``, ``window``, etc.). Each instance may carry stratification
    conditions that restrict it to a specific sub-group via ``for_group``.

    Multi-group schedules are built by calling ``TimePattern.add_group(...)``
    directly on the class. The first call returns a new schedule; chain
    further ``add_group`` and ``set_default`` calls on the returned instance.

    A ``TimePattern`` is accepted directly by
    :meth:`commol.api.model_builder.ModelBuilder.add_transition` via the
    ``rate`` argument; no auxiliary conversion is required.

    Examples
    --------
    >>> from commol import TimePattern
    >>> # Single pattern
    >>> rate = TimePattern.pulse(at=10, amount=0.05)
    >>> str(rate)
    '(if(step == 10, 0.05, 0))'
    >>> # Multi-group schedule
    >>> rate = TimePattern.add_group(
    ...     conditions=[{"stratification": "group", "category": "cat1"}],
    ...     schedule=TimePattern.pulse(at=10, amount=0.05),
    ... ).add_group(
    ...     conditions=[{"stratification": "group", "category": "cat2"}],
    ...     schedule=TimePattern.pulse(at=10, amount=0.02),
    ... )
    """

    # Combine-mode constants — public; usable as TimePattern.SUM, .MAX, .MIN
    SUM: ClassVar[CombineMode] = CombineMode.SUM
    MAX: ClassVar[CombineMode] = CombineMode.MAX
    MIN: ClassVar[CombineMode] = CombineMode.MIN

    conditions: list[ConditionDict] | None = None
    source_compartment: str | None = None

    # `add_group` is a hybrid descriptor (see _AddGroupDescriptor above).
    # The ClassVar annotation tells Pydantic this is not a model field.
    add_group: ClassVar[_AddGroupDescriptor] = _AddGroupDescriptor()

    @property
    def formula(self) -> str:
        """The mathematical expression for this pattern (always parenthesised)."""
        raise NotImplementedError("Subclasses must implement formula")

    def __str__(self) -> str:
        return self.formula

    def for_group(
        self,
        conditions: list[ConditionDict],
        *,
        source_compartment: str | None = None,
    ) -> Self:
        """Return a copy of this pattern restricted to a stratification sub-group."""
        copied: list[ConditionDict] = [_copy_condition(c) for c in conditions]
        return self.model_copy(
            update={
                "conditions": copied,
                "source_compartment": source_compartment,
            }
        )

    def to_stratified_rate(self) -> StratifiedRateDict:
        """
        Convert this pattern to a stratified-rate dict.

        If ``source_compartment`` is set, the rate is multiplied by that
        compartment variable (absolute flow). Otherwise the rate is the
        formula alone (per-capita flow — the engine multiplies by the source
        compartment automatically).
        """
        if self.source_compartment is not None:
            rate = f"{self.formula} * {self.source_compartment}"
        else:
            rate = self.formula
        return StratifiedRateDict(
            conditions=self.conditions if self.conditions is not None else [],
            rate=rate,
        )

    # ------------------------------------------------------------------
    # Builder-consumption hooks
    # ------------------------------------------------------------------
    # The methods below are what ModelBuilder.add_transition reads when it
    # receives a TimePattern as `rate=`. Users do not call them directly.

    def _builder_rate(self) -> str | float | None:
        """Default rate string to pass to add_transition (or None)."""
        if self.conditions is None:
            return self.formula
        return None

    def _builder_stratified_rates(self) -> list[StratifiedRateDict] | None:
        """Stratified-rates list to pass to add_transition (or None)."""
        if self.conditions is None:
            return None
        return [self.to_stratified_rate()]

    # ------------------------------------------------------------------
    # Methods only meaningful on schedules — base raises by default.
    # ------------------------------------------------------------------

    def set_default(self, pattern: "TimePattern") -> "TimePattern":  # noqa: ARG002
        """Register the condition-free fallback pattern (schedules only)."""
        raise TypeError(
            "set_default is only valid on a schedule TimePattern; "
            "start one with TimePattern.add_group(...) first."
        )

    # ------------------------------------------------------------------
    # Pattern factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def pulse(cls, at: int, amount: float | str) -> "TimePattern":
        """Single-step pulse: rate = ``amount`` at step ``at``, else 0."""
        _validate_amount_string(amount)
        return _PulseTimePattern(at=at, amount=amount)

    @classmethod
    def pulses(cls, at: Iterable[int], amount: float | str) -> "TimePattern":
        """Multi-step discrete pulses. Steps must be unique and non-negative."""
        _validate_amount_string(amount)
        return _PulsesTimePattern(steps=list(at), amount=amount)

    @classmethod
    def periodic(
        cls,
        period: int,
        amount: float | str,
        offset: int = 0,
    ) -> "TimePattern":
        """Periodic pulse every ``period`` steps starting at ``offset``."""
        _validate_amount_string(amount)
        return _PeriodicTimePattern(period=period, amount=amount, offset=offset)

    @classmethod
    def window(
        cls,
        start: int,
        end: int,
        amount: float | str,
    ) -> "TimePattern":
        """Constant rate ``amount`` for step in ``[start, end)``."""
        _validate_amount_string(amount)
        return _WindowTimePattern(start=start, end=end, amount=amount)

    @classmethod
    def seasonal(
        cls,
        amplitude: float | str,
        period: float | str,
        phase: float | str = 0,
        baseline: float | str = 0,
    ) -> "TimePattern":
        """``baseline + amplitude * sin(2*pi*(t - phase) / period)``."""
        return _SeasonalTimePattern(
            amplitude=amplitude,
            period=period,
            phase=phase,
            baseline=baseline,
        )

    @classmethod
    def gaussian_pulse(
        cls,
        center: float,
        width: float,
        peak: float | str,
    ) -> "TimePattern":
        """Gaussian bell pulse: ``peak * exp(-(t - center)^2 / (2 * width^2))``."""
        return _GaussianPulseTimePattern(center=center, width=width, peak=peak)

    @classmethod
    def linear_ramp(
        cls,
        start: int,
        end: int,
        start_value: float,
        end_value: float,
    ) -> "TimePattern":
        """Linear interpolation from ``start_value`` to ``end_value`` over ``[start, end)``."""  # noqa: E501
        return _LinearRampTimePattern(
            start=start,
            end=end,
            start_value=start_value,
            end_value=end_value,
        )

    @classmethod
    def combine(
        cls,
        *patterns: "TimePattern",
        mode: CombineMode = CombineMode.SUM,
        config: SecurityConfig | None = None,
    ) -> "TimePattern":
        """
        Combine patterns into a single composed pattern.

        - ``combine(p)`` is the identity (returns ``p`` unchanged).
        - All input patterns must share the same ``conditions`` and
          ``source_compartment`` (or have none); the result preserves them.
        - Raises ``ValueError`` if no patterns are supplied, the inputs have
          inconsistent group bindings, or the combined formula exceeds the
          security length cap.
        """
        if len(patterns) == 0:
            raise ValueError("combine() requires at least one pattern")
        if len(patterns) == 1:
            return patterns[0]

        shared_conditions, shared_source = _require_consistent_group(patterns)
        result = _compose_formulas([p.formula for p in patterns], mode)
        _enforce_length_cap(result, config)

        return _ComputedTimePattern(
            computed_formula=result,
            conditions=shared_conditions,
            source_compartment=shared_source,
        )

    @classmethod
    def from_formula(
        cls,
        expr: str,
        config: SecurityConfig | None = None,
    ) -> "TimePattern":
        """
        Create a pattern from an arbitrary expression string.

        The string is wrapped in ``(...)`` so it composes safely, then run
        through the full expression security validator.
        """
        if not expr or not expr.strip():
            raise ValueError("'expr' must be a non-empty expression")
        wrapped = f"({expr})"
        validate_expression_security(wrapped, config)
        return _ComputedTimePattern(computed_formula=wrapped)


# ---------------------------------------------------------------------------
# Private concrete pattern subclasses
# ---------------------------------------------------------------------------


class _PulseTimePattern(TimePattern):
    at: Annotated[int, Field(ge=0)]
    amount: float | str

    @property
    def formula(self) -> str:
        return f"(if(step == {self.at}, {self.amount}, 0))"


class _PulsesTimePattern(TimePattern):
    steps: list[Annotated[int, Field(ge=0)]]
    amount: float | str

    @field_validator("steps")
    @classmethod
    def validate_steps(cls, v: list[int]) -> list[int]:
        config = SecurityConfig()
        max_pulses = config.max_function_calls - 1
        if not v:
            raise ValueError("'at' must contain at least one step")
        if len(v) > max_pulses:
            raise ValueError(
                f"Too many pulse steps: {len(v)} provided, at most {max_pulses} allowed"
            )
        if len(set(v)) != len(v):
            raise ValueError(f"'at' must contain unique steps (got {v!r})")
        return v

    @property
    def formula(self) -> str:
        terms = " + ".join(f"if(step == {s}, {self.amount}, 0)" for s in self.steps)
        return f"({terms})"


class _PeriodicTimePattern(TimePattern):
    period: Annotated[int, Field(gt=0)]
    amount: float | str
    offset: Annotated[int, Field(ge=0)] = 0

    @model_validator(mode="after")
    def validate_offset_lt_period(self) -> "_PeriodicTimePattern":
        if self.offset >= self.period:
            raise ValueError(
                f"'offset' ({self.offset}) must be less than 'period' ({self.period})"
            )
        return self

    @property
    def formula(self) -> str:
        p = self.period
        if self.offset == 0:
            return f"(if(step - floor(step / {p}) * {p} == 0, {self.amount}, 0))"
        o = self.offset
        return (
            f"(if((step - {o}) - floor((step - {o}) / {p}) * {p} == 0,"
            f" {self.amount}, 0))"
        )


class _WindowTimePattern(TimePattern):
    start: int
    end: int
    amount: float | str

    @model_validator(mode="after")
    def validate_window_bounds(self) -> "_WindowTimePattern":
        if self.start >= self.end:
            raise ValueError(
                f"'start' ({self.start}) must be less than 'end' ({self.end})"
            )
        return self

    @property
    def formula(self) -> str:
        return f"(if(step >= {self.start}, if(step < {self.end}, {self.amount}, 0), 0))"


class _SeasonalTimePattern(TimePattern):
    amplitude: float | str
    period: float | str
    phase: float | str = 0
    baseline: float | str = 0

    @field_validator("period")
    @classmethod
    def validate_period_nonzero(cls, v: float | str) -> float | str:
        if isinstance(v, (int, float)) and v == 0:
            raise ValueError("'period' must not be 0")
        if isinstance(v, str):
            try:
                numeric_value = float(v.strip())
            except ValueError:
                # Symbolic period (e.g. a parameter name) — accept as-is.
                return v
            if numeric_value == 0:
                raise ValueError("'period' must not be 0")
        return v

    @property
    def formula(self) -> str:
        return (
            f"({self.baseline} + {self.amplitude}"
            f" * sin(2 * pi * (t - {self.phase}) / {self.period}))"
        )


class _GaussianPulseTimePattern(TimePattern):
    center: float
    width: Annotated[float, Field(gt=0)]
    peak: float | str

    @property
    def formula(self) -> str:
        return (
            f"({self.peak} * exp(-((t - {self.center}) ** 2)"
            f" / (2 * {self.width} ** 2)))"
        )


class _LinearRampTimePattern(TimePattern):
    start: int
    end: int
    start_value: float
    end_value: float

    @model_validator(mode="after")
    def validate_ramp_bounds(self) -> "_LinearRampTimePattern":
        if self.start >= self.end:
            raise ValueError(
                f"'start' ({self.start}) must be less than 'end' ({self.end})"
            )
        return self

    @property
    def formula(self) -> str:
        slope = self.end_value - self.start_value
        span = self.end - self.start
        inner = f"{self.start_value} + {slope} * (step - {self.start}) / {span}"
        return f"(if(step >= {self.start}, if(step < {self.end}, {inner}, 0), 0))"


class _ComputedTimePattern(TimePattern):
    computed_formula: Annotated[str, Field(min_length=1)]

    @field_validator("computed_formula")
    @classmethod
    def validate_formula_length(cls, v: str) -> str:
        config = SecurityConfig()
        if len(v) > config.max_expression_length:
            raise ValueError(
                f"Formula length {len(v)} exceeds maximum "
                f"{config.max_expression_length} characters"
            )
        return v

    @property
    def formula(self) -> str:
        return self.computed_formula


# ---------------------------------------------------------------------------
# Schedule: private subclass exposed to users only through TimePattern methods
# ---------------------------------------------------------------------------


class _ScheduleTimePattern(TimePattern):
    """
    Internal: a TimePattern that holds a list of grouped sub-patterns and an
    optional default. Users never name this type — they obtain instances by
    calling ``TimePattern.add_group(...)`` on the class.
    """

    patterns: list[TimePattern] = Field(default_factory=list)
    _has_default: bool = PrivateAttr(default=False)

    @property
    def formula(self) -> str:
        raise TypeError(
            "a schedule TimePattern has no single formula; "
            "pass it to ModelBuilder.add_transition(rate=...) instead."
        )

    def __str__(self) -> str:  # pragma: no cover — descriptive only
        return (
            f"<TimePattern schedule with {len(self.patterns)} pattern(s), "
            f"default={'set' if self._has_default else 'unset'}>"
        )

    # ----- registration helpers -----

    def _append_group(
        self,
        conditions: list[ConditionDict],
        schedule: TimePattern,
        *,
        source_compartment: str | None,
    ) -> "_ScheduleTimePattern":
        if not conditions:
            raise ValueError(
                "'conditions' must be non-empty. "
                "Use set_default() for the condition-free fallback."
            )
        # Validate condition keys up front so malformed entries raise a clear
        # ValueError before any other processing.
        key = _condition_key(conditions)
        pattern = schedule.for_group(conditions, source_compartment=source_compartment)
        for existing in self.patterns:
            if _condition_key(existing.conditions) == key:
                raise ValueError(
                    f"A group with conditions {pattern.conditions!r} has "
                    "already been registered."
                )
        self.patterns.append(pattern)
        return self

    def set_default(self, pattern: TimePattern) -> "_ScheduleTimePattern":
        """
        Register the condition-free fallback pattern.

        The default applies to every compartment not matched by a more
        specific ``add_group``. It must not carry conditions or a
        ``source_compartment``.
        """
        if self._has_default:
            raise ValueError("A default pattern has already been registered.")
        if pattern.conditions:
            raise ValueError(
                "Default pattern must not have conditions. "
                "Use add_group() for sub-group-specific patterns."
            )
        if pattern.source_compartment is not None:
            raise ValueError(
                "Default pattern must not have source_compartment set. "
                "The default rate is applied to every unmatched compartment, "
                "so binding it to one compartment is almost certainly a bug."
            )
        self.patterns.append(pattern)
        self._has_default = True
        return self

    # ----- builder hooks -----

    def _builder_rate(self) -> str | float | None:
        for p in self.patterns:
            if not p.conditions:
                return p.to_stratified_rate()["rate"]
        return None

    def _builder_stratified_rates(self) -> list[StratifiedRateDict] | None:
        return [p.to_stratified_rate() for p in self.patterns if p.conditions]


__all__ = [
    "AddGroupFn",
    "CombineMode",
    "ConditionDict",
    "StratifiedRateDict",
    "TimePattern",
]
