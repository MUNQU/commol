"""
Rate patterns along the axis for use in model transitions.

Provides a composable, Pydantic-validated API for expressing rate schedules
along the axis — discrete pulses, periodic events, sliding windows, sinusoids,
and arbitrary user formulas — as mathematical expressions that the
Commol engine can evaluate at each simulation step.

The only public symbol from this module is :class:`AxisPattern`. Multi-group
schedules are built by calling ``AxisPattern.add_group(...)`` directly on
the class and chaining further ``add_group`` / ``set_default`` calls on the
returned instance. The user never instantiates or names any other class.
"""

from abc import abstractmethod
from collections.abc import Iterable
from typing import ClassVar, Literal, Protocol, Self

from pydantic import (
    ConfigDict,
    BaseModel,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from commol.commol_rs import _commol_rs as commol_rs
from commol._data.ranges import RangeChecked
from commol._model.stratification_condition import StratificationCondition
from commol._model.transition import StratifiedRate


_COMBINE_MODES = ("sum", "max", "min")
"""How ``AxisPattern.combine`` can join its patterns."""


class _AddGroupFn(Protocol):
    """Signature exposed by the ``AxisPattern.add_group`` descriptor."""

    def __call__(
        self,
        conditions: list[StratificationCondition],
        schedule: "AxisPattern",
        *,
        source_compartment: str | None = None,
        absolute: bool | None = None,
    ) -> "AxisPattern":
        """Add a group to a schedule; see `AxisPattern.add_group`."""
        ...


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _number(value: float) -> str:
    """A bound as written in a formula: a whole number without ``.0``."""
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def _validate_value_string(value: float | str) -> None:
    """Reject string values that would break the surrounding ``if(..., VALUE, 0)``."""
    if isinstance(value, str) and "," in value:
        raise ValueError(
            f"'value' string must not contain commas (got {value!r}); "
            "commas would break the surrounding if(...) expression"
        )


def _require_consistent_group(
    patterns: "tuple[AxisPattern, ...]",
) -> "tuple[list[StratificationCondition] | None, str | None]":
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


def _compose_formulas(sub_formulas: list[str], mode: str) -> str:
    if mode not in _COMBINE_MODES:
        raise ValueError(
            f"'{mode}' is not a mode of combine: {', '.join(_COMBINE_MODES)}"
        )
    if mode == "sum":
        return f"({' + '.join(sub_formulas)})"
    body = sub_formulas[0]
    for f in sub_formulas[1:]:
        body = f"{mode}({body}, {f})"
    return f"({body})"


def _condition_key(
    conditions: list[StratificationCondition],
) -> frozenset[tuple[str, str, str]]:
    """Return a hashable key for a conditions list, used for duplicate detection."""
    return frozenset(
        (c.stratification, c.category or "", c.to or "") for c in conditions
    )


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

    def __get__(
        self,
        instance: "AxisPattern | None",
        owner: type["AxisPattern"],
    ) -> _AddGroupFn:
        if instance is None:

            def factory(
                conditions: list[StratificationCondition],
                schedule: "AxisPattern",
                *,
                source_compartment: str | None = None,
                absolute: bool | None = None,
            ) -> "AxisPattern":
                empty = _ScheduleAxisPattern()
                return empty._append_group(
                    conditions,
                    schedule,
                    source_compartment=source_compartment,
                    absolute=absolute,
                )

            return factory

        if not isinstance(instance, _ScheduleAxisPattern):

            def reject(
                conditions: list[StratificationCondition],  # noqa: ARG001
                schedule: "AxisPattern",  # noqa: ARG001
                *,
                source_compartment: str | None = None,  # noqa: ARG001
                absolute: bool | None = None,  # noqa: ARG001
            ) -> "AxisPattern":
                raise TypeError(
                    "add_group on a single AxisPattern is only valid as a "
                    "class-level factory (AxisPattern.add_group(...)); call "
                    "it on the class to start a new schedule."
                )

            return reject

        schedule_instance: "_ScheduleAxisPattern" = instance

        def bound(
            conditions: list[StratificationCondition],
            schedule: "AxisPattern",
            *,
            source_compartment: str | None = None,
            absolute: bool | None = None,
        ) -> "AxisPattern":
            return schedule_instance._append_group(
                conditions,
                schedule,
                source_compartment=source_compartment,
                absolute=absolute,
            )

        return bound


# ---------------------------------------------------------------------------
# Public base class
# ---------------------------------------------------------------------------


class AxisPattern(BaseModel):
    """
    An expression that changes along the axis, usable as (or within) a
    transition rate.

    Instances are created exclusively via the factory classmethods (``pulse``,
    ``periodic``, ``window``, etc.). Each instance may carry stratification
    conditions that restrict it to a specific sub-group via ``for_group``.

    Multi-group schedules are built by calling ``AxisPattern.add_group(...)``
    directly on the class. The first call returns a new schedule; chain
    further ``add_group`` and ``set_default`` calls on the returned instance.

    A ``AxisPattern`` is accepted directly by
    :meth:`commol._model.builder.ModelBuilder.add_transition` via the
    ``rate`` argument; no auxiliary conversion is required.

    Attributes
    ----------
    conditions : list[StratificationCondition] | None
        The stratification conditions of the sub-group the pattern applies
        to, set by ``for_group``. None: every compartment of the transition.
    source_compartment : str | None
        A compartment the rate is multiplied by, which makes the flow
        absolute, set by ``for_group``. None: the flow is proportional to its
        source.

    Examples
    --------
    >>> from commol import AxisPattern
    >>> # Single pattern
    >>> rate = AxisPattern.pulse(at=10, value=0.05)
    >>> str(rate)
    '(if(step == 10, 0.05, 0))'
    >>> # Multi-group schedule
    >>> from commol import StratificationCondition
    >>> rate = AxisPattern.add_group(
    ...     conditions=[
    ...         StratificationCondition(stratification="group", category="cat1")
    ...     ],
    ...     schedule=AxisPattern.pulse(at=10, value=0.05),
    ... ).add_group(
    ...     conditions=[
    ...         StratificationCondition(stratification="group", category="cat2")
    ...     ],
    ...     schedule=AxisPattern.pulse(at=10, value=0.02),
    ... )
    """

    model_config = ConfigDict(extra="forbid")

    conditions: list[StratificationCondition] | None = None
    source_compartment: str | None = None

    # `add_group` is a hybrid descriptor (see _AddGroupDescriptor above).
    # The ClassVar annotation tells Pydantic this is not a model field.
    add_group: ClassVar[_AddGroupDescriptor] = _AddGroupDescriptor()

    @property
    @abstractmethod
    def formula(self) -> str:
        """The mathematical expression for this pattern (always parenthesised)."""

    def _as_series_data(self) -> list[tuple[int, float]] | None:
        """Return pulse data as (step, value) points if representable as a series.

        Returns None for patterns that cannot be expressed as a numeric
        series (e.g. formula values, mathematical patterns).
        Overridden only by ``_PulseAxisPattern`` and ``_PulsesAxisPattern``.
        """
        return None

    def __str__(self) -> str:
        """Return the pattern's formula."""
        return self.formula

    def for_group(
        self,
        conditions: list[StratificationCondition],
        *,
        source_compartment: str | None = None,
    ) -> Self:
        """
        Return a copy of this pattern restricted to a stratification sub-group.

        Parameters
        ----------
        conditions : list[StratificationCondition]
            The stratification conditions of the sub-group, as in a
            stratified rate.
        source_compartment : str | None, default=None
            A compartment to multiply the rate by, which makes the flow
            absolute. If None, the flow is proportional to its source.

        Returns
        -------
        AxisPattern
            The copy, with `conditions` and `source_compartment` set.
        """
        return self.model_copy(
            update={
                "conditions": [c.model_copy() for c in conditions],
                "source_compartment": source_compartment,
            }
        )

    def _rate(self) -> str:
        """
        The rate of this pattern: its formula, times ``source_compartment``
        when that is set (an absolute flow). Without it, the engine multiplies
        the formula by the source compartment (a proportional flow).
        """
        if self.source_compartment is not None:
            return f"{self.formula} * {self.source_compartment}"
        return self.formula

    def _stratified_rate(self) -> StratifiedRate:
        """This pattern as the stratified rate of its conditions."""
        return StratifiedRate(conditions=self.conditions or [], rate=self._rate())

    # ------------------------------------------------------------------
    # Builder-consumption hooks
    # ------------------------------------------------------------------
    # The methods below are what ModelBuilder.add_transition reads when it
    # receives a AxisPattern as `rate=`. Users do not call them directly.

    def _builder_rate(self) -> str | float | None:
        """Default rate string to pass to add_transition (or None)."""
        if self.conditions is None:
            return self.formula
        return None

    def _builder_stratified_rates(self) -> list[StratifiedRate] | None:
        """Stratified-rates list to pass to add_transition (or None)."""
        if self.conditions is None:
            return None
        return [self._stratified_rate()]

    # ------------------------------------------------------------------
    # Methods only meaningful on schedules — base raises by default.
    # ------------------------------------------------------------------

    def set_default(self, pattern: "AxisPattern") -> "AxisPattern":  # noqa: ARG002
        """
        Register the condition-free fallback pattern (schedules only).

        Parameters
        ----------
        pattern : AxisPattern
            The pattern of every compartment that no group matches, with no
            conditions and no ``source_compartment``.

        Returns
        -------
        AxisPattern
            The schedule.

        Raises
        ------
        TypeError
            If this pattern is not a schedule, one that
            ``AxisPattern.add_group`` started.
        ValueError
            If the schedule has a default already, or `pattern` has
            conditions or a ``source_compartment``.
        """
        raise TypeError(
            "set_default is only valid on a schedule AxisPattern; "
            "start one with AxisPattern.add_group(...) first."
        )

    # ------------------------------------------------------------------
    # Pattern factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def pulse(cls, at: int, value: float | str) -> "AxisPattern":
        """Single-step pulse: rate = ``value`` at step ``at``, else 0."""
        _validate_value_string(value)
        return _PulseAxisPattern(at=at, value=value)

    @classmethod
    def pulses(
        cls,
        at: Iterable[int],
        value: float | str | Iterable[float | str],
    ) -> "AxisPattern":
        """Multi-step discrete pulses. Steps must be unique and non-negative.

        Parameters
        ----------
        at : Iterable[int]
            Step indices at which the pulse fires. Must be unique and non-negative.
        value : float | str | Iterable[float | str]
            Pulse magnitude. Either a single value applied to every step, or an
            iterable of values with the same length as ``at`` — one per step.
        """
        value_resolved: float | str | list[float | str] = (
            list(value) if not isinstance(value, (float, int, str)) else value
        )
        if isinstance(value_resolved, (float, int, str)):
            _validate_value_string(value_resolved)
        return _PulsesAxisPattern(steps=list(at), value=value_resolved)

    @classmethod
    def periodic(
        cls,
        period: int,
        value: float | str,
        offset: int = 0,
    ) -> "AxisPattern":
        """Periodic pulse every ``period`` steps starting at ``offset``."""
        _validate_value_string(value)
        return _PeriodicAxisPattern(period=period, value=value, offset=offset)

    @classmethod
    def interval(
        cls,
        start: float,
        end: float,
        value: float | str,
    ) -> "AxisPattern":
        """Constant rate ``value`` while ``t`` is in ``[start, end)``, else 0.

        In a DifferenceEquations model ``t`` is the step, so the rate acts on
        the steps ``start <= step < end``. In a DifferentialEquations model the
        interval opens and closes at any position, and the engine restarts its
        integration at both ends.
        """
        _validate_value_string(value)
        return _IntervalAxisPattern(start=start, end=end, value=value)

    @classmethod
    def sinusoid(
        cls,
        amplitude: float | str,
        period: float | str,
        phase: float | str = 0,
        baseline: float | str = 0,
    ) -> "AxisPattern":
        """``baseline + amplitude * sin(2*pi*(t - phase) / period)``."""
        return _SinusoidAxisPattern(
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
    ) -> "AxisPattern":
        """Gaussian bell pulse: ``peak * exp(-(t - center)^2 / (2 * width^2))``."""
        return _GaussianPulseAxisPattern(center=center, width=width, peak=peak)

    @classmethod
    def linear_ramp(
        cls,
        start: float,
        end: float,
        start_value: float,
        end_value: float,
    ) -> "AxisPattern":
        """Linear in ``t`` from ``start_value`` to ``end_value`` on ``[start, end)``.

        The rate is 0 outside the interval. In a DifferenceEquations model
        ``t`` is the step, so the ramp is read at each step; in a
        DifferentialEquations model it is continuous.
        """
        return _LinearRampAxisPattern(
            start=start,
            end=end,
            start_value=start_value,
            end_value=end_value,
        )

    @classmethod
    def combine(
        cls,
        *patterns: "AxisPattern",
        mode: Literal["sum", "max", "min"] = "sum",
    ) -> "AxisPattern":
        """
        Combine patterns into a single composed pattern.

        - ``combine(p)`` is the identity (returns ``p`` unchanged).
        - All input patterns must share the same ``conditions`` and
          ``source_compartment`` (or have none); the result preserves them.
        - ``mode`` is ``"sum"``, ``"max"`` (the largest value at each
          position) or ``"min"`` (the smallest).
        - Raises ``ValueError`` if no patterns are supplied, ``mode`` is not a
          mode, the inputs have inconsistent group bindings, or the parser of
          the engine rejects the combined formula, such as a ``"max"`` of
          more patterns than the nesting limit of a formula.
        """
        if len(patterns) == 0:
            raise ValueError("combine() requires at least one pattern")
        if len(patterns) == 1:
            return patterns[0]

        shared_conditions, shared_source = _require_consistent_group(patterns)
        result = _compose_formulas([p.formula for p in patterns], mode)
        return _ComputedAxisPattern(
            computed_formula=result,
            conditions=shared_conditions,
            source_compartment=shared_source,
        )

    @classmethod
    def from_formula(cls, expr: str) -> "AxisPattern":
        """
        Create a pattern from an arbitrary expression string.

        The string is wrapped in ``(...)`` so it composes safely. Raises
        ``ValueError`` with the message of the parser of the engine if the
        wrapped formula is not in the language of formulas.
        """
        return _ComputedAxisPattern(computed_formula=f"({expr})")


# ---------------------------------------------------------------------------
# Private concrete pattern subclasses
# ---------------------------------------------------------------------------


class _PulseAxisPattern(AxisPattern, RangeChecked):
    at: int
    value: float | str

    @property
    def formula(self) -> str:
        return f"(if(step == {self.at}, {self.value}, 0))"

    def _as_series_data(self) -> list[tuple[int, float]] | None:
        if not isinstance(self.value, (int, float)):
            return None
        return [(self.at, float(self.value))]


class _PulsesAxisPattern(AxisPattern, RangeChecked):
    steps: list[int]
    value: float | str | list[float | str]

    @field_validator("steps")
    @classmethod
    def _validate_steps(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("'at' must contain at least one step")
        if len(set(v)) != len(v):
            raise ValueError(f"'at' must contain unique steps (got {v!r})")
        return v

    @model_validator(mode="after")
    def _validate_value_list(self) -> "_PulsesAxisPattern":
        if isinstance(self.value, list):
            if len(self.value) != len(self.steps):
                raise ValueError(
                    f"'value' list length ({len(self.value)}) must match "
                    f"'at' length ({len(self.steps)})"
                )
            for a in self.value:
                _validate_value_string(a)
        return self

    @property
    def formula(self) -> str:
        if isinstance(self.value, list):
            terms = " + ".join(
                f"if(step == {s}, {a}, 0)" for s, a in zip(self.steps, self.value)
            )
        else:
            terms = " + ".join(f"if(step == {s}, {self.value}, 0)" for s in self.steps)
        return f"({terms})"

    def _as_series_data(self) -> list[tuple[int, float]] | None:
        if isinstance(self.value, list):
            if not all(isinstance(a, (int, float)) for a in self.value):
                return None
            return [(s, float(a)) for s, a in zip(self.steps, self.value)]
        if not isinstance(self.value, (int, float)):
            return None
        return [(s, float(self.value)) for s in self.steps]


class _PeriodicAxisPattern(AxisPattern, RangeChecked):
    period: int
    value: float | str
    offset: int = 0

    @model_validator(mode="after")
    def _validate_offset_lt_period(self) -> "_PeriodicAxisPattern":
        if self.offset >= self.period:
            raise ValueError(
                f"'offset' ({self.offset}) must be less than 'period' ({self.period})"
            )
        return self

    @property
    def formula(self) -> str:
        p = self.period
        if self.offset == 0:
            return f"(if(step - floor(step / {p}) * {p} == 0, {self.value}, 0))"
        o = self.offset
        return (
            f"(if((step - {o}) - floor((step - {o}) / {p}) * {p} == 0,"
            f" {self.value}, 0))"
        )


class _IntervalAxisPattern(AxisPattern, RangeChecked):
    start: float
    end: float
    value: float | str

    @model_validator(mode="after")
    def _validate_interval_bounds(self) -> "_IntervalAxisPattern":
        if self.start >= self.end:
            raise ValueError(
                f"'start' ({self.start}) must be less than 'end' ({self.end})"
            )
        return self

    @property
    def formula(self) -> str:
        start, end = _number(self.start), _number(self.end)
        return f"(if(t >= {start}, if(t < {end}, {self.value}, 0), 0))"


class _SinusoidAxisPattern(AxisPattern):
    amplitude: float | str
    period: float | str
    phase: float | str = 0
    baseline: float | str = 0

    @field_validator("period")
    @classmethod
    def _validate_period_nonzero(cls, v: float | str) -> float | str:
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


class _GaussianPulseAxisPattern(AxisPattern, RangeChecked):
    center: float
    width: float
    peak: float | str

    @property
    def formula(self) -> str:
        return (
            f"({self.peak} * exp(-((t - {self.center}) ** 2)"
            f" / (2 * {self.width} ** 2)))"
        )


class _LinearRampAxisPattern(AxisPattern, RangeChecked):
    start: float
    end: float
    start_value: float
    end_value: float

    @model_validator(mode="after")
    def _validate_ramp_bounds(self) -> "_LinearRampAxisPattern":
        if self.start >= self.end:
            raise ValueError(
                f"'start' ({self.start}) must be less than 'end' ({self.end})"
            )
        return self

    @property
    def formula(self) -> str:
        slope = self.end_value - self.start_value
        start, end = _number(self.start), _number(self.end)
        span = _number(self.end - self.start)
        inner = f"{self.start_value} + {slope} * (t - {start}) / {span}"
        return f"(if(t >= {start}, if(t < {end}, {inner}, 0), 0))"


class _ComputedAxisPattern(AxisPattern):
    """A formula that ``combine`` or ``from_formula`` wrote, which the parser
    of the engine checks."""

    computed_formula: str

    @field_validator("computed_formula")
    @classmethod
    def _check_formula(cls, value: str) -> str:
        """Check the formula with the parser of the engine."""
        commol_rs.core.MathExpression(value).check()
        return value

    @property
    def formula(self) -> str:
        return self.computed_formula


# ---------------------------------------------------------------------------
# Internal entry type for schedule groups
# ---------------------------------------------------------------------------


class _GroupEntry(BaseModel):
    """Pairs a pattern with its absolute-flow flag inside a schedule."""

    pattern: AxisPattern
    absolute: bool | None = None


# ---------------------------------------------------------------------------
# Schedule: private subclass exposed to users only through AxisPattern methods
# ---------------------------------------------------------------------------


class _ScheduleAxisPattern(AxisPattern):
    """
    Internal: a AxisPattern that holds a list of grouped sub-patterns and an
    optional default. Users never name this type — they obtain instances by
    calling ``AxisPattern.add_group(...)`` on the class.
    """

    patterns: list[_GroupEntry] = Field(default_factory=list)
    _has_default: bool = PrivateAttr(default=False)

    @property
    def formula(self) -> str:
        raise TypeError(
            "a schedule AxisPattern has no single formula; "
            "pass it to ModelBuilder.add_transition(rate=...) instead."
        )

    def __str__(self) -> str:
        return (
            f"<AxisPattern schedule with {len(self.patterns)} pattern(s), "
            f"default={'set' if self._has_default else 'unset'}>"
        )

    # ----- registration helpers -----

    def _append_group(
        self,
        conditions: list[StratificationCondition],
        schedule: AxisPattern,
        *,
        source_compartment: str | None,
        absolute: bool | None = None,
    ) -> "_ScheduleAxisPattern":
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
            if _condition_key(existing.pattern.conditions or []) == key:
                raise ValueError(
                    f"A group with conditions {pattern.conditions!r} has "
                    "already been registered."
                )
        self.patterns.append(_GroupEntry(pattern=pattern, absolute=absolute))
        return self

    def set_default(self, pattern: AxisPattern) -> "_ScheduleAxisPattern":
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
        self.patterns.append(_GroupEntry(pattern=pattern))
        self._has_default = True
        return self

    # ----- builder hooks -----

    def _builder_rate(self) -> str | float | None:
        for entry in self.patterns:
            if not entry.pattern.conditions:
                return entry.pattern._rate()
        return None

    def _builder_stratified_rates(self) -> list[StratifiedRate] | None:
        # A schedule starts with add_group, so it has a group with conditions.
        result = []
        for entry in self.patterns:
            if entry.pattern.conditions:
                result.append(
                    entry.pattern._stratified_rate().model_copy(
                        update={"absolute": entry.absolute}
                    )
                )
        return result


__all__ = ["AxisPattern"]
