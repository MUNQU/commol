# TimePattern API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

Time-varying rates and schedules per subgroup. Guide: [Time Patterns](https://munqu.github.io/commol/latest/guide/time-patterns/index.md).

## TimePattern

Bases: `BaseModel`

A time-varying expression usable as (or within) a transition rate.

Instances are created exclusively via the factory classmethods (`pulse`, `periodic`, `window`, etc.). Each instance may carry stratification conditions that restrict it to a specific sub-group via `for_group`.

Multi-group schedules are built by calling `TimePattern.add_group(...)` directly on the class. The first call returns a new schedule; chain further `add_group` and `set_default` calls on the returned instance.

A `TimePattern` is accepted directly by :meth:`commol.api.model_builder.ModelBuilder.add_transition` via the `rate` argument; no auxiliary conversion is required.

Examples:

```pycon
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
```

Source code in `commol/api/time_patterns.py`

```python
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

    def _as_time_series_data(self) -> list[tuple[int, float]] | None:
        """Return pulse data as (step, value) pairs if representable as TimeSeries.

        Returns None for patterns that cannot be expressed as a numeric
        time series (e.g. formula amounts, mathematical patterns).
        Overridden only by ``_PulseTimePattern`` and ``_PulsesTimePattern``.
        """
        return None

    def __str__(self) -> str:
        """Return the pattern's formula."""
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
    def pulses(
        cls,
        at: Iterable[int],
        amount: float | str | Iterable[float | str],
    ) -> "TimePattern":
        """Multi-step discrete pulses. Steps must be unique and non-negative.

        Parameters
        ----------
        at : Iterable[int]
            Step indices at which the pulse fires. Must be unique and non-negative.
        amount : float | str | Iterable[float | str]
            Pulse magnitude. Either a single value applied to every step, or an
            iterable of values with the same length as ``at`` — one per step.
        """
        amount_resolved: float | str | list[float | str] = (
            list(amount) if not isinstance(amount, (float, int, str)) else amount
        )
        if isinstance(amount_resolved, (float, int, str)):
            _validate_amount_string(amount_resolved)
        return _PulsesTimePattern(steps=list(at), amount=amount_resolved)

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
```

### Attributes

#### formula

```python
formula: str
```

The mathematical expression for this pattern (always parenthesised).

### Methods:

#### __str__

```python
__str__() -> str
```

Return the pattern's formula.

Source code in `commol/api/time_patterns.py`

```python
def __str__(self) -> str:
    """Return the pattern's formula."""
    return self.formula
```

#### for_group

```python
for_group(conditions: list[ConditionDict], *, source_compartment: str | None = None) -> Self
```

Return a copy of this pattern restricted to a stratification sub-group.

Source code in `commol/api/time_patterns.py`

```python
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
```

#### to_stratified_rate

```python
to_stratified_rate() -> StratifiedRateDict
```

Convert this pattern to a stratified-rate dict.

If `source_compartment` is set, the rate is multiplied by that compartment variable (absolute flow). Otherwise the rate is the formula alone (per-capita flow — the engine multiplies by the source compartment automatically).

Source code in `commol/api/time_patterns.py`

```python
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
```

#### set_default

```python
set_default(pattern: TimePattern) -> TimePattern
```

Register the condition-free fallback pattern (schedules only).

Source code in `commol/api/time_patterns.py`

```python
def set_default(self, pattern: "TimePattern") -> "TimePattern":  # noqa: ARG002
    """Register the condition-free fallback pattern (schedules only)."""
    raise TypeError(
        "set_default is only valid on a schedule TimePattern; "
        "start one with TimePattern.add_group(...) first."
    )
```

#### pulse

```python
pulse(at: int, amount: float | str) -> TimePattern
```

Single-step pulse: rate = `amount` at step `at`, else 0.

Source code in `commol/api/time_patterns.py`

```python
@classmethod
def pulse(cls, at: int, amount: float | str) -> "TimePattern":
    """Single-step pulse: rate = ``amount`` at step ``at``, else 0."""
    _validate_amount_string(amount)
    return _PulseTimePattern(at=at, amount=amount)
```

#### pulses

```python
pulses(at: Iterable[int], amount: float | str | Iterable[float | str]) -> TimePattern
```

Multi-step discrete pulses. Steps must be unique and non-negative.

Parameters:

| Name     | Type            | Description                                                             | Default         |
| -------- | --------------- | ----------------------------------------------------------------------- | --------------- |
| `at`     | `Iterable[int]` | Step indices at which the pulse fires. Must be unique and non-negative. | *required*      |
| `amount` | \`float         | str                                                                     | Iterable\[float |

Source code in `commol/api/time_patterns.py`

```python
@classmethod
def pulses(
    cls,
    at: Iterable[int],
    amount: float | str | Iterable[float | str],
) -> "TimePattern":
    """Multi-step discrete pulses. Steps must be unique and non-negative.

    Parameters
    ----------
    at : Iterable[int]
        Step indices at which the pulse fires. Must be unique and non-negative.
    amount : float | str | Iterable[float | str]
        Pulse magnitude. Either a single value applied to every step, or an
        iterable of values with the same length as ``at`` — one per step.
    """
    amount_resolved: float | str | list[float | str] = (
        list(amount) if not isinstance(amount, (float, int, str)) else amount
    )
    if isinstance(amount_resolved, (float, int, str)):
        _validate_amount_string(amount_resolved)
    return _PulsesTimePattern(steps=list(at), amount=amount_resolved)
```

#### periodic

```python
periodic(period: int, amount: float | str, offset: int = 0) -> TimePattern
```

Periodic pulse every `period` steps starting at `offset`.

Source code in `commol/api/time_patterns.py`

```python
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
```

#### window

```python
window(start: int, end: int, amount: float | str) -> TimePattern
```

Constant rate `amount` for step in `[start, end)`.

Source code in `commol/api/time_patterns.py`

```python
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
```

#### seasonal

```python
seasonal(amplitude: float | str, period: float | str, phase: float | str = 0, baseline: float | str = 0) -> TimePattern
```

`baseline + amplitude * sin(2*pi*(t - phase) / period)`.

Source code in `commol/api/time_patterns.py`

```python
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
```

#### gaussian_pulse

```python
gaussian_pulse(center: float, width: float, peak: float | str) -> TimePattern
```

Gaussian bell pulse: `peak * exp(-(t - center)^2 / (2 * width^2))`.

Source code in `commol/api/time_patterns.py`

```python
@classmethod
def gaussian_pulse(
    cls,
    center: float,
    width: float,
    peak: float | str,
) -> "TimePattern":
    """Gaussian bell pulse: ``peak * exp(-(t - center)^2 / (2 * width^2))``."""
    return _GaussianPulseTimePattern(center=center, width=width, peak=peak)
```

#### linear_ramp

```python
linear_ramp(start: int, end: int, start_value: float, end_value: float) -> TimePattern
```

Linear interpolation from `start_value` to `end_value` over `[start, end)`.

Source code in `commol/api/time_patterns.py`

```python
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
```

#### combine

```python
combine(*patterns: TimePattern, mode: CombineMode = SUM, config: SecurityConfig | None = None) -> TimePattern
```

Combine patterns into a single composed pattern.

- `combine(p)` is the identity (returns `p` unchanged).
- All input patterns must share the same `conditions` and `source_compartment` (or have none); the result preserves them.
- Raises `ValueError` if no patterns are supplied, the inputs have inconsistent group bindings, or the combined formula exceeds the security length cap.

Source code in `commol/api/time_patterns.py`

```python
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
```

#### from_formula

```python
from_formula(expr: str, config: SecurityConfig | None = None) -> TimePattern
```

Create a pattern from an arbitrary expression string.

The string is wrapped in `(...)` so it composes safely, then run through the full expression security validator.

Source code in `commol/api/time_patterns.py`

```python
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
```
