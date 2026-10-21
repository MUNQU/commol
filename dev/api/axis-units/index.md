# AxisUnits API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

Converts values in units to steps of a model axis, gives the rate that an engine reads from a mean span or a probability over a period, and gives the window grid of a width in a unit. Guide: [Units of the axis](https://munqu.github.io/commol/latest/guide/core-concepts/#units-of-the-axis).

## AxisUnits

Bases: `BaseModel`

The unit of one step on a model axis, and the engine that reads its rates.

The axis of a model is its independent variable, of any quantity: time (`"day"`), a length (`"meter"`) or `"step"`, the default increment. `steps` converts a value in a unit of the same dimension to steps. `rate_from_mean_span` and `rate_from_probability` ask the engine what a rate means (`rate_meaning`) and give the rate it reads. `window_start`, `window_end`, `window_index` and `window_ends` give the grid of windows of one width, a span in a unit: window `w` covers the positions `(w * width, (w + 1) * width]`.

An axis of time with an `origin` is on the calendar: `position_of` and `moment_of` convert between calendar moments and positions, and `day_ends` and `week_ends` give the positions where calendar days and weeks end.

`Model.axis_units` builds the one of a model: its `step_unit`, its `origin` and its engine.

Attributes:

| Name        | Type       | Description                                                                                                          |
| ----------- | ---------- | -------------------------------------------------------------------------------------------------------------------- |
| `step_unit` | `str`      | Unit of one step, of one dimension with exponent 1. A name that pint does not define is a unit of its own dimension. |
| `engine`    | \`str      | None\`                                                                                                               |
| `origin`    | \`datetime | None\`                                                                                                               |

Examples:

```pycon
>>> hours = AxisUnits(step_unit="hour")
>>> hours.steps(3, "day")
72.0
>>> daily = AxisUnits(step_unit="day", engine="DifferentialEquations")
>>> daily.rate_from_mean_span(2, "week")
0.07142857142857142
>>> hours.window_end(2, 1, "week")
504.0
>>> from datetime import date
>>> dated = AxisUnits(step_unit="hour", origin=date(2020, 1, 1))
>>> dated.position_of(date(2020, 1, 2))
24.0
>>> dated.moment_of(36)
datetime.datetime(2020, 1, 2, 12, 0)
```

Source code in `commol/context/axis_units.py`

```python
class AxisUnits(BaseModel):
    """
    The unit of one step on a model axis, and the engine that reads its rates.

    The axis of a model is its independent variable, of any quantity: time
    (``"day"``), a length (``"meter"``) or ``"step"``, the default increment.
    `steps` converts a value in a unit of the same dimension to steps.
    `rate_from_mean_span` and `rate_from_probability` ask the engine what a
    rate means (`rate_meaning`) and give the rate it reads. `window_start`,
    `window_end`, `window_index` and `window_ends` give the grid of windows of
    one width, a span in a unit: window ``w`` covers the positions
    ``(w * width, (w + 1) * width]``.

    An axis of time with an `origin` is on the calendar: `position_of` and
    `moment_of` convert between calendar moments and positions, and
    `day_ends` and `week_ends` give the positions where calendar days and
    weeks end.

    `Model.axis_units` builds the one of a model: its ``step_unit``, its
    ``origin`` and its engine.

    Attributes
    ----------
    step_unit : str
        Unit of one step, of one dimension with exponent 1. A name that
        `pint` does not define is a unit of its own dimension.
    engine : str | None
        Name of the engine that reads the rates, such as
        ``"DifferenceEquations"``. None allows only `steps`.
    origin : datetime | None
        Calendar moment of position 0, with no time zone; a `date` is its
        midnight. It needs a step unit of time whose length is a whole number
        of days or a whole fraction of a day, such as ``"hour"`` or
        ``"day"``. None means the axis has no dates.

    Examples
    --------
    >>> hours = AxisUnits(step_unit="hour")
    >>> hours.steps(3, "day")
    72.0
    >>> daily = AxisUnits(step_unit="day", engine="DifferentialEquations")
    >>> daily.rate_from_mean_span(2, "week")
    0.07142857142857142
    >>> hours.window_end(2, 1, "week")
    504.0
    >>> from datetime import date
    >>> dated = AxisUnits(step_unit="hour", origin=date(2020, 1, 1))
    >>> dated.position_of(date(2020, 1, 2))
    24.0
    >>> dated.moment_of(36)
    datetime.datetime(2020, 1, 2, 12, 0)
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_unit: str = Field(
        default=DEFAULT_STEP_UNIT, description="Unit of one step on the axis."
    )
    engine: str | None = Field(
        default=None, description="Name of the engine that reads the rates."
    )
    origin: Origin | None = Field(
        default=None,
        description="Calendar moment of position 0, with no time zone.",
    )

    @model_validator(mode="after")
    def check_units_and_engine(self) -> Self:
        """Check the step unit and the origin, and ask for the engine."""
        define_units(self.step_unit)
        check_step_unit(self.step_unit)
        if self.origin is not None:
            seconds_per_step(self.step_unit)
        if self.engine is not None:
            commol_rs.engine.EngineType(self.engine)
        return self

    def position_of(self, moment: date) -> float:
        """
        Position of a calendar moment on the axis, in steps from the origin.

        The inverse of `moment_of`. A result within ``1e-9`` of an integer is
        that integer, as in `steps`.

        Parameters
        ----------
        moment : date
            A `datetime.datetime` with no time zone, or a `datetime.date`,
            read as its midnight.

        Returns
        -------
        float
            ``(moment - origin)`` in steps; negative before the origin.

        Raises
        ------
        ValueError
            If the axis has no origin, or `moment` has a time zone.
        """
        elapsed = naive_moment(moment) - self._origin()
        return self.steps(elapsed.total_seconds(), "second")

    def moment_of(self, position: float) -> datetime:
        """
        Calendar moment of a position on the axis.

        The inverse of `position_of`, to the microsecond.

        Parameters
        ----------
        position : float
            A finite position, in steps from the origin.

        Returns
        -------
        datetime
            ``origin + position`` steps, with no time zone.

        Raises
        ------
        ValueError
            If the axis has no origin, or `position` is not finite.
        """
        origin = self._origin()
        if not math.isfinite(position):
            raise ValueError(f"A position on the axis is finite, got {position}.")
        return origin + timedelta(seconds=position * seconds_per_step(self.step_unit))

    def day_ends(self, start: float, end: float) -> list[float]:
        """
        Positions where calendar days end, after `start` and up to `end`.

        A day ends at midnight, where the next starts. The positions cover
        ``(start, end]``, the interval of a window: the first is the first
        midnight after `start`, so a day that starts before `start` ends
        inside the range.

        Parameters
        ----------
        start : float
            First position of the range; a day that ends there is not in it.
        end : float
            Last position of the range.

        Returns
        -------
        list[float]
            The positions of the midnights in ``(start, end]``, ascending.

        Raises
        ------
        ValueError
            If the axis has no origin, or `start` or `end` is not finite.
        """
        return self._period_ends(end, self._next_midnight(start), 1)

    def week_ends(self, start: float, end: float, first_weekday: int) -> list[float]:
        """
        Positions where calendar weeks end, after `start` and up to `end`.

        A week starts at midnight of `first_weekday` and ends where the next
        starts, 7 days later. The positions cover ``(start, end]``, as
        `day_ends`.

        Parameters
        ----------
        start : float
            First position of the range; a week that ends there is not in it.
        end : float
            Last position of the range.
        first_weekday : int
            Day on which a week starts, from 0 (Monday) to 6 (Sunday), as
            `datetime.date.weekday` and the constants of the `calendar`
            module (``calendar.SUNDAY``) number them.

        Returns
        -------
        list[float]
            The positions of the week starts in ``(start, end]``, ascending.

        Raises
        ------
        ValueError
            If the axis has no origin, `start` or `end` is not finite, or
            `first_weekday` is not an integer from 0 to 6.
        """
        if first_weekday not in range(7):
            raise ValueError(
                "first_weekday is an integer from 0 (Monday) to 6 (Sunday), "
                f"got {first_weekday!r}."
            )
        midnight = self._next_midnight(start)
        first = midnight + timedelta(days=(first_weekday - midnight.weekday()) % 7)
        return self._period_ends(end, first, 7)

    def _next_midnight(self, start: float) -> datetime:
        """The first midnight after the moment of `start`."""
        moment = self.moment_of(start)
        return datetime.combine(moment.date(), time()) + timedelta(days=1)

    def _period_ends(self, end: float, first: datetime, days: int) -> list[float]:
        """Positions of `first` and every `days` days after it, up to `end`."""
        if not math.isfinite(end):
            raise ValueError(f"A position on the axis is finite, got {end}.")
        ends: list[float] = []
        moment = first
        position = self.position_of(moment)
        while position < end or math.isclose(
            position, end, rel_tol=WHOLE_STEP_TOLERANCE, abs_tol=WHOLE_STEP_TOLERANCE
        ):
            ends.append(position)
            moment += timedelta(days=days)
            position = self.position_of(moment)
        return ends

    def _origin(self) -> datetime:
        if self.origin is None:
            raise ValueError(
                "This axis has no origin, so it has no dates: give AxisUnits "
                "an origin, or declare origin in ModelBuilder."
            )
        return self.origin

    def steps(self, value: float, unit: str) -> float:
        """
        Number of steps that a value in `unit` spans on the axis.

        A result within ``1e-9`` of an integer is that integer, so
        ``steps(4.1, "hour")`` with a step of ``minute`` is ``246.0``, not
        ``245.99999999999997``. Whether a position must be a whole number of
        steps is the engine's check.

        Parameters
        ----------
        value : float
            The value, in `unit`.
        unit : str
            A unit of the dimension of the step unit, such as ``"day"`` for a
            step of ``"hour"``.

        Returns
        -------
        float
            The value in steps.

        Raises
        ------
        ValueError
            If `unit` has another dimension than the step unit; the message
            names both units, and says to declare a step unit when the step
            unit is ``step``.
        """
        define_units(unit)
        try:
            quantity = ureg.Quantity(float(value), ureg.parse_units(unit))
            result = float(quantity.to(ureg.parse_units(self.step_unit)).magnitude)
        except Exception as error:
            message = (
                f"Cannot convert {value} {unit} to steps of '{self.step_unit}': "
                f"'{unit}' has dimension "
                f"{ureg.parse_units(unit).dimensionality} and '{self.step_unit}' "
                f"has dimension {ureg.parse_units(self.step_unit).dimensionality}."
            )
            if self.step_unit == DEFAULT_STEP_UNIT:
                message += (
                    " Declare the unit of a step (step_unit in ModelBuilder) "
                    "to convert values in that unit."
                )
            raise ValueError(message) from error
        nearest = round(result)
        if math.isclose(result, nearest, rel_tol=0.0, abs_tol=WHOLE_STEP_TOLERANCE):
            return float(nearest)
        return result

    def rate_from_mean_span(self, span: float, unit: str) -> float:
        """
        Rate of one outflow whose residence has a mean `span` on the axis.

        The engine says what a rate means (`rate_meaning`). For a per-step
        rate (``DifferenceEquations``) it is ``1 - exp(-1/n)``, with `span`
        equal to ``n`` steps: the bin keeps ``exp(-k/n)`` of its start at each
        step ``k``, as the continuous process does, and its mean residence is
        ``1/rate`` steps, a little more than ``n``. For a continuous rate
        (``DifferentialEquations``) it is ``1/n``.

        Parameters
        ----------
        span : float
            Mean span of the residence, in `unit`.
        unit : str
            Unit of `span`, of the dimension of the step unit.

        Returns
        -------
        float
            The rate, per step.

        Raises
        ------
        ValueError
            If there is no engine, `unit` has another dimension than the step
            unit, or `span` is not positive.
        """
        return self._engine_type().rate_from_mean_span(self.steps(span, unit))

    def rate_from_probability(
        self, probability: float, period: float, unit: str
    ) -> float:
        """
        Rate of one outflow that moves `probability` of the bin over a period.

        The engine says what a rate means (`rate_meaning`). For a per-step
        rate (``DifferenceEquations``) it is ``1 - (1 - p)^(1/n)``, with the
        period equal to ``n`` steps; for a continuous rate
        (``DifferentialEquations``) it is ``-ln(1 - p)/n``. Either way the
        bin keeps ``1 - p`` of its start after the period.

        Parameters
        ----------
        probability : float
            Fraction of the bin that leaves over the period, between 0 and 1.
            A continuous rate needs it below 1.
        period : float
            Length of the period, in `unit`.
        unit : str
            Unit of `period`, of the dimension of the step unit.

        Returns
        -------
        float
            The rate, per step.

        Raises
        ------
        ValueError
            If there is no engine, `unit` has another dimension than the step
            unit, `period` is not positive, or `probability` is outside the
            range the engine accepts.
        """
        return self._engine_type().rate_from_probability(
            probability, self.steps(period, unit)
        )

    def window_start(self, index: int, span: float, unit: str) -> float:
        """
        Position at which window `index` opens.

        Parameters
        ----------
        index : int
            Zero-based window index.
        span : float
            Width of one window, in `unit`.
        unit : str
            Unit of `span`, of the dimension of the step unit.

        Returns
        -------
        float
            ``index`` times the width in steps.

        Raises
        ------
        ValueError
            If the width is not a positive whole number of steps, `unit` has
            another dimension than the step unit, or `index` is negative.
        """
        return windows.window_start(index, self._window_steps(span, unit))

    def window_end(self, index: int, span: float, unit: str) -> float:
        """
        Position at which window `index` closes, equal to the start of the next.

        Parameters
        ----------
        index : int
            Zero-based window index.
        span : float
            Width of one window, in `unit`.
        unit : str
            Unit of `span`, of the dimension of the step unit.

        Returns
        -------
        float
            ``index + 1`` times the width in steps.

        Raises
        ------
        ValueError
            If the width is not a positive whole number of steps, `unit` has
            another dimension than the step unit, or `index` is negative.
        """
        return windows.window_end(index, self._window_steps(span, unit))

    def window_index(self, at: float, span: float, unit: str) -> int:
        """
        Window that holds a position.

        The exact inverse of `window_end`, and the window that holds any
        position in between: ``max(ceil(at / width) - 1, 0)``, with the width
        in steps. Position 0 belongs to window 0. A report places each
        observation with this rule.

        Parameters
        ----------
        at : float
            A position on the axis, in steps.
        span : float
            Width of one window, in `unit`.
        unit : str
            Unit of `span`, of the dimension of the step unit.

        Returns
        -------
        int
            Zero-based window index.

        Raises
        ------
        ValueError
            If the width is not a positive whole number of steps, `unit` has
            another dimension than the step unit, or `at` is negative.
        """
        return windows.window_index(at, self._window_steps(span, unit))

    def window_ends(self, span: float, unit: str, end: float) -> list[float]:
        """
        Positions that close each complete window up to a position.

        Parameters
        ----------
        span : float
            Width of one window, in `unit`.
        unit : str
            Unit of `span`, of the dimension of the step unit.
        end : float
            Last position covered, in steps, such as the last position of a
            run.

        Returns
        -------
        list[float]
            The width in steps, two widths, ... up to `end`. A trailing
            partial window is not included.

        Raises
        ------
        ValueError
            If the width is not a positive whole number of steps, or `unit`
            has another dimension than the step unit.
        """
        return windows.window_ends(self._window_steps(span, unit), end)

    @property
    def rate_meaning(self) -> str:
        """
        What a rate means to the engine: ``"per_step"`` or ``"continuous"``.

        Raises
        ------
        ValueError
            If there is no engine.
        """
        return self._engine_type().rate_meaning

    def _window_steps(self, span: float, unit: str) -> float:
        """Width of a window in steps; it must be a positive whole number."""
        width = self.steps(span, unit)
        if not (width.is_integer() and width > 0):
            raise ValueError(
                f"A window must be a positive whole number of steps: {span} {unit} "
                f"is {width:g} steps of '{self.step_unit}'."
            )
        return width

    def _engine_type(self) -> "EngineTypeProtocol":
        if self.engine is None:
            raise ValueError(
                "A rate needs the engine that reads it: give AxisUnits an "
                "engine, or use Model.axis_units()."
            )
        return commol_rs.engine.EngineType(self.engine)
```

### Attributes

#### rate_meaning

```python
rate_meaning: str
```

What a rate means to the engine: `"per_step"` or `"continuous"`.

Raises:

| Type         | Description            |
| ------------ | ---------------------- |
| `ValueError` | If there is no engine. |

### Methods:

#### check_units_and_engine

```python
check_units_and_engine() -> Self
```

Check the step unit and the origin, and ask for the engine.

Source code in `commol/context/axis_units.py`

```python
@model_validator(mode="after")
def check_units_and_engine(self) -> Self:
    """Check the step unit and the origin, and ask for the engine."""
    define_units(self.step_unit)
    check_step_unit(self.step_unit)
    if self.origin is not None:
        seconds_per_step(self.step_unit)
    if self.engine is not None:
        commol_rs.engine.EngineType(self.engine)
    return self
```

#### position_of

```python
position_of(moment: date) -> float
```

Position of a calendar moment on the axis, in steps from the origin.

The inverse of `moment_of`. A result within `1e-9` of an integer is that integer, as in `steps`.

Parameters:

| Name     | Type   | Description                                                                      | Default    |
| -------- | ------ | -------------------------------------------------------------------------------- | ---------- |
| `moment` | `date` | A datetime.datetime with no time zone, or a datetime.date, read as its midnight. | *required* |

Returns:

| Type    | Description                                             |
| ------- | ------------------------------------------------------- |
| `float` | (moment - origin) in steps; negative before the origin. |

Raises:

| Type         | Description                                           |
| ------------ | ----------------------------------------------------- |
| `ValueError` | If the axis has no origin, or moment has a time zone. |

Source code in `commol/context/axis_units.py`

```python
def position_of(self, moment: date) -> float:
    """
    Position of a calendar moment on the axis, in steps from the origin.

    The inverse of `moment_of`. A result within ``1e-9`` of an integer is
    that integer, as in `steps`.

    Parameters
    ----------
    moment : date
        A `datetime.datetime` with no time zone, or a `datetime.date`,
        read as its midnight.

    Returns
    -------
    float
        ``(moment - origin)`` in steps; negative before the origin.

    Raises
    ------
    ValueError
        If the axis has no origin, or `moment` has a time zone.
    """
    elapsed = naive_moment(moment) - self._origin()
    return self.steps(elapsed.total_seconds(), "second")
```

#### moment_of

```python
moment_of(position: float) -> datetime
```

Calendar moment of a position on the axis.

The inverse of `position_of`, to the microsecond.

Parameters:

| Name       | Type    | Description                                  | Default    |
| ---------- | ------- | -------------------------------------------- | ---------- |
| `position` | `float` | A finite position, in steps from the origin. | *required* |

Returns:

| Type       | Description                                 |
| ---------- | ------------------------------------------- |
| `datetime` | origin + position steps, with no time zone. |

Raises:

| Type         | Description                                           |
| ------------ | ----------------------------------------------------- |
| `ValueError` | If the axis has no origin, or position is not finite. |

Source code in `commol/context/axis_units.py`

```python
def moment_of(self, position: float) -> datetime:
    """
    Calendar moment of a position on the axis.

    The inverse of `position_of`, to the microsecond.

    Parameters
    ----------
    position : float
        A finite position, in steps from the origin.

    Returns
    -------
    datetime
        ``origin + position`` steps, with no time zone.

    Raises
    ------
    ValueError
        If the axis has no origin, or `position` is not finite.
    """
    origin = self._origin()
    if not math.isfinite(position):
        raise ValueError(f"A position on the axis is finite, got {position}.")
    return origin + timedelta(seconds=position * seconds_per_step(self.step_unit))
```

#### day_ends

```python
day_ends(start: float, end: float) -> list[float]
```

Positions where calendar days end, after `start` and up to `end`.

A day ends at midnight, where the next starts. The positions cover `(start, end]`, the interval of a window: the first is the first midnight after `start`, so a day that starts before `start` ends inside the range.

Parameters:

| Name    | Type    | Description                                                      | Default    |
| ------- | ------- | ---------------------------------------------------------------- | ---------- |
| `start` | `float` | First position of the range; a day that ends there is not in it. | *required* |
| `end`   | `float` | Last position of the range.                                      | *required* |

Returns:

| Type          | Description                                                 |
| ------------- | ----------------------------------------------------------- |
| `list[float]` | The positions of the midnights in (start, end\], ascending. |

Raises:

| Type         | Description                                               |
| ------------ | --------------------------------------------------------- |
| `ValueError` | If the axis has no origin, or start or end is not finite. |

Source code in `commol/context/axis_units.py`

```python
def day_ends(self, start: float, end: float) -> list[float]:
    """
    Positions where calendar days end, after `start` and up to `end`.

    A day ends at midnight, where the next starts. The positions cover
    ``(start, end]``, the interval of a window: the first is the first
    midnight after `start`, so a day that starts before `start` ends
    inside the range.

    Parameters
    ----------
    start : float
        First position of the range; a day that ends there is not in it.
    end : float
        Last position of the range.

    Returns
    -------
    list[float]
        The positions of the midnights in ``(start, end]``, ascending.

    Raises
    ------
    ValueError
        If the axis has no origin, or `start` or `end` is not finite.
    """
    return self._period_ends(end, self._next_midnight(start), 1)
```

#### week_ends

```python
week_ends(start: float, end: float, first_weekday: int) -> list[float]
```

Positions where calendar weeks end, after `start` and up to `end`.

A week starts at midnight of `first_weekday` and ends where the next starts, 7 days later. The positions cover `(start, end]`, as `day_ends`.

Parameters:

| Name            | Type    | Description                                                                                                                                                 | Default    |
| --------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `start`         | `float` | First position of the range; a week that ends there is not in it.                                                                                           | *required* |
| `end`           | `float` | Last position of the range.                                                                                                                                 | *required* |
| `first_weekday` | `int`   | Day on which a week starts, from 0 (Monday) to 6 (Sunday), as datetime.date.weekday and the constants of the calendar module (calendar.SUNDAY) number them. | *required* |

Returns:

| Type          | Description                                                   |
| ------------- | ------------------------------------------------------------- |
| `list[float]` | The positions of the week starts in (start, end\], ascending. |

Raises:

| Type         | Description                                                                                            |
| ------------ | ------------------------------------------------------------------------------------------------------ |
| `ValueError` | If the axis has no origin, start or end is not finite, or first_weekday is not an integer from 0 to 6. |

Source code in `commol/context/axis_units.py`

```python
def week_ends(self, start: float, end: float, first_weekday: int) -> list[float]:
    """
    Positions where calendar weeks end, after `start` and up to `end`.

    A week starts at midnight of `first_weekday` and ends where the next
    starts, 7 days later. The positions cover ``(start, end]``, as
    `day_ends`.

    Parameters
    ----------
    start : float
        First position of the range; a week that ends there is not in it.
    end : float
        Last position of the range.
    first_weekday : int
        Day on which a week starts, from 0 (Monday) to 6 (Sunday), as
        `datetime.date.weekday` and the constants of the `calendar`
        module (``calendar.SUNDAY``) number them.

    Returns
    -------
    list[float]
        The positions of the week starts in ``(start, end]``, ascending.

    Raises
    ------
    ValueError
        If the axis has no origin, `start` or `end` is not finite, or
        `first_weekday` is not an integer from 0 to 6.
    """
    if first_weekday not in range(7):
        raise ValueError(
            "first_weekday is an integer from 0 (Monday) to 6 (Sunday), "
            f"got {first_weekday!r}."
        )
    midnight = self._next_midnight(start)
    first = midnight + timedelta(days=(first_weekday - midnight.weekday()) % 7)
    return self._period_ends(end, first, 7)
```

#### steps

```python
steps(value: float, unit: str) -> float
```

Number of steps that a value in `unit` spans on the axis.

A result within `1e-9` of an integer is that integer, so `steps(4.1, "hour")` with a step of `minute` is `246.0`, not `245.99999999999997`. Whether a position must be a whole number of steps is the engine's check.

Parameters:

| Name    | Type    | Description                                                                   | Default    |
| ------- | ------- | ----------------------------------------------------------------------------- | ---------- |
| `value` | `float` | The value, in unit.                                                           | *required* |
| `unit`  | `str`   | A unit of the dimension of the step unit, such as "day" for a step of "hour". | *required* |

Returns:

| Type    | Description         |
| ------- | ------------------- |
| `float` | The value in steps. |

Raises:

| Type         | Description                                                                                                                                 |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If unit has another dimension than the step unit; the message names both units, and says to declare a step unit when the step unit is step. |

Source code in `commol/context/axis_units.py`

```python
def steps(self, value: float, unit: str) -> float:
    """
    Number of steps that a value in `unit` spans on the axis.

    A result within ``1e-9`` of an integer is that integer, so
    ``steps(4.1, "hour")`` with a step of ``minute`` is ``246.0``, not
    ``245.99999999999997``. Whether a position must be a whole number of
    steps is the engine's check.

    Parameters
    ----------
    value : float
        The value, in `unit`.
    unit : str
        A unit of the dimension of the step unit, such as ``"day"`` for a
        step of ``"hour"``.

    Returns
    -------
    float
        The value in steps.

    Raises
    ------
    ValueError
        If `unit` has another dimension than the step unit; the message
        names both units, and says to declare a step unit when the step
        unit is ``step``.
    """
    define_units(unit)
    try:
        quantity = ureg.Quantity(float(value), ureg.parse_units(unit))
        result = float(quantity.to(ureg.parse_units(self.step_unit)).magnitude)
    except Exception as error:
        message = (
            f"Cannot convert {value} {unit} to steps of '{self.step_unit}': "
            f"'{unit}' has dimension "
            f"{ureg.parse_units(unit).dimensionality} and '{self.step_unit}' "
            f"has dimension {ureg.parse_units(self.step_unit).dimensionality}."
        )
        if self.step_unit == DEFAULT_STEP_UNIT:
            message += (
                " Declare the unit of a step (step_unit in ModelBuilder) "
                "to convert values in that unit."
            )
        raise ValueError(message) from error
    nearest = round(result)
    if math.isclose(result, nearest, rel_tol=0.0, abs_tol=WHOLE_STEP_TOLERANCE):
        return float(nearest)
    return result
```

#### rate_from_mean_span

```python
rate_from_mean_span(span: float, unit: str) -> float
```

Rate of one outflow whose residence has a mean `span` on the axis.

The engine says what a rate means (`rate_meaning`). For a per-step rate (`DifferenceEquations`) it is `1 - exp(-1/n)`, with `span` equal to `n` steps: the bin keeps `exp(-k/n)` of its start at each step `k`, as the continuous process does, and its mean residence is `1/rate` steps, a little more than `n`. For a continuous rate (`DifferentialEquations`) it is `1/n`.

Parameters:

| Name   | Type    | Description                                      | Default    |
| ------ | ------- | ------------------------------------------------ | ---------- |
| `span` | `float` | Mean span of the residence, in unit.             | *required* |
| `unit` | `str`   | Unit of span, of the dimension of the step unit. | *required* |

Returns:

| Type    | Description         |
| ------- | ------------------- |
| `float` | The rate, per step. |

Raises:

| Type         | Description                                                                                    |
| ------------ | ---------------------------------------------------------------------------------------------- |
| `ValueError` | If there is no engine, unit has another dimension than the step unit, or span is not positive. |

Source code in `commol/context/axis_units.py`

```python
def rate_from_mean_span(self, span: float, unit: str) -> float:
    """
    Rate of one outflow whose residence has a mean `span` on the axis.

    The engine says what a rate means (`rate_meaning`). For a per-step
    rate (``DifferenceEquations``) it is ``1 - exp(-1/n)``, with `span`
    equal to ``n`` steps: the bin keeps ``exp(-k/n)`` of its start at each
    step ``k``, as the continuous process does, and its mean residence is
    ``1/rate`` steps, a little more than ``n``. For a continuous rate
    (``DifferentialEquations``) it is ``1/n``.

    Parameters
    ----------
    span : float
        Mean span of the residence, in `unit`.
    unit : str
        Unit of `span`, of the dimension of the step unit.

    Returns
    -------
    float
        The rate, per step.

    Raises
    ------
    ValueError
        If there is no engine, `unit` has another dimension than the step
        unit, or `span` is not positive.
    """
    return self._engine_type().rate_from_mean_span(self.steps(span, unit))
```

#### rate_from_probability

```python
rate_from_probability(probability: float, period: float, unit: str) -> float
```

Rate of one outflow that moves `probability` of the bin over a period.

The engine says what a rate means (`rate_meaning`). For a per-step rate (`DifferenceEquations`) it is `1 - (1 - p)^(1/n)`, with the period equal to `n` steps; for a continuous rate (`DifferentialEquations`) it is `-ln(1 - p)/n`. Either way the bin keeps `1 - p` of its start after the period.

Parameters:

| Name          | Type    | Description                                                                                           | Default    |
| ------------- | ------- | ----------------------------------------------------------------------------------------------------- | ---------- |
| `probability` | `float` | Fraction of the bin that leaves over the period, between 0 and 1. A continuous rate needs it below 1. | *required* |
| `period`      | `float` | Length of the period, in unit.                                                                        | *required* |
| `unit`        | `str`   | Unit of period, of the dimension of the step unit.                                                    | *required* |

Returns:

| Type    | Description         |
| ------- | ------------------- |
| `float` | The rate, per step. |

Raises:

| Type         | Description                                                                                                                                           |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If there is no engine, unit has another dimension than the step unit, period is not positive, or probability is outside the range the engine accepts. |

Source code in `commol/context/axis_units.py`

```python
def rate_from_probability(
    self, probability: float, period: float, unit: str
) -> float:
    """
    Rate of one outflow that moves `probability` of the bin over a period.

    The engine says what a rate means (`rate_meaning`). For a per-step
    rate (``DifferenceEquations``) it is ``1 - (1 - p)^(1/n)``, with the
    period equal to ``n`` steps; for a continuous rate
    (``DifferentialEquations``) it is ``-ln(1 - p)/n``. Either way the
    bin keeps ``1 - p`` of its start after the period.

    Parameters
    ----------
    probability : float
        Fraction of the bin that leaves over the period, between 0 and 1.
        A continuous rate needs it below 1.
    period : float
        Length of the period, in `unit`.
    unit : str
        Unit of `period`, of the dimension of the step unit.

    Returns
    -------
    float
        The rate, per step.

    Raises
    ------
    ValueError
        If there is no engine, `unit` has another dimension than the step
        unit, `period` is not positive, or `probability` is outside the
        range the engine accepts.
    """
    return self._engine_type().rate_from_probability(
        probability, self.steps(period, unit)
    )
```

#### window_start

```python
window_start(index: int, span: float, unit: str) -> float
```

Position at which window `index` opens.

Parameters:

| Name    | Type    | Description                                      | Default    |
| ------- | ------- | ------------------------------------------------ | ---------- |
| `index` | `int`   | Zero-based window index.                         | *required* |
| `span`  | `float` | Width of one window, in unit.                    | *required* |
| `unit`  | `str`   | Unit of span, of the dimension of the step unit. | *required* |

Returns:

| Type    | Description                     |
| ------- | ------------------------------- |
| `float` | index times the width in steps. |

Raises:

| Type         | Description                                                                                                                |
| ------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If the width is not a positive whole number of steps, unit has another dimension than the step unit, or index is negative. |

Source code in `commol/context/axis_units.py`

```python
def window_start(self, index: int, span: float, unit: str) -> float:
    """
    Position at which window `index` opens.

    Parameters
    ----------
    index : int
        Zero-based window index.
    span : float
        Width of one window, in `unit`.
    unit : str
        Unit of `span`, of the dimension of the step unit.

    Returns
    -------
    float
        ``index`` times the width in steps.

    Raises
    ------
    ValueError
        If the width is not a positive whole number of steps, `unit` has
        another dimension than the step unit, or `index` is negative.
    """
    return windows.window_start(index, self._window_steps(span, unit))
```

#### window_end

```python
window_end(index: int, span: float, unit: str) -> float
```

Position at which window `index` closes, equal to the start of the next.

Parameters:

| Name    | Type    | Description                                      | Default    |
| ------- | ------- | ------------------------------------------------ | ---------- |
| `index` | `int`   | Zero-based window index.                         | *required* |
| `span`  | `float` | Width of one window, in unit.                    | *required* |
| `unit`  | `str`   | Unit of span, of the dimension of the step unit. | *required* |

Returns:

| Type    | Description                         |
| ------- | ----------------------------------- |
| `float` | index + 1 times the width in steps. |

Raises:

| Type         | Description                                                                                                                |
| ------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If the width is not a positive whole number of steps, unit has another dimension than the step unit, or index is negative. |

Source code in `commol/context/axis_units.py`

```python
def window_end(self, index: int, span: float, unit: str) -> float:
    """
    Position at which window `index` closes, equal to the start of the next.

    Parameters
    ----------
    index : int
        Zero-based window index.
    span : float
        Width of one window, in `unit`.
    unit : str
        Unit of `span`, of the dimension of the step unit.

    Returns
    -------
    float
        ``index + 1`` times the width in steps.

    Raises
    ------
    ValueError
        If the width is not a positive whole number of steps, `unit` has
        another dimension than the step unit, or `index` is negative.
    """
    return windows.window_end(index, self._window_steps(span, unit))
```

#### window_index

```python
window_index(at: float, span: float, unit: str) -> int
```

Window that holds a position.

The exact inverse of `window_end`, and the window that holds any position in between: `max(ceil(at / width) - 1, 0)`, with the width in steps. Position 0 belongs to window 0. A report places each observation with this rule.

Parameters:

| Name   | Type    | Description                                      | Default    |
| ------ | ------- | ------------------------------------------------ | ---------- |
| `at`   | `float` | A position on the axis, in steps.                | *required* |
| `span` | `float` | Width of one window, in unit.                    | *required* |
| `unit` | `str`   | Unit of span, of the dimension of the step unit. | *required* |

Returns:

| Type  | Description              |
| ----- | ------------------------ |
| `int` | Zero-based window index. |

Raises:

| Type         | Description                                                                                                             |
| ------------ | ----------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If the width is not a positive whole number of steps, unit has another dimension than the step unit, or at is negative. |

Source code in `commol/context/axis_units.py`

```python
def window_index(self, at: float, span: float, unit: str) -> int:
    """
    Window that holds a position.

    The exact inverse of `window_end`, and the window that holds any
    position in between: ``max(ceil(at / width) - 1, 0)``, with the width
    in steps. Position 0 belongs to window 0. A report places each
    observation with this rule.

    Parameters
    ----------
    at : float
        A position on the axis, in steps.
    span : float
        Width of one window, in `unit`.
    unit : str
        Unit of `span`, of the dimension of the step unit.

    Returns
    -------
    int
        Zero-based window index.

    Raises
    ------
    ValueError
        If the width is not a positive whole number of steps, `unit` has
        another dimension than the step unit, or `at` is negative.
    """
    return windows.window_index(at, self._window_steps(span, unit))
```

#### window_ends

```python
window_ends(span: float, unit: str, end: float) -> list[float]
```

Positions that close each complete window up to a position.

Parameters:

| Name   | Type    | Description                                                          | Default    |
| ------ | ------- | -------------------------------------------------------------------- | ---------- |
| `span` | `float` | Width of one window, in unit.                                        | *required* |
| `unit` | `str`   | Unit of span, of the dimension of the step unit.                     | *required* |
| `end`  | `float` | Last position covered, in steps, such as the last position of a run. | *required* |

Returns:

| Type          | Description                                                                               |
| ------------- | ----------------------------------------------------------------------------------------- |
| `list[float]` | The width in steps, two widths, ... up to end. A trailing partial window is not included. |

Raises:

| Type         | Description                                                                                             |
| ------------ | ------------------------------------------------------------------------------------------------------- |
| `ValueError` | If the width is not a positive whole number of steps, or unit has another dimension than the step unit. |

Source code in `commol/context/axis_units.py`

```python
def window_ends(self, span: float, unit: str, end: float) -> list[float]:
    """
    Positions that close each complete window up to a position.

    Parameters
    ----------
    span : float
        Width of one window, in `unit`.
    unit : str
        Unit of `span`, of the dimension of the step unit.
    end : float
        Last position covered, in steps, such as the last position of a
        run.

    Returns
    -------
    list[float]
        The width in steps, two widths, ... up to `end`. A trailing
        partial window is not included.

    Raises
    ------
    ValueError
        If the width is not a positive whole number of steps, or `unit`
        has another dimension than the step unit.
    """
    return windows.window_ends(self._window_steps(span, unit), end)
```
