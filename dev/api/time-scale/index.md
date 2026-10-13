# TimeScale API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

Converts between physical time and steps. Guide: [Steps and physical time](https://munqu.github.io/commol/latest/guide/core-concepts/#steps-and-physical-time).

## TimeScale

Bases: `BaseModel`

Physical duration of one simulation step.

A model is defined purely in steps. Attaching a duration to a step makes it possible to express periods in physical units, and to convert durations and probabilities into the per-step rates a transition expects.

Units up to a week have a fixed length and are provided directly. Months and years vary in length and have no single definition in step terms; give them as an explicit number of days.

Warnings

`rate_from_mean_duration` and `rate_from_probability` return per-step probabilities for a DifferenceEquations model. A DifferentialEquations model needs a continuous rate instead: `1 / duration` for a mean duration, and `-ln(1 - p) / period` for a probability `p` over a period (both in steps).

Attributes:

| Name           | Type    | Description                                              |
| -------------- | ------- | -------------------------------------------------------- |
| `step_seconds` | `float` | Duration represented by one simulation step, in seconds. |

Source code in `commol/api/time_scale.py`

```python
class TimeScale(BaseModel):
    """
    Physical duration of one simulation step.

    A model is defined purely in steps. Attaching a duration to a step makes it
    possible to express periods in physical units, and to convert durations and
    probabilities into the per-step rates a transition expects.

    Units up to a week have a fixed length and are provided directly. Months
    and years vary in length and have no single definition in step terms; give
    them as an explicit number of days.

    Warnings
    --------
    `rate_from_mean_duration` and `rate_from_probability` return per-step
    probabilities for a DifferenceEquations model. A DifferentialEquations
    model needs a continuous rate instead: ``1 / duration`` for a mean
    duration, and ``-ln(1 - p) / period`` for a probability ``p`` over a
    period (both in steps).

    Attributes
    ----------
    step_seconds : float
        Duration represented by one simulation step, in seconds.
    """

    model_config = ConfigDict(frozen=True)

    step_seconds: float = Field(
        gt=0.0, description="Duration of one simulation step, in seconds."
    )

    @classmethod
    def from_step_unit(cls, unit: str) -> "TimeScale":
        """
        Time scale whose step lasts one `unit`, such as ``"day"`` or ``"h"``.

        The axis of a model is its independent variable and need not be time
        (``step_unit="meter"``, for example); a time scale applies only when it
        is time.

        Parameters
        ----------
        unit : str
            Step unit of a model, of dimension time.

        Returns
        -------
        TimeScale
            A scale with ``step_seconds`` equal to one `unit`.

        Raises
        ------
        ValueError
            If `unit` is not a unit of time; the message names it.
        """
        from commol.utils.equations import ureg

        try:
            quantity = ureg.Quantity(1.0, unit)
        except Exception as error:
            raise ValueError(f"'{unit}' is not a unit: {error}") from error
        if quantity.dimensionality != ureg.second.dimensionality:
            raise ValueError(
                f"A TimeScale needs a step unit of time; '{unit}' has dimension "
                f"{quantity.dimensionality}. A model whose axis is not time has "
                f"no time scale."
            )
        return cls(step_seconds=float(quantity.to("second").magnitude))

    def steps_from_seconds(self, seconds: float) -> int:
        """
        Number of steps spanning a duration.

        Parameters
        ----------
        seconds : float
            Duration to convert.

        Returns
        -------
        int
            Steps spanning that duration.

        Raises
        ------
        ValueError
            If the duration is not a whole number of steps.
        """
        steps = seconds / self.step_seconds
        rounded = round(steps)
        if not math.isclose(steps, rounded, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"A step of {self.step_seconds} seconds does not divide "
                f"{seconds} seconds into a whole number of steps (got {steps})."
            )
        return int(rounded)

    def steps_from_minutes(self, minutes: float) -> int:
        """Number of steps spanning `minutes` minutes."""
        return self.steps_from_seconds(minutes * SECONDS_PER_MINUTE)

    def steps_from_hours(self, hours: float) -> int:
        """Number of steps spanning `hours` hours."""
        return self.steps_from_seconds(hours * SECONDS_PER_HOUR)

    def steps_from_days(self, days: float) -> int:
        """Number of steps spanning `days` days."""
        return self.steps_from_seconds(days * SECONDS_PER_DAY)

    def steps_from_weeks(self, weeks: float) -> int:
        """Number of steps spanning `weeks` weeks."""
        return self.steps_from_seconds(weeks * SECONDS_PER_WEEK)

    @property
    def steps_per_hour(self) -> int:
        """Steps in one hour."""
        return self.steps_from_seconds(SECONDS_PER_HOUR)

    @property
    def steps_per_day(self) -> int:
        """Steps in one day."""
        return self.steps_from_seconds(SECONDS_PER_DAY)

    @property
    def steps_per_week(self) -> int:
        """Steps in one week."""
        return self.steps_from_seconds(SECONDS_PER_WEEK)

    def rate_from_mean_duration(self, duration_seconds: float) -> float:
        """
        Per-step rate of leaving a state with a given mean residence time.

        Returns ``1 - exp(-step_seconds / duration_seconds)``, the probability
        that an exponential process of that mean duration fires within one
        step.

        This is a rate for a DifferenceEquations model only. A
        DifferentialEquations model needs the continuous rate
        ``step_seconds / duration_seconds``.

        Parameters
        ----------
        duration_seconds : float
            Mean time spent in the state, in seconds.

        Returns
        -------
        float
            Per-step rate, between 0 and 1.

        Raises
        ------
        ValueError
            If `duration_seconds` is not positive.
        """
        if duration_seconds <= 0.0:
            raise ValueError(
                f"duration_seconds must be positive, got {duration_seconds}."
            )
        return 1.0 - math.exp(-self.step_seconds / duration_seconds)

    def rate_from_probability(self, probability: float, period_steps: int) -> float:
        """
        Per-step rate equivalent to a probability over a longer period.

        Returns ``1 - (1 - probability) ** (1 / period_steps)``, the per-step
        rate that accumulates to `probability` over `period_steps` steps.

        This is a rate for a DifferenceEquations model only. A
        DifferentialEquations model needs the continuous rate
        ``-ln(1 - probability) / period_steps``.

        Parameters
        ----------
        probability : float
            Probability of the event over the whole period, between 0 and 1.
        period_steps : int
            Length of the period, in steps.

        Returns
        -------
        float
            Per-step rate, between 0 and 1.

        Raises
        ------
        ValueError
            If `probability` is outside [0, 1], or `period_steps` is not
            positive.
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability must be between 0 and 1, got {probability}.")
        if period_steps <= 0:
            raise ValueError(f"period_steps must be positive, got {period_steps}.")
        return 1.0 - (1.0 - probability) ** (1.0 / period_steps)

    def window_start(self, index: int, period_steps: int) -> int:
        """
        First step of a window.

        Parameters
        ----------
        index : int
            Zero-based window index.
        period_steps : int
            Length of one window, in steps.

        Returns
        -------
        int
            Step at which the window opens.

        Raises
        ------
        ValueError
            If `period_steps` is not positive or `index` is negative.
        """
        self._validate_window(index, period_steps)
        return index * period_steps

    def window_end(self, index: int, period_steps: int) -> int:
        """
        Step closing a window.

        The window covers ``(window_start(index), window_end(index)]``, so a
        quantity observed over it is the change between those two steps.

        Parameters
        ----------
        index : int
            Zero-based window index.
        period_steps : int
            Length of one window, in steps.

        Returns
        -------
        int
            Step at which the window closes, equal to the start of the next.

        Raises
        ------
        ValueError
            If `period_steps` is not positive or `index` is negative.
        """
        self._validate_window(index, period_steps)
        return (index + 1) * period_steps

    def window_index(self, step: int, period_steps: int) -> int:
        """
        Window a step belongs to.

        The exact inverse of :meth:`window_end`, and the containing window for
        any step in between. Step 0 belongs to window 0.

        Parameters
        ----------
        step : int
            Simulation step.
        period_steps : int
            Length of one window, in steps.

        Returns
        -------
        int
            Zero-based window index.

        Raises
        ------
        ValueError
            If `period_steps` is not positive or `step` is negative.
        """
        if step < 0:
            raise ValueError(f"step must not be negative, got {step}.")
        if period_steps <= 0:
            raise ValueError(f"period_steps must be positive, got {period_steps}.")
        return max(step - 1, 0) // period_steps

    @staticmethod
    def _validate_window(index: int, period_steps: int) -> None:
        """Reject a negative window index or a non-positive period."""
        if index < 0:
            raise ValueError(f"index must not be negative, got {index}.")
        if period_steps <= 0:
            raise ValueError(f"period_steps must be positive, got {period_steps}.")
```

### Attributes

#### steps_per_hour

```python
steps_per_hour: int
```

Steps in one hour.

#### steps_per_day

```python
steps_per_day: int
```

Steps in one day.

#### steps_per_week

```python
steps_per_week: int
```

Steps in one week.

### Methods:

#### from_step_unit

```python
from_step_unit(unit: str) -> TimeScale
```

Time scale whose step lasts one `unit`, such as `"day"` or `"h"`.

The axis of a model is its independent variable and need not be time (`step_unit="meter"`, for example); a time scale applies only when it is time.

Parameters:

| Name   | Type  | Description                              | Default    |
| ------ | ----- | ---------------------------------------- | ---------- |
| `unit` | `str` | Step unit of a model, of dimension time. | *required* |

Returns:

| Type        | Description                                  |
| ----------- | -------------------------------------------- |
| `TimeScale` | A scale with step_seconds equal to one unit. |

Raises:

| Type         | Description                                          |
| ------------ | ---------------------------------------------------- |
| `ValueError` | If unit is not a unit of time; the message names it. |

Source code in `commol/api/time_scale.py`

```python
@classmethod
def from_step_unit(cls, unit: str) -> "TimeScale":
    """
    Time scale whose step lasts one `unit`, such as ``"day"`` or ``"h"``.

    The axis of a model is its independent variable and need not be time
    (``step_unit="meter"``, for example); a time scale applies only when it
    is time.

    Parameters
    ----------
    unit : str
        Step unit of a model, of dimension time.

    Returns
    -------
    TimeScale
        A scale with ``step_seconds`` equal to one `unit`.

    Raises
    ------
    ValueError
        If `unit` is not a unit of time; the message names it.
    """
    from commol.utils.equations import ureg

    try:
        quantity = ureg.Quantity(1.0, unit)
    except Exception as error:
        raise ValueError(f"'{unit}' is not a unit: {error}") from error
    if quantity.dimensionality != ureg.second.dimensionality:
        raise ValueError(
            f"A TimeScale needs a step unit of time; '{unit}' has dimension "
            f"{quantity.dimensionality}. A model whose axis is not time has "
            f"no time scale."
        )
    return cls(step_seconds=float(quantity.to("second").magnitude))
```

#### steps_from_seconds

```python
steps_from_seconds(seconds: float) -> int
```

Number of steps spanning a duration.

Parameters:

| Name      | Type    | Description          | Default    |
| --------- | ------- | -------------------- | ---------- |
| `seconds` | `float` | Duration to convert. | *required* |

Returns:

| Type  | Description                   |
| ----- | ----------------------------- |
| `int` | Steps spanning that duration. |

Raises:

| Type         | Description                                     |
| ------------ | ----------------------------------------------- |
| `ValueError` | If the duration is not a whole number of steps. |

Source code in `commol/api/time_scale.py`

```python
def steps_from_seconds(self, seconds: float) -> int:
    """
    Number of steps spanning a duration.

    Parameters
    ----------
    seconds : float
        Duration to convert.

    Returns
    -------
    int
        Steps spanning that duration.

    Raises
    ------
    ValueError
        If the duration is not a whole number of steps.
    """
    steps = seconds / self.step_seconds
    rounded = round(steps)
    if not math.isclose(steps, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"A step of {self.step_seconds} seconds does not divide "
            f"{seconds} seconds into a whole number of steps (got {steps})."
        )
    return int(rounded)
```

#### steps_from_minutes

```python
steps_from_minutes(minutes: float) -> int
```

Number of steps spanning `minutes` minutes.

Source code in `commol/api/time_scale.py`

```python
def steps_from_minutes(self, minutes: float) -> int:
    """Number of steps spanning `minutes` minutes."""
    return self.steps_from_seconds(minutes * SECONDS_PER_MINUTE)
```

#### steps_from_hours

```python
steps_from_hours(hours: float) -> int
```

Number of steps spanning `hours` hours.

Source code in `commol/api/time_scale.py`

```python
def steps_from_hours(self, hours: float) -> int:
    """Number of steps spanning `hours` hours."""
    return self.steps_from_seconds(hours * SECONDS_PER_HOUR)
```

#### steps_from_days

```python
steps_from_days(days: float) -> int
```

Number of steps spanning `days` days.

Source code in `commol/api/time_scale.py`

```python
def steps_from_days(self, days: float) -> int:
    """Number of steps spanning `days` days."""
    return self.steps_from_seconds(days * SECONDS_PER_DAY)
```

#### steps_from_weeks

```python
steps_from_weeks(weeks: float) -> int
```

Number of steps spanning `weeks` weeks.

Source code in `commol/api/time_scale.py`

```python
def steps_from_weeks(self, weeks: float) -> int:
    """Number of steps spanning `weeks` weeks."""
    return self.steps_from_seconds(weeks * SECONDS_PER_WEEK)
```

#### rate_from_mean_duration

```python
rate_from_mean_duration(duration_seconds: float) -> float
```

Per-step rate of leaving a state with a given mean residence time.

Returns `1 - exp(-step_seconds / duration_seconds)`, the probability that an exponential process of that mean duration fires within one step.

This is a rate for a DifferenceEquations model only. A DifferentialEquations model needs the continuous rate `step_seconds / duration_seconds`.

Parameters:

| Name               | Type    | Description                               | Default    |
| ------------------ | ------- | ----------------------------------------- | ---------- |
| `duration_seconds` | `float` | Mean time spent in the state, in seconds. | *required* |

Returns:

| Type    | Description                     |
| ------- | ------------------------------- |
| `float` | Per-step rate, between 0 and 1. |

Raises:

| Type         | Description                          |
| ------------ | ------------------------------------ |
| `ValueError` | If duration_seconds is not positive. |

Source code in `commol/api/time_scale.py`

```python
def rate_from_mean_duration(self, duration_seconds: float) -> float:
    """
    Per-step rate of leaving a state with a given mean residence time.

    Returns ``1 - exp(-step_seconds / duration_seconds)``, the probability
    that an exponential process of that mean duration fires within one
    step.

    This is a rate for a DifferenceEquations model only. A
    DifferentialEquations model needs the continuous rate
    ``step_seconds / duration_seconds``.

    Parameters
    ----------
    duration_seconds : float
        Mean time spent in the state, in seconds.

    Returns
    -------
    float
        Per-step rate, between 0 and 1.

    Raises
    ------
    ValueError
        If `duration_seconds` is not positive.
    """
    if duration_seconds <= 0.0:
        raise ValueError(
            f"duration_seconds must be positive, got {duration_seconds}."
        )
    return 1.0 - math.exp(-self.step_seconds / duration_seconds)
```

#### rate_from_probability

```python
rate_from_probability(probability: float, period_steps: int) -> float
```

Per-step rate equivalent to a probability over a longer period.

Returns `1 - (1 - probability) ** (1 / period_steps)`, the per-step rate that accumulates to `probability` over `period_steps` steps.

This is a rate for a DifferenceEquations model only. A DifferentialEquations model needs the continuous rate `-ln(1 - probability) / period_steps`.

Parameters:

| Name           | Type    | Description                                                      | Default    |
| -------------- | ------- | ---------------------------------------------------------------- | ---------- |
| `probability`  | `float` | Probability of the event over the whole period, between 0 and 1. | *required* |
| `period_steps` | `int`   | Length of the period, in steps.                                  | *required* |

Returns:

| Type    | Description                     |
| ------- | ------------------------------- |
| `float` | Per-step rate, between 0 and 1. |

Raises:

| Type         | Description                                                        |
| ------------ | ------------------------------------------------------------------ |
| `ValueError` | If probability is outside [0, 1], or period_steps is not positive. |

Source code in `commol/api/time_scale.py`

```python
def rate_from_probability(self, probability: float, period_steps: int) -> float:
    """
    Per-step rate equivalent to a probability over a longer period.

    Returns ``1 - (1 - probability) ** (1 / period_steps)``, the per-step
    rate that accumulates to `probability` over `period_steps` steps.

    This is a rate for a DifferenceEquations model only. A
    DifferentialEquations model needs the continuous rate
    ``-ln(1 - probability) / period_steps``.

    Parameters
    ----------
    probability : float
        Probability of the event over the whole period, between 0 and 1.
    period_steps : int
        Length of the period, in steps.

    Returns
    -------
    float
        Per-step rate, between 0 and 1.

    Raises
    ------
    ValueError
        If `probability` is outside [0, 1], or `period_steps` is not
        positive.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be between 0 and 1, got {probability}.")
    if period_steps <= 0:
        raise ValueError(f"period_steps must be positive, got {period_steps}.")
    return 1.0 - (1.0 - probability) ** (1.0 / period_steps)
```

#### window_start

```python
window_start(index: int, period_steps: int) -> int
```

First step of a window.

Parameters:

| Name           | Type  | Description                     | Default    |
| -------------- | ----- | ------------------------------- | ---------- |
| `index`        | `int` | Zero-based window index.        | *required* |
| `period_steps` | `int` | Length of one window, in steps. | *required* |

Returns:

| Type  | Description                     |
| ----- | ------------------------------- |
| `int` | Step at which the window opens. |

Raises:

| Type         | Description                                           |
| ------------ | ----------------------------------------------------- |
| `ValueError` | If period_steps is not positive or index is negative. |

Source code in `commol/api/time_scale.py`

```python
def window_start(self, index: int, period_steps: int) -> int:
    """
    First step of a window.

    Parameters
    ----------
    index : int
        Zero-based window index.
    period_steps : int
        Length of one window, in steps.

    Returns
    -------
    int
        Step at which the window opens.

    Raises
    ------
    ValueError
        If `period_steps` is not positive or `index` is negative.
    """
    self._validate_window(index, period_steps)
    return index * period_steps
```

#### window_end

```python
window_end(index: int, period_steps: int) -> int
```

Step closing a window.

The window covers `(window_start(index), window_end(index)]`, so a quantity observed over it is the change between those two steps.

Parameters:

| Name           | Type  | Description                     | Default    |
| -------------- | ----- | ------------------------------- | ---------- |
| `index`        | `int` | Zero-based window index.        | *required* |
| `period_steps` | `int` | Length of one window, in steps. | *required* |

Returns:

| Type  | Description                                                      |
| ----- | ---------------------------------------------------------------- |
| `int` | Step at which the window closes, equal to the start of the next. |

Raises:

| Type         | Description                                           |
| ------------ | ----------------------------------------------------- |
| `ValueError` | If period_steps is not positive or index is negative. |

Source code in `commol/api/time_scale.py`

```python
def window_end(self, index: int, period_steps: int) -> int:
    """
    Step closing a window.

    The window covers ``(window_start(index), window_end(index)]``, so a
    quantity observed over it is the change between those two steps.

    Parameters
    ----------
    index : int
        Zero-based window index.
    period_steps : int
        Length of one window, in steps.

    Returns
    -------
    int
        Step at which the window closes, equal to the start of the next.

    Raises
    ------
    ValueError
        If `period_steps` is not positive or `index` is negative.
    """
    self._validate_window(index, period_steps)
    return (index + 1) * period_steps
```

#### window_index

```python
window_index(step: int, period_steps: int) -> int
```

Window a step belongs to.

The exact inverse of :meth:`window_end`, and the containing window for any step in between. Step 0 belongs to window 0.

Parameters:

| Name           | Type  | Description                     | Default    |
| -------------- | ----- | ------------------------------- | ---------- |
| `step`         | `int` | Simulation step.                | *required* |
| `period_steps` | `int` | Length of one window, in steps. | *required* |

Returns:

| Type  | Description              |
| ----- | ------------------------ |
| `int` | Zero-based window index. |

Raises:

| Type         | Description                                          |
| ------------ | ---------------------------------------------------- |
| `ValueError` | If period_steps is not positive or step is negative. |

Source code in `commol/api/time_scale.py`

```python
def window_index(self, step: int, period_steps: int) -> int:
    """
    Window a step belongs to.

    The exact inverse of :meth:`window_end`, and the containing window for
    any step in between. Step 0 belongs to window 0.

    Parameters
    ----------
    step : int
        Simulation step.
    period_steps : int
        Length of one window, in steps.

    Returns
    -------
    int
        Zero-based window index.

    Raises
    ------
    ValueError
        If `period_steps` is not positive or `step` is negative.
    """
    if step < 0:
        raise ValueError(f"step must not be negative, got {step}.")
    if period_steps <= 0:
        raise ValueError(f"period_steps must be positive, got {period_steps}.")
    return max(step - 1, 0) // period_steps
```
