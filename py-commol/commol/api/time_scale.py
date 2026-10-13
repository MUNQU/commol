"""Conversion between physical time and simulation steps."""

import math

from pydantic import BaseModel, ConfigDict, Field

SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 60 * SECONDS_PER_MINUTE
SECONDS_PER_DAY = 24 * SECONDS_PER_HOUR
SECONDS_PER_WEEK = 7 * SECONDS_PER_DAY


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

    model_config = ConfigDict(extra="forbid", frozen=True)

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
