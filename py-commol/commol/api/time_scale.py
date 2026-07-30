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

    Attributes
    ----------
    step_seconds : float
        Duration represented by one simulation step, in seconds.
    """

    model_config = ConfigDict(frozen=True)

    step_seconds: float = Field(
        gt=0.0, description="Duration of one simulation step, in seconds."
    )

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
