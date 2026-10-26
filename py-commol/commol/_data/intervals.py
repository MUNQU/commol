"""What a report and a figure read of prediction intervals.

A conformal method gives its intervals (`ConformalResult.intervals`); a
report and a figure of a run, of replicates or of a predictive take them and
read them through these protocols, so they import no method.
"""

from collections.abc import Mapping, Sequence
from typing import Protocol


class SeriesIntervalsData(Protocol):
    """The prediction intervals of one series, as `SeriesIntervals` holds
    them."""

    @property
    def positions(self) -> Sequence[float]:
        """Where the intervals are."""
        ...

    @property
    def prediction(self) -> Sequence[float]:
        """The point prediction at each position."""
        ...

    @property
    def lower(self) -> Sequence[float]:
        """The lower end of each interval."""
        ...

    @property
    def upper(self) -> Sequence[float]:
        """The upper end of each interval."""
        ...


class PredictionIntervalsData(Protocol):
    """Prediction intervals at a confidence level, by series, as
    `PredictionIntervals` holds them."""

    @property
    def confidence_level(self) -> float:
        """The level of the intervals."""
        ...

    @property
    def series(self) -> Mapping[str, SeriesIntervalsData]:
        """The intervals of each series, by label."""
        ...

    @property
    def unbounded(self) -> Mapping[str, Sequence[float]]:
        """The positions of each series whose interval is unbounded."""
        ...
