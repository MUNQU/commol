"""The result of a dependence analysis: the statistics of each series, of
each pair of series and of the series together, with their tests."""

from typing import TYPE_CHECKING

from pydantic import ConfigDict, Field

from commol._data.result_model import ResultModel
from commol.dependence.references import AtParameters, Refitted
from commol.observations.noise import DistributionNoise, NoNoise

if TYPE_CHECKING:
    from commol.commol_rs._commol_rs import (
        SeriesDependenceProtocol,
        SeriesStatisticsProtocol,
    )


class DrawPValues(ResultModel):
    """
    The spread of the p-values of the draws of a sample, each against its
    own reference: no test of the sample.

    Attributes
    ----------
    median : float
        The weighted median.
    share_below : float
        The weighted share at or below ``1 − confidence_level``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    median: float = Field(description="The weighted median.")
    share_below: float = Field(description="The weighted share below the level.")


def _draw_p_values(values: tuple[float, float] | None) -> DrawPValues | None:
    return (
        None if values is None else DrawPValues(median=values[0], share_below=values[1])
    )


class SeriesStatistics(ResultModel):
    """
    The dependence of the values of one series.

    The series reads its values in position order, on the
    grid of its smallest spacing; each lag pairs the values that exist at
    that distance.

    Attributes
    ----------
    label : str
        The label of the series.
    values : int
        The values it reads.
    reason : str | None
        Why it has no statistics: two observations at one position, a
        position off its grid, or a grid of more than 10 points per value.
    max_lag : int
        The largest lag, in steps of the grid.
    correlations : list[float | None]
        The autocorrelation at lags ``1 … max_lag``, each over its pairs;
        None at a lag with no pair.
    pairs : list[int]
        The pairs of values each lag reads.
    few_pairs : list[tuple[int, int]]
        Each lag of fewer than 4 pairs, as ``(lag, pairs)``: it keeps its
        correlation and is left out of ``Q̃`` and of its degrees of freedom.
    portmanteau : float | None
        The robust portmanteau ``Q̃`` (Dalla, Giraitis and Phillips 2022).
    portmanteau_reason : str | None
        Why there is no ``Q̃``: the series has no statistics, every lag
        reads fewer than 4 pairs, or no lag of 4 pairs or more has a
        product other than 0, as of constant values.
    degrees_of_freedom : int
        The lags in the sum of ``Q̃``: those of 4 pairs or more with a
        product other than 0.
    p_value : float | None
        The p-value of ``Q̃`` under the reference, with one draw.
    draw_p_values : DrawPValues | None
        The spread of the p-values of the draws, with many.
    tau : float | None
        The integrated autocorrelation time, by the initial monotone
        sequence of Geyer (1992).
    tau_reason : str | None
        Why ``τ`` is undefined: fewer than 4 values, constant values, or an
        estimate not above 0, which reads as below its band.
    tau_band : tuple[float, float] | None
        The central interval of ``τ`` under independence at the confidence
        level, with one draw; ``-inf`` where the replicates have no ``τ``,
        and None when none has one.
    cut_lag : int | None
        The largest lag in the sum of ``τ``; over several sets, the largest
        of theirs.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    values: int
    reason: str | None
    max_lag: int
    correlations: list[float | None]
    pairs: list[int]
    few_pairs: list[tuple[int, int]]
    portmanteau: float | None
    portmanteau_reason: str | None
    degrees_of_freedom: int
    p_value: float | None
    draw_p_values: DrawPValues | None
    tau: float | None
    tau_reason: str | None
    tau_band: tuple[float, float] | None
    cut_lag: int | None

    @classmethod
    def _of(cls, raw: "SeriesStatisticsProtocol") -> "SeriesStatistics":
        return cls(
            label=raw.label,
            values=raw.values,
            reason=raw.reason,
            max_lag=raw.max_lag,
            correlations=raw.correlations,
            pairs=raw.pairs,
            few_pairs=raw.few_pairs,
            portmanteau=raw.portmanteau,
            portmanteau_reason=raw.portmanteau_reason,
            degrees_of_freedom=raw.degrees_of_freedom,
            p_value=raw.p_value,
            draw_p_values=_draw_p_values(raw.draw_p_values),
            tau=raw.tau,
            tau_reason=raw.tau_reason,
            tau_band=raw.tau_band,
            cut_lag=raw.cut_lag,
        )


class SeriesCorrelation(ResultModel):
    """
    The correlation at lag 0 of two series, over the positions they share.

    Attributes
    ----------
    first : str
        The label of the first series.
    second : str
        The label of the second series.
    correlation : float | None
        None with fewer than two shared positions or a constant series.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    first: str
    second: str
    correlation: float | None


class ProblemDependence(ResultModel):
    """
    The dependence of the series together, over the standardized series
    ``z``: how many times the observations count their information.

    Attributes
    ----------
    variance_factor : float | None
        ``Var(Σz) / ΣVar(z)``, ``Σz`` summed at each position of any
        series: the dependence across series at one position.
    tau : float | None
        ``τ`` of ``Σz``: the dependence of the sum along the positions.
    tau_reason : str | None
        Why ``τ`` of ``Σz`` is undefined.
    information_factor : float | None
        Their product, ``LRV(Σz) / ΣVar(z)``: 1 for independent
        observations, 2 when each counts its information twice.
    p_value : float | None
        The p-value of the information factor under the reference, with one
        draw.
    draw_p_values : DrawPValues | None
        The spread of the p-values of the draws, with many.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    variance_factor: float | None
    tau: float | None
    tau_reason: str | None
    information_factor: float | None
    p_value: float | None
    draw_p_values: DrawPValues | None


class DependenceSettings(ResultModel):
    """
    The settings of a dependence analysis.

    Attributes
    ----------
    max_lag : int
        The largest lag of each series, in steps of its grid.
    confidence_level : float
        The level of the band of ``τ``.
    seed : int
        The seed of its random numbers.
    replicates : int | None
        The wild sets of `series_dependence`.
    noise : NoNoise | DistributionNoise | None
        The residuals of `residual_dependence`.
    randomizations : int | None
        The randomizations of `residual_dependence`.
    reference : AtParameters | Refitted | None
        The reference of `residual_dependence`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_lag: int
    confidence_level: float
    seed: int
    replicates: int | None = None
    noise: NoNoise | DistributionNoise | None = Field(
        default=None, discriminator="kind"
    )
    randomizations: int | None = None
    reference: AtParameters | Refitted | None = Field(
        default=None, discriminator="kind"
    )


class SeriesDependence(ResultModel):
    """
    Whether the values of each series are dependent, and how strongly.

    Attributes
    ----------
    series : list[SeriesStatistics]
        Each series, in order of first appearance of its label.
    correlations : list[SeriesCorrelation]
        Each pair of series with statistics.
    problem : ProblemDependence
        The series together.
    settings : DependenceSettings
        The settings of the analysis.
    draws : int
        The draws of the sample; 1 for values. With many, the statistics are
        the weighted means of the draws, and the result has no test of its
        own.
    failed : list[str]
        Each refit of a `Refitted` reference that failed (also logged).
    not_converged : int
        The refits kept although their optimizer did not converge.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    series: list[SeriesStatistics]
    correlations: list[SeriesCorrelation]
    problem: ProblemDependence
    settings: DependenceSettings
    draws: int
    failed: list[str]
    not_converged: int

    @classmethod
    def _of(
        cls, raw: "SeriesDependenceProtocol", settings: DependenceSettings
    ) -> "SeriesDependence":
        return cls(
            series=[SeriesStatistics._of(series) for series in raw.series],
            correlations=[
                SeriesCorrelation(first=first, second=second, correlation=value)
                for first, second, value in raw.correlations
            ],
            problem=ProblemDependence(
                variance_factor=raw.variance_factor,
                tau=raw.tau,
                tau_reason=raw.tau_reason,
                information_factor=raw.information_factor,
                p_value=raw.p_value,
                draw_p_values=_draw_p_values(raw.draw_p_values),
            ),
            settings=settings,
            draws=raw.draws,
            failed=raw.failed,
            not_converged=raw.not_converged,
        )
