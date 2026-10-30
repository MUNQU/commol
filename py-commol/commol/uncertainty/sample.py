"""The weighted parameter sample that every method with draws returns."""

import math
from collections.abc import Iterator
from typing import TYPE_CHECKING, Self

from pydantic import (
    ConfigDict,
    Field,
    JsonValue,
    PrivateAttr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from commol._data.result_model import ResultModel
from commol.commol_rs import _commol_rs as commol_rs

if TYPE_CHECKING:
    from commol.commol_rs._commol_rs import ParameterSampleProtocol


class SampleMethod(ResultModel):
    """
    What made a sample: the name of the method and its settings.

    Attributes
    ----------
    name : str
        The name of the method, such as ``"bayesian_bootstrap"``.
    settings : dict[str, JsonValue]
        The settings the method ran with.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="The name of the method.")
    settings: dict[str, JsonValue] = Field(
        default_factory=dict, description="The settings the method ran with."
    )

    @field_validator("settings")
    @classmethod
    def _settings_are_finite(
        cls, settings: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        """
        Refuse a number that is not finite: JSON has none, so a file would
        read it back as a text, and no setting of a method takes one.
        """
        pending: list[tuple[str, JsonValue]] = list(settings.items())
        while pending:
            path, value = pending.pop()
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"settings.{path} is not finite, got {value}")
            if isinstance(value, dict):
                pending.extend((f"{path}.{key}", item) for key, item in value.items())
            elif isinstance(value, list):
                pending.extend(
                    (f"{path}.{index}", item) for index, item in enumerate(value)
                )
        return settings


class ParameterSummary(ResultModel):
    """
    Weighted statistics of one parameter of a sample.

    Attributes
    ----------
    mean : float
        The weighted mean.
    median : float
        The weighted median.
    std : float
        The weighted standard deviation (`ParameterSample.covariance`).
    lower : float
        The lower end of the central interval of the confidence level.
    upper : float
        The upper end of the central interval of the confidence level.
    min : float
        The smallest value of positive weight.
    max : float
        The largest value of positive weight.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mean: float
    median: float
    std: float
    lower: float
    upper: float
    min: float
    max: float


class ParameterSample(ResultModel):
    """
    Draws of the parameters of a calibration problem, each with a weight.

    A method that has draws returns them in a sample: a bootstrap, a
    sequential Monte Carlo run, a Markov chain. Plots, reports, the
    predictive and the scores read a sample, never the result of one method.
    Every statistic is weighted and computed in Rust; with equal weights it
    is the unweighted statistic.

    A sample has no single set of parameters: to run one draw, take it from
    `draws`.

    Attributes
    ----------
    names : list[str]
        One name per column, in the order of the problem's parameters.
    bounds : list[tuple[float, float]]
        The bounds of each parameter; every value lies inside them.
    values : list[list[float]]
        One row per draw, one column per name.
    log_weights : list[float]
        The natural logarithm of each draw's weight, normalized so that the
        weights sum to 1. A draw of weight zero has ``-inf``.
    groups : list[int] | None
        The group of each draw (a chain, a replicate or a wave), or None.
    method : SampleMethod
        What made the sample.

    Raises
    ------
    ValueError
        If a value is not finite or outside its bounds, a log weight is NaN or
        ``+inf``, every log weight is ``-inf``, a name repeats, or the shapes
        differ.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    names: list[str] = Field(description="One name per column.")
    bounds: list[tuple[float, float]] = Field(description="Bounds of each parameter.")
    values: list[list[float]] = Field(description="One row per draw.")
    log_weights: list[float] = Field(description="Normalized log weight of each draw.")
    groups: list[int] | None = Field(default=None, description="Group of each draw.")
    method: SampleMethod = Field(description="What made the sample.")

    _rust: "ParameterSampleProtocol" = PrivateAttr()

    @field_validator("log_weights")
    @classmethod
    def _normalize_log_weights(
        cls, log_weights: list[float], info: ValidationInfo
    ) -> list[float]:
        """
        Check the invariants in Rust and normalize the log weights; with a
        field before them that is not valid, pydantic reports that field.
        """
        data = info.data
        if not {"names", "bounds", "values"} <= data.keys():
            return log_weights
        checked = commol_rs.inference.ParameterSample(
            data["names"], data["bounds"], data["values"], log_weights, None
        )
        return checked.log_weights

    @model_validator(mode="after")
    def _build_engine_sample(self) -> Self:
        """Keep the sample of the engine, which computes the statistics and
        which the engine reads as the handle of this sample."""
        self._rust = commol_rs.inference.ParameterSample(
            self.names, self.bounds, self.values, self.log_weights, self.groups
        )
        return self

    @classmethod
    def equally_weighted(
        cls,
        names: list[str],
        bounds: list[tuple[float, float]],
        values: list[list[float]],
        method: SampleMethod,
        groups: list[int] | None = None,
    ) -> Self:
        """
        A sample whose draws have equal weights.

        Parameters
        ----------
        names : list[str]
            One name per column.
        bounds : list[tuple[float, float]]
            The bounds of each parameter.
        values : list[list[float]]
            One row per draw.
        method : SampleMethod
            What made the sample.
        groups : list[int] | None, optional
            The group of each draw.

        Returns
        -------
        ParameterSample
            The sample.
        """
        return cls(
            names=names,
            bounds=bounds,
            values=values,
            log_weights=[0.0] * len(values),
            groups=groups,
            method=method,
        )

    @property
    def weights(self) -> list[float]:
        """The weight of each draw; they sum to 1."""
        return self._rust.weights

    def __len__(self) -> int:
        """The number of draws."""
        return len(self.values)

    def draws(self) -> Iterator[dict[str, float]]:
        """
        Every draw, as a mapping of parameter name to value.

        Yields
        ------
        dict[str, float]
            One draw, in the order of `values`.
        """
        for row in self.values:
            yield dict(zip(self.names, row, strict=True))

    def mean(self) -> dict[str, float]:
        """
        The weighted mean of each parameter.

        Returns
        -------
        dict[str, float]
            One mean per name.
        """
        return dict(zip(self.names, self._rust.mean(), strict=True))

    def covariance(self) -> list[list[float]]:
        """
        The weighted covariance matrix, in the order of `names`.

        With weights ``w`` that sum to 1 it is
        ``Σ wᵢ(xᵢ − x̄)(yᵢ − ȳ) / (1 − Σ wᵢ²)``: with equal weights, the
        sample covariance. It is NaN for a sample of one effective draw.

        Returns
        -------
        list[list[float]]
            One row per name.
        """
        return self._rust.covariance()

    def correlation(self) -> list[list[float]]:
        """
        The weighted correlation matrix, in the order of `names`.

        Returns
        -------
        list[list[float]]
            One row per name; NaN for a parameter with no spread.
        """
        return self._rust.correlation()

    def quantiles(self, levels: list[float]) -> dict[str, list[float]]:
        """
        The weighted quantiles of each parameter.

        The quantile is the inverse of the weighted empirical distribution,
        interpolated between the midpoints of the cumulative weights; with
        equal weights it is NumPy's ``method="hazen"``.

        Parameters
        ----------
        levels : list[float]
            Quantile levels, each between 0 and 1.

        Returns
        -------
        dict[str, list[float]]
            One quantile per level for each name.

        Raises
        ------
        ValueError
            If a level is not between 0 and 1.
        """
        rows = self._rust.quantiles([float(level) for level in levels])
        return {
            name: [row[column] for row in rows]
            for column, name in enumerate(self.names)
        }

    def interval(self, confidence_level: float) -> dict[str, tuple[float, float]]:
        """
        The central interval of each parameter.

        Parameters
        ----------
        confidence_level : float
            Strictly between 0 and 1, such as 0.95.

        Returns
        -------
        dict[str, tuple[float, float]]
            ``(lower, upper)`` for each name: the quantiles
            ``(1 − c)/2`` and ``(1 + c)/2``.

        Raises
        ------
        ValueError
            If the confidence level is not strictly between 0 and 1.
        """
        lower, upper = self._rust.interval(confidence_level)
        return {
            name: (low, high)
            for name, low, high in zip(self.names, lower, upper, strict=True)
        }

    @property
    def effective_size(self) -> float:
        """
        Kish's effective sample size, ``1 / Σ wᵢ²``: between 1 and the number
        of draws; equal weights give the number of draws.
        """
        return self._rust.effective_size()

    def resample(self, count: int, seed: int) -> "ParameterSample":
        """
        Draw `count` draws with replacement, each with probability its weight.

        Parameters
        ----------
        count : int
            Number of draws.
        seed : int
            Seed of the generator.

        Returns
        -------
        ParameterSample
            A sample of equal weights.
        """
        indices = self._rust.resample(count, seed)
        return ParameterSample.equally_weighted(
            self.names,
            self.bounds,
            [self.values[i] for i in indices],
            SampleMethod(
                name="resample",
                settings={"of": self.method.name, "count": count, "seed": seed},
            ),
        )

    def summary(self, confidence_level: float) -> dict[str, ParameterSummary]:
        """
        Weighted statistics of each parameter, for a report.

        Parameters
        ----------
        confidence_level : float
            Confidence level of the interval, strictly between 0 and 1.

        Returns
        -------
        dict[str, ParameterSummary]
            One summary per name.

        Raises
        ------
        ValueError
            If the confidence level is not strictly between 0 and 1.
        """
        lower, upper = self._rust.interval(confidence_level)
        quantiles = self.quantiles([0.0, 0.5, 1.0])
        mean = self.mean()
        covariance = self.covariance()
        return {
            name: ParameterSummary(
                mean=mean[name],
                median=quantiles[name][1],
                std=covariance[index][index] ** 0.5,
                lower=lower[index],
                upper=upper[index],
                min=quantiles[name][0],
                max=quantiles[name][2],
            )
            for index, name in enumerate(self.names)
        }
