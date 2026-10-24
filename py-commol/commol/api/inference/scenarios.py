"""A sample run under scenarios, and the paired comparison of their draws."""

from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from commol.api.inference.predictive import Predictive, PredictiveSource
from commol.commol_rs import _commol_rs as commol_rs
from commol.context.inference.sample import ParameterSample

if TYPE_CHECKING:
    from commol.context.inference.quantities import QuantityBase
    from commol.context.parameter import ParameterUpdate


class ScenarioSummary(BaseModel):
    """
    The weighted distribution of one quantity across the runs of a scenario,
    or of its difference between two scenarios.

    Attributes
    ----------
    label : str
        The quantity, as `expand_quantities` labels it.
    confidence_level : float
        The level of the central interval.
    mean : float
        The weighted mean.
    median : float
        The weighted median.
    lower : float | None
        The quantile ``(1 − c)/2``; None when there is one run.
    upper : float | None
        The quantile ``(1 + c)/2``; None when there is one run.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    confidence_level: float
    mean: float
    median: float
    lower: float | None
    upper: float | None


class Scenarios(Mapping[str, Predictive]):
    """
    Every draw of a sample run under each scenario, with the draws paired.

    A scenario is a set of parameter values that override those of the model
    of the source; the source and its model stay unchanged. Each scenario
    holds the `Predictive` of the sample on a copy of the model with its
    values, built from the source with the seed of the source
    (`with_model`). So run ``k`` of every scenario is the same draw, and with
    a stochastic engine it runs with the same seed: common random numbers,
    whose noise cancels in a difference.

    `Scenarios` is a mapping from the name of a scenario to its `Predictive`,
    so ``plot_compared(scenarios)`` draws them on the same panels.
    """

    def __init__(
        self,
        source: PredictiveSource,
        sample: ParameterSample,
        scenarios: Mapping[str, Mapping[str, "ParameterUpdate"]],
        positions: Sequence[float],
    ) -> None:
        """
        Run every draw of `sample` under each scenario, at `positions`.

        Parameters
        ----------
        source : PredictiveSource
            The calibration problem the sample is a sample of (`Calibrator`),
            or the sensitivity analysis of its inputs (`SensitivityAnalysis`).
        sample : ParameterSample
            Its parameters, in the order of the problem or the analysis.
        scenarios : Mapping[str, Mapping[str, ParameterUpdate]]
            Each scenario by name: the value of each parameter it overrides,
            any value `Model.update_parameters` takes, such as a number, a
            formula or the points of a series. An empty mapping keeps the
            model as it is.
        positions : Sequence[float]
            Positions to report at, sorted and unique. A comparison reads the
            runs here, so they hold ``quantity_schedule(quantities)``.

        Raises
        ------
        ValueError
            Before any draw runs: if there is no scenario, a scenario sets a
            parameter of the sample or one the model does not have (the
            message names the scenario and the parameter), or a value does
            not fit the model. Then as `Predictive` raises.
        InfeasibleStateError
            If a draw leaves the values its model allows, as `Predictive`
            raises.
        """
        if not scenarios:
            raise ValueError("Give at least one scenario.")
        model = source.simulation.model_definition
        models = {}
        for name, overrides in scenarios.items():
            held = [parameter for parameter in overrides if parameter in sample.names]
            if held:
                raise ValueError(
                    f"Scenario '{name}' sets {held}, which the sample holds: each "
                    "draw sets them. A scenario overrides only parameters the "
                    "sample does not hold."
                )
            scenario_model = model.model_copy(deep=True)
            try:
                scenario_model.update_parameters(overrides)
            except ValueError as error:
                raise ValueError(f"Scenario '{name}': {error}") from error
            models[name] = scenario_model
        self.sample: ParameterSample = sample
        self.positions: list[float] = [float(position) for position in positions]
        self._predictives: dict[str, Predictive] = {
            name: Predictive(source.with_model(scenario_model), sample, positions)
            for name, scenario_model in models.items()
        }

    def __getitem__(self, name: str) -> Predictive:
        """
        The predictive of one scenario.

        Parameters
        ----------
        name : str
            The name of the scenario.

        Returns
        -------
        Predictive
            The runs of every draw under the scenario.

        Raises
        ------
        KeyError
            If there is no scenario of that name; the message lists the names.
        """
        if name not in self._predictives:
            raise KeyError(
                f"No scenario '{name}'. Scenarios: {list(self._predictives)}."
            )
        return self._predictives[name]

    def __iter__(self) -> Iterator[str]:
        """The names of the scenarios, in the order they were given."""
        return iter(self._predictives)

    def __len__(self) -> int:
        """The number of scenarios."""
        return len(self._predictives)

    @property
    def weights(self) -> list[float]:
        """The weight of each run, the same in every scenario; they sum to 1."""
        return next(iter(self._predictives.values())).weights

    def summary(
        self,
        name: str,
        quantities: "Sequence[QuantityBase]",
        confidence_level: float,
    ) -> list[ScenarioSummary]:
        """
        The weighted distribution of each quantity across the runs of one
        scenario.

        Parameters
        ----------
        name : str
            The scenario.
        quantities : Sequence[Quantity]
            The quantities; a `Trajectory` gives one scalar per position.
        confidence_level : float
            Strictly between 0 and 1.

        Returns
        -------
        list[ScenarioSummary]
            One summary per label of the quantities, in their order. With one
            run, the value of that run and no interval.

        Raises
        ------
        KeyError
            If there is no scenario of that name.
        ValueError
            If the confidence level is not strictly between 0 and 1, a
            quantity does not fit the runs (as `Predictive.quantities`
            raises), or it is NaN in a run.
        """
        return _summaries(
            self[name].quantities(quantities), self.weights, confidence_level
        )

    def compare(
        self,
        name: str,
        baseline: str,
        quantities: "Sequence[QuantityBase]",
        confidence_level: float,
    ) -> list[ScenarioSummary]:
        """
        The weighted distribution of the difference of each quantity between
        two scenarios, taken run by run.

        The difference ``name − baseline`` is taken within each run, where
        both scenarios have the same draw and seed, and then reduced across
        the runs with their weights. So its interval is the interval of the
        paired differences, not the difference of two intervals.

        Parameters
        ----------
        name : str
            The scenario compared.
        baseline : str
            The scenario subtracted.
        quantities : Sequence[Quantity]
            The quantities; a `Trajectory` gives one scalar per position, each
            compared at its position.
        confidence_level : float
            Strictly between 0 and 1.

        Returns
        -------
        list[ScenarioSummary]
            One summary per label of the quantities, in their order. With one
            run, the difference of that run and no interval.

        Raises
        ------
        KeyError
            If there is no scenario of either name.
        ValueError
            If the confidence level is not strictly between 0 and 1, a
            quantity does not fit the runs (as `Predictive.quantities`
            raises), or it is NaN in a run of either scenario.
        """
        compared = self[name].quantities(quantities)
        subtracted = self[baseline].quantities(quantities)
        differences = [
            {label: value - other[label] for label, value in run.items()}
            for run, other in zip(compared, subtracted, strict=True)
        ]
        return _summaries(differences, self.weights, confidence_level)


def _summaries(
    runs: list[dict[str, float]], weights: list[float], confidence_level: float
) -> list[ScenarioSummary]:
    """The weighted mean, median and central interval of each label of `runs`."""
    levels = [0.5, *commol_rs.inference.interval_levels(confidence_level)]
    one_run = len(runs) == 1
    summaries = []
    for label in runs[0]:
        column = [run[label] for run in runs]
        try:
            median, lower, upper = commol_rs.inference.weighted_quantiles(
                column, weights, levels
            )
        except ValueError as error:
            raise ValueError(
                f"Quantity '{label}', in the runs counted from 0: {error}"
            ) from error
        summaries.append(
            ScenarioSummary(
                label=label,
                confidence_level=confidence_level,
                mean=commol_rs.inference.weighted_mean(column, weights),
                median=median,
                lower=None if one_run else lower,
                upper=None if one_run else upper,
            )
        )
    return summaries
