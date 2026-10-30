"""The report of the parameters a calibration applied to a model
(`Model.applied_parameters`)."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, NotRequired, TypedDict

from commol._fit.result import CalibrationResult
from commol._model.report_entry import (
    EntrySummary,
    EntrySpread,
    SummarizedSample,
    entry_spread,
)
from commol._model.structure import ReportSections
from commol._problem.problem import CalibrationProblem, calibrated_values

if TYPE_CHECKING:
    from commol._model.model import Model


class ParameterEntry(TypedDict):
    """Reported value of one parameter."""

    value: float
    calibrated: bool
    interval: NotRequired[EntrySpread]


class AppliedParametersReport(ReportSections):
    """
    Every parameter of a calibrated model, and the sections of its kind
    (`ReportSections`, which `ModelStructure.report_sections` gives).
    """

    confidence_level: NotRequired[float]
    parameters: dict[str, ParameterEntry]


def _parameter_entry(
    entry_id: str,
    value: float,
    calibrated_ids: set[str],
    statistics: Mapping[str, EntrySummary] | None,
) -> ParameterEntry:
    """One reported parameter, with its interval when there is one."""
    entry: ParameterEntry = {
        "value": value,
        "calibrated": entry_id in calibrated_ids,
    }
    interval = entry_spread(entry_id, statistics)
    if interval is not None:
        entry["interval"] = interval
    return entry


def _sample_statistics(
    sample: SummarizedSample | None,
    confidence_level: float | None,
) -> Mapping[str, EntrySummary] | None:
    """The statistics of each parameter of a sample, at a confidence level."""
    if sample is None:
        return None
    if confidence_level is None:
        raise ValueError("A report of a sample needs a confidence_level.")
    return sample.summary(confidence_level)


def applied_parameters_report(
    model: "Model",
    result: CalibrationResult | Mapping[str, float],
    problem: CalibrationProblem | None,
    sample: SummarizedSample | None,
    confidence_level: float | None,
) -> AppliedParametersReport:
    """
    Every parameter of a calibrated model, and the values of its kind.

    The block of `Model.applied_parameters`, which documents it.
    """
    calibrated = calibrated_values(result)
    parameter_statistics = _sample_statistics(sample, confidence_level)
    calibrated_ids = set(calibrated)

    parameters: dict[str, ParameterEntry] = {
        parameter.id: _parameter_entry(
            parameter.id, parameter.value, calibrated_ids, parameter_statistics
        )
        for parameter in model.parameters
        if isinstance(parameter.value, (int, float))
    }
    for observation_id in _observation_parameter_ids(problem):
        if observation_id in calibrated:
            parameters[observation_id] = _parameter_entry(
                observation_id,
                calibrated[observation_id],
                calibrated_ids,
                parameter_statistics,
            )

    # `confidence_level` leads the report, so it is written first.
    report: AppliedParametersReport = {"parameters": parameters}
    if confidence_level is not None:
        report = {"confidence_level": confidence_level, "parameters": parameters}
    report.update(
        model.structure.report_sections(model, calibrated_ids, parameter_statistics)
    )
    return report


def _observation_parameter_ids(problem: CalibrationProblem | None) -> list[str]:
    """Ids of the problem's factor and noise parameters, which describe the
    observations, in declaration order."""
    if problem is None:
        return []
    return [
        parameter.id
        for parameter in problem.parameters
        if parameter.parameter_type in ("factor", "noise")
    ]
