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
from commol._problem.problem import (
    CalibrationProblem,
    calibrated_values,
    declared_values,
)

if TYPE_CHECKING:
    from commol._model.model import Model


class ParameterEntry(TypedDict):
    """Reported value of one parameter."""

    value: float
    calibrated: bool
    spread: NotRequired[EntrySpread]


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
    calibrated: bool,
    statistics: Mapping[str, EntrySummary] | None,
) -> ParameterEntry:
    """One reported parameter, with the spread of its calibrated value when
    there is one."""
    entry: ParameterEntry = {"value": value, "calibrated": calibrated}
    spread = entry_spread(entry_id, statistics) if calibrated else None
    if spread is not None:
        entry["spread"] = spread
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
    problem: CalibrationProblem,
    sample: SummarizedSample | None,
    confidence_level: float | None,
) -> AppliedParametersReport:
    """
    Every parameter of a calibrated model, and the values of its kind.

    The block of `Model.applied_parameters`, which documents it. A parameter
    is calibrated when the problem declares a ``parameter`` of its id and
    `result` holds its value; the structure reads the values of its kind
    (`ModelStructure.report_sections`).
    """
    values = calibrated_values(result)
    calibrated = declared_values(problem, values)
    parameter_statistics = _sample_statistics(sample, confidence_level)
    parameter_ids = {
        parameter.id
        for parameter in calibrated
        if parameter.parameter_type == "parameter"
    }

    parameters: dict[str, ParameterEntry] = {
        parameter.id: _parameter_entry(
            parameter.id,
            parameter.value,
            parameter.id in parameter_ids,
            parameter_statistics,
        )
        for parameter in model.parameters
        if isinstance(parameter.value, (int, float))
    }
    # Factor and noise values describe the observations, so the model holds
    # none of them.
    for parameter in calibrated:
        if parameter.parameter_type in ("factor", "noise"):
            parameters[parameter.id] = _parameter_entry(
                parameter.id, values[parameter.id], True, parameter_statistics
            )

    # `confidence_level` leads the report, so it is written first.
    report: AppliedParametersReport = {"parameters": parameters}
    if confidence_level is not None:
        report = {"confidence_level": confidence_level, "parameters": parameters}
    report.update(
        model.structure.report_sections(model, problem, values, parameter_statistics)
    )
    return report
