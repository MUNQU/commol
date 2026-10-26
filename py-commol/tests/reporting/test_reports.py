"""Each report is a method of the object whose data it describes: a run, the
replicates of a stochastic engine, a predictive, a model and an
identifiability analysis."""

import math
import pickle
from collections.abc import Sequence

import numpy as np
import pytest

import commol
import commol.api
from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    ParameterSample,
    Predictive,
    SampleMethod,
    Simulation,
    member_statistics,
)
from commol.context.inference.identifiability import (
    IdentifiabilityReport,
)
from commol.api.results import (
    SimulationResults,
)
from commol.api import reporting
from commol.context.inference import identifiability
from commol.api.reporting import _ReportLeaf, SimulationReport
from commol.context.inference.identifiability import (
    ParameterVerdicts,
    PracticalVerdict,
    ProfileSection,
    StartsSection,
    StructuralVerdict,
)
from tests.support import a_to_b_builder, bounded, build, build_stochastic

WINDOW = 5
WINDOWS = 4
ENDS = [5, 10, 15, 20]
SERIES = ["A_g1", "A_g2", "B_g1", "B_g2"]
EVENTS = ["events_g1", "events_g2"]
OBSERVED = [
    ObservedDataPoint(at=5, output="events", value=1.0),
    ObservedDataPoint(at=15, output="events", value=3.0, weight=2.0),
]
OBSERVED_BLOCK = {
    "events": {"values": [1.0, None, 3.0, None], "weights": [1.0, None, 2.0, None]}
}

type Nested = dict[str, "Nested"] | _ReportLeaf | list[float | None]


def _builder(k: float | None = 0.1):
    """A → B split 40/60 by ``g``, counted by the accumulator ``events``."""
    return a_to_b_builder(
        parameters={"k": k},
        stratifications={"g": {"g1": 0.4, "g2": 0.6}},
        accumulator="events",
    )


def _key_order(block: Nested) -> list[object]:
    """The keys of every nested dictionary, in order."""
    if isinstance(block, dict):
        return [(key, _key_order(value)) for key, value in block.items()]
    return []


def _assert_same_block(report: SimulationReport, expected: Nested) -> None:
    assert report == expected
    assert _key_order(report) == _key_order(expected)


def _run_block(run: SimulationResults) -> SimulationReport:
    """The report of a run on integer steps, written from its definition."""

    def windowed(series: Sequence[float]) -> list[float]:
        return [series[end] - series[end - WINDOW] for end in ENDS]

    def total(step: int) -> float:
        return run["events_g1"][step] + run["events_g2"][step]

    return {
        "observed": OBSERVED_BLOCK,
        "series": {name: [run[name][end] for end in ENDS] for name in SERIES},
        "accumulators": {
            "events": {
                "total": run["events_g1"][-1] + run["events_g2"][-1],
                "windowed": {
                    "total": windowed(
                        [total(step) for step in range(len(run.positions))]
                    ),
                    "by_output": {name: windowed(run[name]) for name in EVENTS},
                },
                "cumulative": {
                    "total": [total(end) for end in ENDS],
                    "by_output": {
                        name: [run[name][end] for end in ENDS] for name in EVENTS
                    },
                },
            }
        },
    }


def _reduced(
    blocks: Sequence[Nested], confidence_level: float, weights: list[float] | None
) -> Nested:
    """Each leaf of the blocks of the members, reduced across them."""
    first = blocks[0]
    if isinstance(first, dict):
        return {
            key: _reduced([block[key] for block in blocks], confidence_level, weights)
            for key in first
        }
    members = np.asarray(blocks, dtype=float)
    return member_statistics(members, confidence_level, weights)


def _ensemble_block(
    runs: Sequence[SimulationResults],
    confidence_level: float,
    weights: list[float] | None = None,
) -> Nested:
    """The report of an ensemble: the report of each run, reduced leaf by leaf."""
    blocks = [_run_block(run) for run in runs]
    return {
        "observed": OBSERVED_BLOCK,
        "series": _reduced(
            [block["series"] for block in blocks], confidence_level, weights
        ),
        "accumulators": _reduced(
            [block["accumulators"] for block in blocks], confidence_level, weights
        ),
    }


def _report_run(run: SimulationResults) -> SimulationReport:
    return run.report(
        ENDS, WINDOW, WINDOWS, observed_data=OBSERVED, accumulators=["events"]
    )


def test_the_report_of_a_run_is_a_method_of_its_results() -> None:
    run = Simulation(build(_builder())).run(20)

    _assert_same_block(_report_run(run), _run_block(run))


def test_the_report_of_replicates_is_a_method_of_the_replicates() -> None:
    simulation = Simulation(build_stochastic(_builder(), "StochasticSteps"))
    replicates = simulation.replicates(5, seed=3).run(20)

    report = replicates.report(
        0.9, ENDS, WINDOW, WINDOWS, observed_data=OBSERVED, accumulators=["events"]
    )

    _assert_same_block(report, _ensemble_block(replicates, 0.9))


def test_the_report_of_a_predictive_is_a_method_of_predictive() -> None:
    problem = CalibrationProblem(
        observed_data=[ObservedDataPoint(at=5, output="B_g1", value=20.0)],
        parameters=[bounded("k")],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=1),
    )
    calibrator = Calibrator(Simulation(build(_builder(None))), problem)
    weights = [0.2, 0.3, 0.5]
    sample = ParameterSample(
        names=["k"],
        bounds=[(0.0, 1.0)],
        values=[[0.05], [0.1], [0.2]],
        log_weights=[math.log(weight) for weight in weights],
        method=SampleMethod(name="test"),
    )
    predictive = Predictive(calibrator, sample, range(21))

    report = predictive.report(
        0.9, ENDS, WINDOW, WINDOWS, observed_data=OBSERVED, accumulators=["events"]
    )

    _assert_same_block(
        report, _ensemble_block(predictive.runs(), 0.9, predictive.weights)
    )


def test_the_applied_parameters_are_a_method_of_model() -> None:
    model = build(_builder())
    problem = CalibrationProblem(
        observed_data=OBSERVED,
        parameters=[bounded("k"), bounded("f", kind="factor")],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=1),
    )

    report = model.applied_parameters({"k": 0.1, "f": 0.5}, problem)

    assert report == {
        "parameters": {
            "k": {"value": 0.1, "calibrated": True},
            "f": {"value": 0.5, "calibrated": True},
        },
        "initial_amounts": {
            "A": {"amount": 1000.0, "unit": None, "calibrated": False},
            "B": {"amount": 0.0, "unit": None, "calibrated": False},
        },
        "category_splits": {
            "g1": {"group": [], "fraction": 0.4, "calibrated": False},
            "g2": {"group": [], "fraction": 0.6, "calibrated": False},
        },
    }


def test_the_identifiability_report_is_a_method_of_its_result() -> None:
    report = IdentifiabilityReport(
        parameters=["k", "f"],
        verdicts={
            "k": ParameterVerdicts(
                structural=StructuralVerdict(
                    verdict="in_combination",
                    reasons=["k and f enter as k · f"],
                    combinations=["k · f"],
                ),
                practical=PracticalVerdict(
                    verdict="poorly_determined", reasons=["starts differ"]
                ),
            ),
            "f": ParameterVerdicts(),
        },
        starts=[
            StartsSection(
                parameter_tolerance=0.05,
                loss_tolerance=1e-3,
                near_best=[0, 1],
                spread={"k": 0.6, "f": 0.0},
                flagged=["k"],
            )
        ],
    )

    data = report.report()

    expected = {
        "parameters": {
            "k": {
                "structural": {
                    "verdict": "in_combination",
                    "reasons": ["k and f enter as k · f"],
                    "combinations": ["k · f"],
                },
                "practical": {
                    "verdict": "poorly_determined",
                    "reasons": ["starts differ"],
                },
            },
            "f": {"structural": None, "practical": None},
        },
        "sections": {
            "structural": [],
            "collinearity": [],
            "sample": [],
            "starts": [report.starts[0].model_dump()],
            "profile": [],
        },
    }
    assert data == expected
    assert _key_order(data) == _key_order(expected)


def test_an_identifiability_report_has_a_section_of_each_method() -> None:
    report = IdentifiabilityReport(
        parameters=["k"],
        verdicts={"k": ParameterVerdicts()},
        profile=[
            ProfileSection(
                threshold="chi_square",
                confidence_level=0.95,
                tolerance=1e-6,
                verdicts={"k": "identifiable"},
                intervals={"k": (0.1, 0.4)},
            )
        ],
    )

    sections = report.report()["sections"]

    methods = [
        name
        for name in IdentifiabilityReport.model_fields
        if name not in ("parameters", "verdicts")
    ]
    assert list(sections) == methods
    assert sections["profile"] == [report.profile[0].model_dump()]


@pytest.mark.parametrize(
    "model",
    [
        "PointRank",
        "Combination",
        "StructuralSection",
        "SubsetIndex",
        "CollinearitySection",
        "CorrelatedPair",
        "SampleSection",
        "StartsSection",
        "ProfileSection",
    ],
)
def test_the_data_of_a_section_has_the_fields_of_its_model(model: str) -> None:
    fields = getattr(identifiability, model).model_fields
    data = getattr(reporting, f"{model}Data")

    assert list(data.__annotations__) == list(fields)


def test_reports_are_only_methods() -> None:
    names = {"simulation_report", "applied_parameters_report", "identifiability_report"}

    assert names.isdisjoint(dir(commol))
    assert names.isdisjoint(dir(commol.api))


def test_the_results_know_the_outputs_of_their_accumulators() -> None:
    model = build(_builder())
    run = Simulation(model).run(20)

    accumulator = _report_run(run)["accumulators"]["events"]

    outputs = model.structure.outputs_by_source(model)["events"]
    assert list(accumulator["cumulative"]["by_output"]) == outputs == EVENTS
    assert accumulator["cumulative"]["total"] == [
        sum(run[name][end] for name in outputs) for end in ENDS
    ]


def test_results_built_by_hand_know_no_accumulator() -> None:
    run = SimulationResults([0.0, 5.0], {"A": [1.0, 0.5], "events": [0.0, 0.5]})

    with pytest.raises(ValueError, match="cannot report 'events'"):
        run.report([5.0], WINDOW, 1, accumulators=["events"])


def test_results_with_their_model_pickle() -> None:
    run = Simulation(build(_builder())).run(20)

    loaded = pickle.loads(pickle.dumps(run))

    assert loaded == run
    assert loaded.positions == run.positions
    _assert_same_block(_report_run(loaded), _report_run(run))
