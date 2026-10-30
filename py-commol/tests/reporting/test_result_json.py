"""A result writes itself to a JSON file and its class reads it back equal; a
result read back feeds the methods that take a result, and the file of
another result is refused."""

import importlib
import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from commol import CalibrationResult, Simulation
from commol._data.result_model import ResultModel
from commol.identifiability import (
    Collinearity,
    IdentifiabilityReportData,
    StartsIdentifiability,
)
from commol.distributions import Normal
from commol.resampling import BayesianBootstrap, BootstrapResult
from commol.sensitivity import (
    ChaosResult,
    InputSpace,
    PolynomialChaos,
    SensitivityAnalysis,
)
from tests.suite.public_surface import EXPORTS
from tests.support import (
    abc_builder,
    bounded,
    build,
    flow_calibrator,
    line_calibrator,
)
from tests.support.results import EXAMPLES_OF_RESULTS, FLOWS, LINE, QUANTITIES


def _public_results() -> dict[str, type[ResultModel]]:
    """Each subclass of `ResultModel` on the public surface, by name."""
    results = {}
    for module, names in EXPORTS.items():
        for name in names:
            item = getattr(importlib.import_module(module), name)
            if isinstance(item, type) and issubclass(item, ResultModel):
                results[name] = item
    return results


PUBLIC_RESULTS = _public_results()


def _reject_constant(name: str) -> float:
    raise AssertionError(f"{name} is not standard JSON")


def _bootstrap() -> BootstrapResult:
    calibrator = flow_calibrator(FLOWS, windowed=True)
    return calibrator.uncertainty(BayesianBootstrap(replicates=8, block_length=1))


def test_a_calibration_result_reads_back_equal(tmp_path: Path) -> None:
    result = line_calibrator(LINE).run(starts=4)
    path = tmp_path / "fit.json"

    result.to_json(path)
    read = CalibrationResult.from_json(path)

    assert len(result.starts) == 4
    assert read == result
    assert read.starts == result.starts
    assert read.seed == result.seed


@pytest.mark.parametrize("name", sorted(PUBLIC_RESULTS.keys() | EXAMPLES_OF_RESULTS))
def test_every_result_reads_back_equal(name: str, tmp_path: Path) -> None:
    assert name in PUBLIC_RESULTS, f"{name} is no public subclass of ResultModel"
    assert name in EXAMPLES_OF_RESULTS, (
        f"{name} has no example in EXAMPLES_OF_RESULTS (tests/support/results.py)"
    )
    result = EXAMPLES_OF_RESULTS[name]()
    assert type(result) is PUBLIC_RESULTS[name]
    path = tmp_path / f"{name}.json"

    result.to_json(path)
    read = PUBLIC_RESULTS[name].from_json(path)

    assert type(read) is PUBLIC_RESULTS[name]
    assert read == result
    json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)


def test_an_infinite_bound_in_a_nested_model_reads_back_equal(tmp_path: Path) -> None:
    space = InputSpace.independent(
        [bounded("k1", 0.1, 0.5), bounded("k2", 0.0, math.inf)],
        {"k1": Normal(mean=0.3, sd=0.1), "k2": Normal(mean=0.5, sd=0.1)},
    )
    analysis = SensitivityAnalysis(Simulation(build(abc_builder())), space, seed=7)
    result = analysis.run(PolynomialChaos(degree=1, draws=8, quantities=QUANTITIES))
    path = tmp_path / "chaos.json"

    result.to_json(path)
    read = ChaosResult.from_json(path)

    assert read == result
    assert read.input_space.parameters[1].max_bound == math.inf
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["input_space"]["parameters"][1]["max_bound"] == "Infinity"


def test_a_result_read_back_feeds_the_identifiability(tmp_path: Path) -> None:
    calibrator = line_calibrator(LINE)
    result = calibrator.run(starts=3)
    result.to_json(tmp_path / "fit.json")
    read = CalibrationResult.from_json(tmp_path / "fit.json")

    def reports(fit: CalibrationResult) -> list[IdentifiabilityReportData]:
        return [
            calibrator.identifiability(method).report()
            for method in (
                StartsIdentifiability(
                    result=fit, parameter_tolerance=0.05, loss_tolerance=0.01
                ),
                Collinearity(at=fit, max_subset_size=2, threshold=10.0),
            )
        ]

    in_memory = reports(result)
    assert reports(read) == in_memory
    assert in_memory[0]["sections"]["starts"]
    assert in_memory[1]["sections"]["collinearity"]


def test_a_sample_read_back_draws_the_same_members(tmp_path: Path) -> None:
    result = _bootstrap()
    result.to_json(tmp_path / "bootstrap.json")

    read = BootstrapResult.from_json(tmp_path / "bootstrap.json")

    drawn = result.sample.resample(20, seed=11)
    assert read.sample.resample(20, seed=11) == drawn
    assert len({tuple(values) for values in drawn.values}) > 1
    assert read.sample.mean() == result.sample.mean()
    assert read.sample.interval(0.9) == result.sample.interval(0.9)


def test_a_file_of_another_result_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bootstrap.json"
    _bootstrap().to_json(path)

    with pytest.raises(ValidationError) as error:
        CalibrationResult.from_json(path)

    fields = {entry["loc"][0] for entry in error.value.errors()}
    assert error.value.title == "CalibrationResult"
    assert {"best_parameters", "final_loss", "sample"} <= fields
    assert "best_parameters" in str(error.value)
