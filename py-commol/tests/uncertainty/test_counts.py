"""Every count and every seed a binding takes, called with no Python object in
between, gets the message of the rule of its setting in the engine when it is
negative, at the path of its field when it is part of the JSON of an object;
and so does a seed of each entry point of the Python API."""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from commol import CalibrationProblem, Simulation
from commol._calibration.calibrator import Calibrator
from commol._data.engine_json import engine_json
from commol.commol_rs import _commol_rs as commol_rs
from commol.distributions import Uniform
from commol.sensitivity import SensitivityAnalysis
from tests.support import (
    a_to_b_builder,
    abc_builder,
    abc_rates,
    build,
    build_stochastic,
    line_calibrator,
)
from tests.support.data_objects import EXAMPLES

if TYPE_CHECKING:
    from commol.commol_rs._commol_rs import ForwardAnalysisProtocol

inference = commol_rs.inference


@dataclass(frozen=True)
class Bindings:
    """The objects whose methods the cases call."""

    calibrator: Calibrator
    analysis: "ForwardAnalysisProtocol"


@pytest.fixture(scope="module")
def bindings() -> Bindings:
    calibrator = line_calibrator([7.4, 8.6, 11.3, 12.8])
    analysis = SensitivityAnalysis(
        Simulation(build(abc_builder())), abc_rates(), seed=1
    )._rust
    return Bindings(calibrator, analysis)


def _problem(b: Bindings):
    return b.calibrator._engine_problem


def _inputs() -> str:
    return engine_json(EXAMPLES["InputSpace"]())


def _uniform():
    return inference.Distribution(engine_json(Uniform()), 0.0, 1.0)


def _sample():
    return inference.ParameterSample(["c"], [(0.0, 1.0)], [[0.5]], [0.0], None)


CASES: dict[str, tuple[str, Callable[[Bindings], object]]] = {
    "num_steps of Engine.run": (
        "num_steps",
        lambda b: b.calibrator.simulation.engine.run(-1),
    ),
    "count of Engine.run_replicates": (
        "count",
        lambda b: b.calibrator.simulation.engine.run_replicates(None, -1, 5),
    ),
    "starts of CalibrationProblem.optimize": (
        "starts",
        lambda b: _problem(b).optimize(-1),
    ),
    "count of ParameterSample.resample": (
        "count",
        lambda _: _sample().resample(-1, 1),
    ),
    "replicates of ForwardAnalysis": (
        "replicates",
        lambda b: inference.ForwardAnalysis(
            b.calibrator.simulation.engine, _inputs(), replicates=-1, seed=1
        ),
    ),
    "degree of chaos_evaluate": (
        "degree",
        lambda _: inference.chaos_evaluate(_inputs(), -1, 1.0, None, [], []),
    ),
    "max_interaction of chaos_evaluate": (
        "max_interaction",
        lambda _: inference.chaos_evaluate(_inputs(), 2, 1.0, -1, [], []),
    ),
    "count of Distribution.sample": (
        "count",
        lambda _: _uniform().sample(-1, 1),
    ),
}
"""Each count a binding takes as a number, called with -1."""


RULE = re.compile(
    r"(?P<setting>\w+) is a whole number (of at least \d+|from 0 to \d+), got -1"
)
"""The message of the rule of a count, for -1."""


@pytest.mark.parametrize("case", CASES)
def test_a_negative_count_gets_the_message_of_its_rule(
    bindings: Bindings, case: str
) -> None:
    setting, call = CASES[case]

    with pytest.raises(ValueError) as error:
        call(bindings)

    message = RULE.fullmatch(str(error.value))
    assert message is not None and message["setting"] == setting, str(error.value)


def test_a_count_above_its_type_names_its_largest_value(bindings: Bindings) -> None:
    with pytest.raises(ValueError) as error:
        bindings.calibrator.simulation.engine.run(2**32)

    assert str(error.value) == (
        f"num_steps is a whole number from 0 to {2**32 - 1}, got {2**32}"
    )


def test_a_count_that_is_not_a_whole_number_is_a_type_error(
    bindings: Bindings,
) -> None:
    with pytest.raises(TypeError):
        _problem(bindings).optimize(1.5)  # ty: ignore[invalid-argument-type]


def json_counts() -> list[tuple[str, str, str]]:
    """Each count of an object whose JSON the engine reads: its object, its
    setting and the message of its rule for -1."""
    objects = set(commol_rs.json.objects())
    return [
        (owner, name, message.replace(f"got {outside}", "got -1"))
        for owner, name, outside, message in commol_rs.core.range_rules()
        if owner in objects and isinstance(outside, int)
    ]


@pytest.mark.parametrize(("owner", "name", "message"), json_counts())
def test_a_negative_count_in_the_json_gets_the_message_of_its_rule_at_its_path(
    owner: str, name: str, message: str
) -> None:
    settings = json.loads(engine_json(EXAMPLES[owner]()))
    settings[name] = [-1] if isinstance(settings[name], list) else -1

    with pytest.raises(ValueError) as error:
        commol_rs.json.as_read(owner, json.dumps(settings))

    assert str(error.value) == f"{name}: {message}"


SEED_RULE = "seed is a whole number from 0 to 18446744073709551615, got {}"
"""The message of the rule of a seed."""


SEEDS: dict[str, Callable[[Bindings], object]] = {
    "Engine.set_seed": lambda b: b.calibrator.simulation.engine.set_seed(-1),
    "Engine.run_replicates": lambda b: b.calibrator.simulation.engine.run_replicates(
        -1, 2, 5
    ),
    "ParameterSample.resample": lambda _: _sample().resample(1, -1),
    "Distribution.sample": lambda _: _uniform().sample(1, -1),
    "ForwardAnalysis": lambda b: inference.ForwardAnalysis(
        b.calibrator.simulation.engine, _inputs(), seed=-1
    ),
}
"""Each binding that takes a seed, called with the seed -1."""


@pytest.mark.parametrize("case", SEEDS)
def test_a_negative_seed_of_a_binding_gets_the_message_of_its_rule(
    bindings: Bindings, case: str
) -> None:
    with pytest.raises(ValueError) as error:
        SEEDS[case](bindings)

    assert str(error.value) == SEED_RULE.format(-1)


def _stochastic() -> Simulation:
    return Simulation(
        build_stochastic(a_to_b_builder(amounts={"A": 10, "B": 0}), "StochasticSteps")
    )


ENTRY_POINTS: dict[str, Callable[[int], object]] = {
    "Simulation.run": lambda seed: _stochastic().run(1, seed=seed),
    "Simulation.run_at": lambda seed: _stochastic().run_at([1.0], seed=seed),
    "Simulation.replicates": lambda seed: _stochastic().replicates(2, seed=seed),
    "SensitivityAnalysis": lambda seed: SensitivityAnalysis(
        Simulation(build(abc_builder())), abc_rates(), seed=seed
    ),
    "CalibrationProblem": lambda seed: CalibrationProblem.model_validate(
        {
            **line_calibrator([7.4, 8.6]).problem.model_dump(
                exclude={"seed", "optimization_config"}
            ),
            "optimization_config": line_calibrator([7.4]).problem.optimization_config,
            "seed": seed,
        }
    ),
}
"""Each entry point of the Python API that takes a seed."""


@pytest.mark.parametrize("seed", [-1, 2**64])
@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_a_seed_outside_its_range_gets_the_message_of_the_engine(
    entry: str, seed: int
) -> None:
    with pytest.raises(ValueError, match=re.escape(SEED_RULE.format(seed))):
        ENTRY_POINTS[entry](seed)
