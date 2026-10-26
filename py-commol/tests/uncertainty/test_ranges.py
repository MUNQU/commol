"""Every setting with a range rule gets the message of the engine when its
object is built with a value outside the range.

The cases come from the rules of the engine (``commol_rs.core.range_rules``):
each gives an object, a setting, a value outside its range and the message of
the rule. A count also gets ``-1``, which has the same wording.
"""

import math
import re
from collections.abc import Callable

import pytest
from pydantic import BaseModel

from commol import (
    Increments,
    AxisPattern,
    NormalObservations,
    ParametricBootstrap,
    Simulation,
    StudentTObservations,
)
from commol.api.simulation import ReplicateRuns
from commol.commol_rs import _commol_rs as commol_rs
from tests.support import a_to_b_builder, build_stochastic
from tests.support.data_objects import EXAMPLES


def replicate_runs() -> ReplicateRuns:
    """Two replicates of a stochastic model A -> B."""
    simulation = Simulation(
        build_stochastic(a_to_b_builder(amounts={"A": 10, "B": 0}), "StochasticSteps")
    )
    return ReplicateRuns(simulation, 2, None)


def pattern(name: str) -> Callable[[], BaseModel]:
    """A valid axis pattern of the private class ``name``."""
    patterns = {
        "_PulseAxisPattern": lambda: AxisPattern.pulse(at=1, value=1.0),
        "_PulsesAxisPattern": lambda: AxisPattern.pulses(at=[1, 2], value=1.0),
        "_PeriodicAxisPattern": lambda: AxisPattern.periodic(period=2, value=1.0),
        "_IntervalAxisPattern": lambda: AxisPattern.interval(
            start=0.0, end=1.0, value=1.0
        ),
        "_GaussianPulseAxisPattern": lambda: AxisPattern.gaussian_pulse(
            center=1.0, width=1.0, peak=1.0
        ),
        "_LinearRampAxisPattern": lambda: AxisPattern.linear_ramp(
            start=0.0, end=1.0, start_value=0.0, end_value=1.0
        ),
    }
    return patterns[name]


VALID: dict[str, Callable[[], object]] = {
    **{
        name: pattern(name)
        for name in (
            "_PulseAxisPattern",
            "_PulsesAxisPattern",
            "_PeriodicAxisPattern",
            "_IntervalAxisPattern",
            "_GaussianPulseAxisPattern",
            "_LinearRampAxisPattern",
        )
    },
    **EXAMPLES,
    "Increments": lambda: Increments(outputs=["B"], window=1.0),
    "ReplicateRuns": replicate_runs,
}
"""A valid object of each object of the engine that has range rules."""


def cases() -> list[tuple[str, str, object, str]]:
    """Each rule with its value outside the range, and ``-1`` for a count."""
    found = []
    for owner, name, outside, message in commol_rs.core.range_rules():
        found.append((owner, name, outside, message))
        if isinstance(outside, int) and outside != -1:
            found.append((owner, name, -1, message.replace(f"got {outside}", "got -1")))
    return found


def outside_settings(model: object, name: str, outside: object) -> dict:
    """The settings of ``model`` (the fields of a pydantic model, or the
    attributes of another object) with ``name`` outside its range: the value,
    or a list of it for a setting that is a list."""
    fields = type(model).model_fields if isinstance(model, BaseModel) else vars(model)
    settings = {field: getattr(model, field) for field in fields}
    current = settings[name]
    settings[name] = [outside] if isinstance(current, list) else outside
    return settings


@pytest.mark.parametrize(
    ("owner", "name", "outside", "message"),
    cases(),
    ids=lambda value: str(value),
)
def test_a_setting_outside_its_range_gets_the_message_of_the_engine(
    owner: str, name: str, outside: object, message: str
) -> None:
    valid = VALID[owner]()
    settings = outside_settings(valid, name, outside)

    with pytest.raises(ValueError, match=re.escape(message)):
        type(valid)(**settings)


@pytest.mark.parametrize("name", ["NaN", "Infinity", "-Infinity"])
def test_a_parameter_named_like_a_number_is_not_a_number(name: str) -> None:
    assert NormalObservations(sd=name).sd == name
    assert StudentTObservations(df=name, scale=name).scale == name


@pytest.mark.parametrize(
    ("value", "shown"), [(math.nan, "NaN"), (math.inf, "inf"), (-math.inf, "-inf")]
)
def test_a_number_that_is_not_finite_gets_the_message_of_the_engine(
    value: float, shown: str
) -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(f"sd is a finite number greater than 0, got {shown}"),
    ):
        NormalObservations(sd=value)


@pytest.mark.parametrize(
    ("count", "shown"),
    [(10**40, "1" + "0" * 40), (10**400, "inf"), (-(10**400), "-inf")],
)
def test_a_count_beyond_the_whole_numbers_of_the_engine_is_rejected(
    count: int, shown: str
) -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(f"replicates is a whole number of at least 1, got {shown}"),
    ):
        ParametricBootstrap(replicates=count)


def test_each_number_of_a_tuple_is_checked() -> None:
    with pytest.raises(
        ValueError, match=re.escape("at is a finite number of at least 0, got -1")
    ):
        commol_rs.core.check_ranges("CalibrationConstraint", {"at": (1.0, -1.0)})
