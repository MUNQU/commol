"""The dynamics of a model of every kind of flow, as each engine describes it."""

import pytest

from tests.support.described import every_flow_builder
from tests.support.description import matches_snapshot


@pytest.mark.parametrize(
    "engine", ["StochasticSteps", "StochasticEvents", "DifferentialEquations"]
)
def test_the_dynamics_match_their_snapshot(engine: str) -> None:
    builder = every_flow_builder()
    model = builder.build(builder.structure, engine=engine)

    rendered = model.describe().section("dynamics").model_dump_json(indent=1) + "\n"

    matches_snapshot(f"dynamics_{engine}.json", rendered)
