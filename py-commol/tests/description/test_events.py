"""The dynamics of a stochastic model: an event table, one row per flow."""

import pytest

from commol._model.description import Table, Text
from commol._model.model import Model
from commol.engines import StochasticEventsSettings
from commol.engines.engine import EngineName
from tests.support import STOCHASTIC_ENGINES, abc_builder, build_stochastic


def _model(engine: EngineName, settings=None) -> Model:
    """A → B → C with an inflow into A, an outflow from C and a pair A + B → C."""
    builder = abc_builder(ab="k1", bc="k2 * B")
    builder.add_transition(id="in", source=[], target=["A"], rate="2")
    builder.add_transition(id="out", source=["C"], target=[], rate="k2")
    builder.add_transition(
        id="pair", source=["A", "B"], target=["C"], rate="k1 * A * B / total"
    )
    return build_stochastic(builder, engine, settings)


def _events(model: Model) -> Table:
    blocks = model.describe().section("dynamics").blocks
    return next(
        block
        for block in blocks
        if isinstance(block, Table) and block.heading == "Events"
    )


class TestEventTable:
    def test_a_continuous_engine_gives_each_flow_its_propensity(self) -> None:
        table = _events(_model("StochasticEvents"))

        assert table.columns == ["Transition", "Change", "Propensity"]
        assert table.rows == [
            ["ab", "A -1, B +1", "k1 * A"],
            ["bc", "B -1, C +1", "k2 * B"],
            ["in", "A +1", "2"],
            ["out", "C -1", "k2 * C"],
            ["pair", "A -1, C +1, B -1", "k1 * A * B / total"],
        ]

    def test_a_step_engine_gives_each_flow_its_draw(self) -> None:
        table = _events(_model("StochasticSteps"))

        assert table.columns == ["Transition", "Change", "Draw"]
        assert [row[2] for row in table.rows] == [
            "Binomial(A, k1)",
            "Poisson(k2 * B), at most B",
            "Poisson(2)",
            "Binomial(C, k2)",
            "Poisson(k1 * A * B / total), at most A, B",
        ]

    @pytest.mark.parametrize("engine", STOCHASTIC_ENGINES)
    def test_a_stochastic_model_has_no_equations(self, engine: EngineName) -> None:
        blocks = _model(engine).describe().section("dynamics").blocks

        assert all(block.heading != "Expanded form" for block in blocks)

    def test_the_schedule_says_the_method_and_its_settings(self) -> None:
        settings = StochasticEventsSettings(method="tau_leaping", tau=0.5)
        schedule = _model("StochasticEvents", settings).describe().section("schedule")

        text, table = schedule.blocks
        assert isinstance(text, Text)
        assert "one at a time, at random times" in text.text
        assert isinstance(table, Table)
        assert ["method", "tau_leaping"] in table.rows
        assert ["tau", 0.5] in table.rows
        assert ["rounding", "none"] in table.rows
