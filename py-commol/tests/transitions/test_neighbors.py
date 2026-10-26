"""The neighbors of a unit: ``neighbors_<bin>`` and ``neighbors`` in a rate.

A rate that reads them and no compartment is proportional, a rate per unit of
its source, so the same process carries over to other structures. Under
``Aggregate`` they resolve to sums of amounts (`Complete`, `Groups`).
"""

import pytest
from pydantic import ValidationError

from commol import (
    Complete,
    Groups,
    Model,
    Simulation,
    StratificationCondition,
    StratifiedRate,
)
from commol._model.engine_state import transition_flows
from commol._model.neighbors import _Neighbors
from tests.support import ENGINE_REL, FactoryBuilder, abc_builder, assert_close, build

GROUPS = {"group": {"g1": 0.3, "g2": 0.7}}
WEIGHTS = {"g1": {"g1": 1.0, "g2": 0.4}, "g2": {"g1": 0.25, "g2": 1.0}}


def _abc(
    rate: str,
    neighbors: _Neighbors | None,
    *,
    stratifications: dict[str, dict[str, float]] | None = None,
) -> FactoryBuilder:
    builder = abc_builder(ab=rate, bc="k2", stratifications=stratifications)
    builder.structure = builder.structure.model_copy(update={"neighbors": neighbors})
    return builder


def _run(model: Model, steps: int = 30) -> dict[str, list[float]]:
    return Simulation(model).run(steps)


class TestComplete:
    def test_the_neighbors_of_every_unit_give_the_bin_total_rate(self) -> None:
        by_neighbors = build(_abc("k1 * neighbors_B / neighbors", Complete()))
        by_hand = build(_abc("k1 * A * B / total", None))

        assert_close(_run(by_neighbors), _run(by_hand), rel=ENGINE_REL)

    def test_a_rate_of_the_neighbors_is_proportional_to_its_source(self) -> None:
        model = build(_abc("k1 * neighbors_B / neighbors", Complete()))

        flow = next(
            flow
            for flow in transition_flows(model.process, model.structure)
            if flow.transition == "ab"
        )

        assert flow.rate == "k1 * (B) / (total)"
        assert flow.proportional_to == "A"


class TestGroups:
    def test_weighted_groups_give_the_rate_written_by_hand(self) -> None:
        by_neighbors = build(
            _abc(
                "k1 * neighbors_B / neighbors",
                Groups("group", weights=WEIGHTS),
                stratifications=GROUPS,
            )
        )
        by_hand_rate = {
            own: (
                f"k1 * A_{own} * ({WEIGHTS[own]['g1']} * B_g1 + "
                f"{WEIGHTS[own]['g2']} * B_g2) / ({WEIGHTS[own]['g1']} * total_g1 + "
                f"{WEIGHTS[own]['g2']} * total_g2)"
            )
            for own in ("g1", "g2")
        }
        by_hand = build(
            abc_builder(ab="0", bc="k2", stratifications=GROUPS).add_transition(
                id="ab_by_hand",
                source=["A"],
                target=["B"],
                stratified_rates=[
                    StratifiedRate(
                        conditions=[
                            StratificationCondition(
                                stratification="group", category=own
                            )
                        ],
                        rate=rate,
                    )
                    for own, rate in by_hand_rate.items()
                ],
            )
        )

        assert_close(_run(by_neighbors), _run(by_hand), rel=ENGINE_REL)

    def test_groups_without_weights_see_their_own_category(self) -> None:
        model = build(_abc("k1 * neighbors_B", Groups("group"), stratifications=GROUPS))

        rates = {
            flow.source: flow.rate
            for flow in transition_flows(model.process, model.structure)
            if flow.transition == "ab"
        }

        assert rates == {"A_g1": "k1 * (B_g1)", "A_g2": "k1 * (B_g2)"}

    def test_groups_of_an_unknown_stratification_name_it(self) -> None:
        with pytest.raises(ValidationError, match="'level'"):
            build(_abc("k1 * neighbors_B", Groups("level"), stratifications=GROUPS))

    def test_groups_take_no_other_field(self) -> None:
        with pytest.raises(TypeError, match="unknown"):
            Groups("group", unknown=1)  # ty: ignore[unknown-argument]

    def test_weights_need_every_pair_of_categories(self) -> None:
        weights = {"g1": {"g1": 1.0}, "g2": {"g1": 0.5, "g2": 1.0}}

        with pytest.raises(ValidationError, match="'g1'"):
            build(
                _abc(
                    "k1 * neighbors_B",
                    Groups("group", weights=weights),
                    stratifications=GROUPS,
                )
            )

    def test_groups_save_and_load_back(self, tmp_path) -> None:
        model = build(
            _abc(
                "k1 * neighbors_B / neighbors",
                Groups("group", weights=WEIGHTS),
                stratifications=GROUPS,
            )
        )
        path = tmp_path / "model.json"
        model.to_json(path)

        loaded = Model.from_json(path)

        assert loaded == model
        assert loaded.model_dump(mode="json")["structure"]["neighbors"] == {
            "kind": "groups",
            "stratification": "group",
            "weights": WEIGHTS,
        }


class TestRequired:
    def test_a_rate_that_reads_the_neighbors_needs_them_and_lists_the_options(
        self,
    ) -> None:
        with pytest.raises(ValidationError) as error:
            build(_abc("k1 * neighbors_B / neighbors", None))

        assert "Complete()" in str(error.value)
        assert "Groups(" in str(error.value)

    def test_neighbors_of_an_id_that_is_not_a_bin_name_it(self) -> None:
        with pytest.raises(ValidationError, match="'D' is not a bin"):
            build(_abc("k1 * neighbors_D", Complete()))

    def test_a_parameter_cannot_be_named_after_the_neighbors(self) -> None:
        builder = _abc("k1 * A", None)
        builder.add_parameter(id="neighbors_A", value=1.0)

        with pytest.raises(ValidationError, match="neighbors_A"):
            build(builder)

    def test_a_formula_parameter_does_not_read_the_neighbors(self) -> None:
        builder = _abc("k1 * A", Complete())
        builder.add_parameter(id="f", value="k1 * neighbors")

        with pytest.raises(ValidationError, match="only a transition rate"):
            build(builder)
