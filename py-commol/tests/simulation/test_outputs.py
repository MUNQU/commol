"""Tests for resolving simulation outputs by bin and accumulator."""

import pytest

from commol import Simulation
from tests.support import a_to_b_builder, build, override_builder


def _model(
    *,
    accumulator_id: str = "events",
    conditional: bool = False,
    stratified: bool = True,
) -> Simulation:
    stratifications = {"group": {"group1": 0.6, "group2": 0.4}}
    if conditional:
        stratifications["sub"] = {"s1": 0.5, "s2": 0.5}
    builder = a_to_b_builder(
        parameters={"k1": 0.1},
        rate="k1",
        accumulator=accumulator_id,
        stratifications=stratifications if stratified else None,
        conditions={"sub": {"group": "group1"}} if conditional else None,
    )
    return Simulation(build(builder))


class TestOutputsBySource:
    def test_expansion_matches_the_engine_output_names(self) -> None:
        simulation = _model()

        outputs = simulation.model_definition.structure.outputs_by_source(
            simulation.model_definition
        )
        derived = [name for names in outputs.values() for name in names]

        assert derived == simulation.simulation_outputs

    def test_expansion_matches_the_engine_under_conditional_stratification(
        self,
    ) -> None:
        simulation = _model(conditional=True)

        outputs = simulation.model_definition.structure.outputs_by_source(
            simulation.model_definition
        )
        derived = [name for names in outputs.values() for name in names]

        assert derived == simulation.simulation_outputs

    def test_unstratified_source_yields_its_own_id(self) -> None:
        simulation = _model(stratified=False)

        assert simulation.outputs_for("A") == ["A"]
        assert simulation.outputs_for("events") == ["events"]

    def test_conditional_stratification_gives_ragged_names(self) -> None:
        simulation = _model(conditional=True)

        assert simulation.outputs_for("A") == ["A_group1_s1", "A_group1_s2", "A_group2"]

    def test_accumulators_expand_like_bins(self) -> None:
        simulation = _model()

        assert simulation.outputs_for("events") == ["events_group1", "events_group2"]

    def test_unknown_source_is_rejected(self) -> None:
        simulation = _model()

        with pytest.raises(KeyError, match="not a bin or accumulator"):
            simulation.outputs_for("A_group1")

    def test_group_outputs_keeps_the_requested_order(self) -> None:
        simulation = _model()

        grouped = simulation.group_outputs(["B", "A"])

        assert list(grouped) == ["B", "A"]
        assert grouped["A"] == ["A_group1", "A_group2"]


class TestSourceIdSharingAPrefix:
    """An accumulator id may begin with a bin id followed by an underscore."""

    def test_bin_outputs_exclude_the_accumulator_that_shares_its_prefix(self) -> None:
        simulation = _model(accumulator_id="A_events")

        assert simulation.outputs_for("A") == ["A_group1", "A_group2"]
        assert simulation.outputs_for("A_events") == [
            "A_events_group1",
            "A_events_group2",
        ]

    def test_total_excludes_the_accumulator_that_shares_its_prefix(self) -> None:
        simulation = _model(accumulator_id="A_events")
        results = simulation.run(20)

        prefix_matched = sum(
            results[name][-1]
            for name in results
            if name == "A" or name.startswith("A_")
        )
        total = simulation.total_series(results, ["A"])[-1]

        assert total == pytest.approx(
            sum(results[n][-1] for n in ("A_group1", "A_group2"))
        )
        assert total < prefix_matched


class TestTotalSeries:
    def test_sums_every_output_of_the_given_sources(self) -> None:
        simulation = _model()
        results = simulation.run(10)

        total = simulation.total_series(results, ["A", "B"])

        assert len(total) == 11
        assert total[0] == pytest.approx(1000.0)
        assert all(value == pytest.approx(1000.0) for value in total)

    def test_matches_a_hand_summed_reference(self) -> None:
        simulation = _model()
        results = simulation.run(10)

        total = simulation.total_series(results, ["A"])
        expected = [
            a + b for a, b in zip(results["A_group1"], results["A_group2"], strict=True)
        ]

        assert total == pytest.approx(expected)

    def test_no_sources_gives_an_empty_series(self) -> None:
        simulation = _model()
        results = simulation.run(10)

        assert simulation.total_series(results, []) == []

    def test_missing_output_series_is_rejected(self) -> None:
        simulation = _model()
        results = simulation.run(10)
        del results["A_group2"]

        with pytest.raises(KeyError, match="Missing output series"):
            simulation.total_series(results, ["A"])

    def test_an_output_name_stands_for_itself(self) -> None:
        simulation = _model()
        results = simulation.run(10)

        total = simulation.total_series(results, ["A_group1", "B"])
        expected = [
            a1 + b1 + b2
            for a1, b1, b2 in zip(
                results["A_group1"],
                results["B_group1"],
                results["B_group2"],
                strict=True,
            )
        ]

        assert total == pytest.approx(expected)

    def test_unknown_id_is_rejected_with_what_it_could_be(self) -> None:
        simulation = _model()
        results = simulation.run(10)

        with pytest.raises(KeyError, match="not a bin, accumulator or output") as error:
            simulation.total_series(results, ["nope"])

        message = str(error.value)
        assert "Bins and accumulators: ['A', 'B', 'events']" in message
        assert "Outputs: ['A_group1', 'A_group2', 'B_group1'" in message
        assert "'events_group2']" in message


class TestSubgroupTotal:
    """
    Subgroup totals derived from fractions must match the state the engine
    builds, which multiplies bin and stratification fractions the same way.
    """

    def test_no_categories_is_the_total(self) -> None:
        simulation = _model()

        assert simulation.model_definition.structure.subgroup_total(
            simulation.model_definition.process
        ) == pytest.approx(1000.0)

    def test_one_category_scales_by_its_fraction(self) -> None:
        simulation = _model()

        assert simulation.model_definition.structure.subgroup_total(
            simulation.model_definition.process, ["group1"]
        ) == pytest.approx(600.0)

    def test_matches_the_engine_for_an_unconditional_stratification(self) -> None:
        simulation = _model()
        results = simulation.run(0)
        model = simulation.model_definition

        for category in ("group1", "group2"):
            engine_total = sum(
                results[name][0]
                for name in simulation.simulation_outputs
                if name.split("_")[1:2] == [category]
            )
            assert model.structure.subgroup_total(
                model.process, [category]
            ) == pytest.approx(engine_total)

    def test_matches_the_engine_for_a_nested_conditional_subgroup(self) -> None:
        simulation = _model(conditional=True)
        results = simulation.run(0)
        model = simulation.model_definition

        # `sub` applies only under group1, so its fractions are relative to it.
        engine_total = sum(
            results[name][0]
            for name in simulation.simulation_outputs
            if name.startswith(("A_group1_s1", "B_group1_s1"))
        )

        assert model.structure.subgroup_total(
            model.process, ["group1", "s1"]
        ) == pytest.approx(engine_total)

    def test_unknown_category_is_rejected(self) -> None:
        simulation = _model()

        with pytest.raises(ValueError, match="Category 'nope' not found"):
            simulation.model_definition.structure.subgroup_total(
                simulation.model_definition.process, ["nope"]
            )

    def test_overrides_are_folded_into_each_subgroup(self) -> None:
        model = build(override_builder(300))

        # group1 selects A_g1 (the override of 300) and B_g1 (0).
        assert model.structure.subgroup_total(model.process, ("g1",)) == pytest.approx(
            300.0
        )
        # group2 selects A_g2 (500) and B_g2 (0).
        assert model.structure.subgroup_total(model.process, ("g2",)) == pytest.approx(
            500.0
        )

    def test_a_single_overridden_compartment_sets_its_subgroup(self) -> None:
        model = build(
            a_to_b_builder(
                rate="k * A",
                stratifications={"group": {"group1": 1.0, "group2": 0.0}},
                compartment_amounts={"A_group1": 200},
            )
        )

        assert model.structure.subgroup_total(
            model.process, ("group1",)
        ) == pytest.approx(200.0)


class TestConditioningCategories:
    def test_unconditional_stratification_has_none(self) -> None:
        simulation = _model()

        assert (
            simulation.model_definition.structure.get_conditioning_categories(
                simulation.model_definition.process, "group"
            )
            == ()
        )

    def test_conditional_stratification_reports_its_condition(self) -> None:
        simulation = _model(conditional=True)

        assert simulation.model_definition.structure.get_conditioning_categories(
            simulation.model_definition.process, "sub"
        ) == ("group1",)

    def test_unknown_stratification_is_rejected(self) -> None:
        simulation = _model()

        with pytest.raises(ValueError, match="Stratification 'nope' not found"):
            simulation.model_definition.structure.get_conditioning_categories(
                simulation.model_definition.process, "nope"
            )
