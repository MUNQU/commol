"""Tests for resolving simulation outputs by bin and accumulator."""

import pytest

from commol import ModelBuilder, Simulation
from commol.api.model_builder import BinFractionDict, StratificationFractionsDict
from commol.constants import ModelTypes


def _model(
    *,
    accumulator_id: str = "events",
    conditional: bool = False,
    stratified: bool = True,
) -> "Simulation":
    builder = (
        ModelBuilder(name="Outputs", version="1.0")
        .add_bin(id="A", name="A")
        .add_bin(id="B", name="B")
        .add_accumulator(id=accumulator_id, name="Events")
        .add_parameter(id="k1", value=0.1)
        .add_transition(
            id="flow",
            source=["A"],
            target=["B"],
            rate="k1",
            accumulators=[accumulator_id],
        )
    )
    bin_fractions: list[BinFractionDict] = [
        {"bin": "A", "fraction": 1.0},
        {"bin": "B", "fraction": 0.0},
    ]
    if not stratified:
        builder.set_initial_conditions(
            population_size=1000, bin_fractions=bin_fractions
        )
        return Simulation(builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value))

    builder.add_stratification(id="group", categories=["group1", "group2"])
    stratification_fractions: list[StratificationFractionsDict] = [
        {
            "stratification": "group",
            "fractions": [
                {"category": "group1", "fraction": 0.6},
                {"category": "group2", "fraction": 0.4},
            ],
        }
    ]
    if conditional:
        builder.add_stratification(
            id="sub",
            categories=["s1", "s2"],
            conditions=[{"stratification": "group", "category": "group1"}],
        )
        stratification_fractions.append(
            {
                "stratification": "sub",
                "fractions": [
                    {"category": "s1", "fraction": 0.5},
                    {"category": "s2", "fraction": 0.5},
                ],
            }
        )
    builder.set_initial_conditions(
        population_size=1000,
        bin_fractions=bin_fractions,
        stratification_fractions=stratification_fractions,
    )
    return Simulation(builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value))


class TestOutputsBySource:
    def test_expansion_matches_the_engine_output_names(self) -> None:
        simulation = _model()

        derived = [
            name
            for names in simulation.model_definition.get_outputs_by_source().values()
            for name in names
        ]

        assert derived == simulation.simulation_outputs

    def test_expansion_matches_the_engine_under_conditional_stratification(
        self,
    ) -> None:
        simulation = _model(conditional=True)

        derived = [
            name
            for names in simulation.model_definition.get_outputs_by_source().values()
            for name in names
        ]

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

    def test_unknown_source_is_rejected(self) -> None:
        simulation = _model()
        results = simulation.run(10)

        with pytest.raises(KeyError, match="not a bin or accumulator"):
            simulation.total_series(results, ["nope"])


class TestSubgroupPopulation:
    """
    Head counts derived from fractions must match the populations the engine
    builds, which multiplies bin and stratification fractions the same way.
    """

    def test_no_categories_is_the_whole_population(self) -> None:
        simulation = _model()

        assert simulation.model_definition.subgroup_population() == pytest.approx(
            1000.0
        )

    def test_one_category_scales_by_its_fraction(self) -> None:
        simulation = _model()

        assert simulation.model_definition.subgroup_population(
            ["group1"]
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
            assert model.subgroup_population([category]) == pytest.approx(engine_total)

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

        assert model.subgroup_population(["group1", "s1"]) == pytest.approx(
            engine_total
        )

    def test_unknown_category_is_rejected(self) -> None:
        simulation = _model()

        with pytest.raises(ValueError, match="Category 'nope' not found"):
            simulation.model_definition.subgroup_population(["nope"])


class TestConditioningCategories:
    def test_unconditional_stratification_has_none(self) -> None:
        simulation = _model()

        assert simulation.model_definition.get_conditioning_categories("group") == ()

    def test_conditional_stratification_reports_its_condition(self) -> None:
        simulation = _model(conditional=True)

        assert simulation.model_definition.get_conditioning_categories("sub") == (
            "group1",
        )

    def test_unknown_stratification_is_rejected(self) -> None:
        simulation = _model()

        with pytest.raises(ValueError, match="Stratification 'nope' not found"):
            simulation.model_definition.get_conditioning_categories("nope")
