"""A description is data: the sections of a model and of its kind."""

import pytest

from commol import Complete, Groups
from commol.context.description import (
    Equations,
    Matrix,
    ModelDescription,
    StateGraph,
    Table,
    Text,
)
from commol.context.model import Model
from tests.support import abc_builder, build, build_ode
from tests.support.kinds import DecayStructure, decay_model

DEFAULT_KEYS = [
    "overview",
    "entities",
    "dynamics",
    "schedule",
    "initialization",
    "parameters",
    "outputs",
]
ODD_KEYS = [
    "purpose",
    "entities",
    "process_overview",
    "design_concepts",
    "initialization",
    "input_data",
    "submodels",
]


def _stratified() -> Model:
    """A → B → C split by ``group``, with a move from g1 to g2."""
    builder = abc_builder(stratifications={"group": {"g1": 0.3, "g2": 0.7}})
    builder.add_transition(id="move", source=["g1"], target=["g2"], rate="0.01")
    return build(builder)


def _cells(description: ModelDescription, key: str) -> list[str]:
    """Every cell of the tables of a section, as text."""
    return [
        str(cell)
        for block in description.section(key).blocks
        if isinstance(block, Table)
        for row in block.rows
        for cell in row
    ]


class TestSections:
    def test_the_default_layout_has_the_seven_sections_in_order(self) -> None:
        description = _stratified().describe()

        assert [section.key for section in description.sections] == DEFAULT_KEYS
        assert description.layout == "default"

    def test_an_aggregate_description_holds_every_part_of_the_model(self) -> None:
        model = _stratified()
        description = model.describe()

        entities = _cells(description, "entities")
        for bin_item in model.process.bins:
            assert bin_item.id in entities
        for compartment in model.structure.initial_state(model.process):
            assert compartment.name in entities
        assert "g1, g2" in entities

        groups = [
            group.heading
            for block in description.section("dynamics").blocks
            if isinstance(block, Equations)
            for group in block.groups
        ]
        for transition in model.process.transitions:
            assert any(
                heading is not None and heading.startswith(f"{transition.id} (")
                for heading in groups
            )
        flows = {
            (equation.left, equation.right)
            for block in description.section("dynamics").blocks
            if isinstance(block, Equations)
            for group in block.groups
            for equation in group.equations
            if equation.relation == "flow"
        }
        assert ("A_g1 -> A_g2", "0.01 * A_g1") in flows
        assert ("B_g2 -> C_g2", "k2 * B") in flows

        parameters = _cells(description, "parameters")
        assert {"k1", "k2"} <= set(parameters)

        amounts = {
            row[0]: row[1]
            for block in description.section("initialization").blocks
            if isinstance(block, Table) and block.heading.startswith("Initial state")
            for row in block.rows
        }
        assert amounts == {
            compartment.name: compartment.amount
            for compartment in model.structure.initial_state(model.process)
        }

    def test_the_overview_names_the_kind_the_engine_and_its_settings(self) -> None:
        description = build_ode(abc_builder()).describe()

        overview = dict(
            (row[0], row[1])
            for block in description.section("overview").blocks
            if isinstance(block, Table)
            for row in block.rows
        )
        assert overview["Kind"] == "aggregate"
        assert overview["Engine"] == "DifferentialEquations"
        assert "1e-10" in str(overview["Engine settings"])
        assert overview["Dynamics"] == "differential equation"

    def test_the_schedule_comes_from_the_engine(self) -> None:
        difference = build(abc_builder()).describe().section("schedule")
        differential = build_ode(abc_builder()).describe().section("schedule")

        assert isinstance(difference.blocks[0], Text)
        assert "Each step applies every flow once" in difference.blocks[0].text
        assert isinstance(differential.blocks[0], Text)
        assert "integrates" in differential.blocks[0].text

    def test_a_parameter_left_for_calibration_says_so(self) -> None:
        description = build(abc_builder(parameters={"k1": None, "k2": 0.1})).describe()

        rows = [
            row
            for block in description.section("parameters").blocks
            if isinstance(block, Table)
            for row in block.rows
        ]
        assert rows[0][:4] == ["k1", None, None, True]
        assert rows[1][:4] == ["k2", 0.1, None, False]

    def test_the_neighbors_of_groups_give_their_weights(self) -> None:
        builder = abc_builder(
            ab="k1 * neighbors_B / neighbors",
            bc="k2",
            stratifications={"group": {"g1": 0.5, "g2": 0.5}},
        )
        weights = {"g1": {"g1": 1.0, "g2": 0.25}, "g2": {"g1": 0.5, "g2": 1.0}}
        builder.structure = builder.structure.model_copy(
            update={"neighbors": Groups("group", weights=weights)}
        )

        blocks = build(builder).describe().section("dynamics").blocks

        matrix = next(block for block in blocks if isinstance(block, Matrix))
        assert matrix.row_labels == ["g1", "g2"]
        assert matrix.values == [[1.0, 0.25], [0.5, 1.0]]


class TestKinds:
    def test_the_test_kind_gives_its_own_dynamics(self) -> None:
        description = decay_model().describe()

        dynamics = description.section("dynamics").blocks
        assert isinstance(dynamics[0], Equations)
        equation = dynamics[0].groups[0].equations[0]
        assert (equation.left, equation.right) == ("dx/dt", "-k * x")
        assert description.section("entities").blocks == []

    def test_a_kind_that_adds_nothing_still_has_the_shared_sections(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(DecayStructure, "describe", lambda *_: [])

        description = decay_model().describe()

        shown = [section.key for section in description.sections if section.blocks]
        assert shown == ["overview", "schedule", "parameters", "outputs"]
        assert _cells(description, "outputs") == ["x", "x"]


class TestStateGraphs:
    def test_an_aggregate_gives_a_graph_of_bins_and_one_of_compartments(
        self,
    ) -> None:
        graphs = _stratified().describe().graphs()

        states, compartments = graphs
        assert [node.id for node in states.nodes] == [
            "A",
            "B",
            "C",
            "group:g1",
            "group:g2",
        ]
        assert not states.expanded
        assert {(edge.source, edge.target) for edge in states.edges} == {
            ("A", "B"),
            ("B", "C"),
            ("group:g1", "group:g2"),
        }
        assert compartments.expanded
        assert len(compartments.nodes) == 6
        assert len(compartments.edges) == 2 + 2 + 3

    def test_inflows_and_outflows_have_no_node_on_one_side(self) -> None:
        builder = abc_builder()
        builder.add_parameter(id="b", value=1.0)
        builder.add_transition(id="in", source=[], target=["A"], rate="b")
        builder.add_transition(id="out", source=["C"], target=[], rate="0.01")

        states = build(builder).describe().graphs()[0]

        assert StateGraph.model_validate(states.model_dump()) == states
        ends = {
            edge.label.split(":")[0]: (edge.source, edge.target)
            for edge in states.edges
        }
        assert ends["in"] == (None, "A")
        assert ends["out"] == ("C", None)


class TestOddLayout:
    def test_the_odd_layout_follows_the_protocol(self) -> None:
        description = _stratified().describe(layout="odd")

        assert [section.key for section in description.sections] == ODD_KEYS
        assert [section.title for section in description.sections] == [
            "Purpose and patterns",
            "Entities, state variables and scales",
            "Process overview and scheduling",
            "Design concepts",
            "Initialization",
            "Input data",
            "Submodels",
        ]

    def test_only_what_commol_cannot_fill_is_marked(self) -> None:
        builder = abc_builder(ab="k1 * neighbors_B / neighbors", bc="k2")
        builder.structure = builder.structure.model_copy(
            update={"neighbors": Complete()}
        )
        model = build(builder)
        model.description = "Two moves between three states."

        description = model.describe(layout="odd")

        marker = "To be written by the modeller."
        purpose = description.section("purpose").blocks[0]
        assert isinstance(purpose, Text) and purpose.text == model.description
        concepts = {
            row[0]: row[1]
            for block in description.section("design_concepts").blocks
            if isinstance(block, Table)
            for row in block.rows
        }
        filled = {"Interaction", "Stochasticity", "Observation"}
        assert {
            element for element, text in concepts.items() if text != marker
        } == filled
        for key in (
            "entities",
            "process_overview",
            "initialization",
            "input_data",
            "submodels",
        ):
            text = description.section(key).model_dump_json()
            assert marker not in text, key

    def test_a_model_without_a_description_has_its_purpose_marked(self) -> None:
        purpose = build(abc_builder()).describe(layout="odd").section("purpose")

        assert isinstance(purpose.blocks[0], Text)
        assert purpose.blocks[0].text == "To be written by the modeller."

    def test_an_unknown_layout_lists_the_layouts(self) -> None:
        with pytest.raises(ValueError, match="odd"):
            build(abc_builder()).describe(layout="report")  # ty: ignore[invalid-argument-type]

    def test_a_missing_section_lists_the_sections(self) -> None:
        with pytest.raises(KeyError, match="dynamics"):
            build(abc_builder()).describe(layout="odd").section("dynamics")
