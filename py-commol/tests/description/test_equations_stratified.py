"""The equations of the description of stratified models: one line per expanded flow."""

from commol import ModelBuilder
from commol.context.model import Model
from tests.support.description import equation_forms
from tests.support import build, build_ode, abc_builder


def _move(name: str) -> ModelBuilder:
    """``abc_builder`` split by group, with a transition from g1 to g2."""
    return abc_builder(
        name=name,
        parameters={"k1": 0.3, "k2": 0.1, "k3": 0.01},
        stratifications={"group": {"g1": 0.7, "g2": 0.3}},
    ).add_transition(id="move", source=["g1"], target=["g2"], rate="k3")


def _conditional(second: str, categories: tuple[str, str]) -> ModelBuilder:
    """``abc_builder`` split by group, and by ``second`` only for ``g2``."""
    return abc_builder(
        name="ABC Conditional",
        parameters={"k1": 0.3, "k2": 0.1, "k3": 0.01},
        ab=None,
        bc=None,
        fractions={"A": 0.99, "B": 0.005, "C": 0.005},
        stratifications={
            "group": {"g1": 0.7, "g2": 0.3},
            second: {categories[0]: 0.6, categories[1]: 0.4},
        },
        conditions={second: {"group": "g2"}},
    )


def _per_compartment_abc(builder: ModelBuilder) -> ModelBuilder:
    return builder.add_transition(
        id="ab",
        source=["A"],
        target=["B"],
        rate="k1 * A * B / total",
        per_compartment=True,
    ).add_transition(
        id="bc", source=["B"], target=["C"], rate="k2", per_compartment=True
    )


def _move_to(
    builder: ModelBuilder, transition_id: str, rate: str, second: str, category: str
) -> ModelBuilder:
    """Move ``A_g1`` to ``A_g2`` and into ``category`` of ``second``."""
    return builder.add_transition(
        id=transition_id,
        source=["A"],
        target=["A"],
        stratified_rates=[
            {
                "conditions": [
                    {"stratification": "group", "category": "g1", "to": "g2"},
                    {"stratification": second, "to": category},
                ],
                "rate": rate,
            }
        ],
    )


def _status_model() -> Model:
    builder = _per_compartment_abc(_conditional("status", ("s1", "s2")))
    return build(_move_to(builder, "move_transition", "k3 * A_g1", "status", "s1"))


class TestText:
    def test_with_stratification(self) -> None:
        output = equation_forms(build(_move("Group-Stratified ABC")))

        expected = (
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "ab (A -> B):\n"
            "  A_g1 -> B_g1: k1 * A * B / total\n"
            "  A_g2 -> B_g2: k1 * A * B / total\n"
            "\n"
            "bc (B -> C):\n"
            "  B_g1 -> C_g1: k2 * B\n"
            "  B_g2 -> C_g2: k2 * B\n"
            "\n"
            "Group Stratification Transitions:\n"
            "move (g1 -> g2):\n"
            "  A_g1 -> A_g2: k3 * A_g1\n"
            "  B_g1 -> B_g2: k3 * B_g1\n"
            "  C_g1 -> C_g2: k3 * C_g1\n"
            "\n"
            "Total System: 6 coupled equations (3 bins × 2 group)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "[A_g1(t+Dt) - A_g1(t)] / Dt = - (k1 * A * B / total) - "
            "(k3 * A_g1)\n"
            "[A_g2(t+Dt) - A_g2(t)] / Dt = - (k1 * A * B / total) + (k3 * "
            "A_g1)\n"
            "[B_g1(t+Dt) - B_g1(t)] / Dt = (k1 * A * B / total) - (k2 * B) "
            "- (k3 * B_g1)\n"
            "[B_g2(t+Dt) - B_g2(t)] / Dt = (k1 * A * B / total) - (k2 * B) + "
            "(k3 * B_g1)\n"
            "[C_g1(t+Dt) - C_g1(t)] / Dt = (k2 * B) - (k3 * "
            "C_g1)\n"
            "[C_g2(t+Dt) - C_g2(t)] / Dt = (k2 * B) + (k3 * C_g1)"
        )
        assert output == expected

    def test_with_multiple_stratifications(self) -> None:
        """
        Rates differ by category, use the totals ``total_g1`` and ``total_g2``, and
        fall back to the transition rate for an unlisted category.
        """
        g1, g2 = (
            {"stratification": "group", "category": "g1"},
            {"stratification": "group", "category": "g2"},
        )
        t1, t2 = (
            {"stratification": "type", "category": "t1"},
            {"stratification": "type", "category": "t2"},
        )
        model = build(
            abc_builder(
                name="Group-Type Stratified ABC",
                parameters={
                    "k1_g1_t1": 0.4,
                    "k1_g1_t2": 0.25,
                    "k1_g2_t1": 0.35,
                    "k2_g1": 0.15,
                    "k3": 0.01,
                    "k4_g1": 0.02,
                    "k4_g2": 0.01,
                },
                ab=None,
                bc=None,
                stratifications={
                    "group": {"g1": 0.7, "g2": 0.3},
                    "type": {"t1": 0.6, "t2": 0.4},
                },
            )
            .add_transition(
                id="ab",
                source=["A"],
                target=["B"],
                rate="0.3 * A * B / total",
                stratified_rates=[
                    {
                        "conditions": [g1, t1],
                        "rate": "k1_g1_t1 * A * B / total_g1",
                    },
                    {
                        "conditions": [g1, t2],
                        "rate": "k1_g1_t2 * A * B / total_g1",
                    },
                    {
                        "conditions": [g2, t1],
                        "rate": "k1_g2_t1 * A * B / total_g2",
                    },
                ],
            )
            .add_transition(
                id="bc",
                source=["B"],
                target=["C"],
                rate="0.1 * B",
                stratified_rates=[{"conditions": [g1], "rate": "k2_g1 * B"}],
            )
            .add_transition(id="move", source=["g1"], target=["g2"], rate="k3")
            .add_transition(
                id="cross",
                source=["t1"],
                target=["t2"],
                rate="0.015",
                stratified_rates=[
                    {"conditions": [g1], "rate": "k4_g1"},
                    {"conditions": [g2], "rate": "k4_g2"},
                ],
            )
        )

        output = equation_forms(model)

        expected = (
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "ab (A -> B):\n"
            "  A_g1_t1 -> B_g1_t1: k1_g1_t1 * A * B / total_g1\n"
            "  A_g1_t2 -> B_g1_t2: k1_g1_t2 * A * B / total_g1\n"
            "  A_g2_t1 -> B_g2_t1: k1_g2_t1 * A * B / total_g2\n"
            "  A_g2_t2 -> B_g2_t2: 0.3 * A * B / total\n"
            "\n"
            "bc (B -> C):\n"
            "  B_g1_t1 -> C_g1_t1: k2_g1 * B\n"
            "  B_g1_t2 -> C_g1_t2: k2_g1 * B\n"
            "  B_g2_t1 -> C_g2_t1: 0.1 * B\n"
            "  B_g2_t2 -> C_g2_t2: 0.1 * B\n"
            "\n"
            "Group Stratification Transitions:\n"
            "move (g1 -> g2):\n"
            "  A_g1_t1 -> A_g2_t1: k3 * A_g1_t1\n"
            "  A_g1_t2 -> A_g2_t2: k3 * A_g1_t2\n"
            "  B_g1_t1 -> B_g2_t1: k3 * B_g1_t1\n"
            "  B_g1_t2 -> B_g2_t2: k3 * B_g1_t2\n"
            "  C_g1_t1 -> C_g2_t1: k3 * C_g1_t1\n"
            "  C_g1_t2 -> C_g2_t2: k3 * C_g1_t2\n"
            "\n"
            "Type Stratification Transitions:\n"
            "cross (t1 -> t2):\n"
            "  A_g1_t1 -> A_g1_t2: k4_g1 * A_g1_t1\n"
            "  A_g2_t1 -> A_g2_t2: k4_g2 * A_g2_t1\n"
            "  B_g1_t1 -> B_g1_t2: k4_g1 * B_g1_t1\n"
            "  B_g2_t1 -> B_g2_t2: k4_g2 * B_g2_t1\n"
            "  C_g1_t1 -> C_g1_t2: k4_g1 * C_g1_t1\n"
            "  C_g2_t1 -> C_g2_t2: k4_g2 * C_g2_t1\n"
            "\n"
            "Total System: 12 coupled equations (3 bins × 2 group × 2 type)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "[A_g1_t1(t+Dt) - A_g1_t1(t)] / Dt = - (k1_g1_t1 * "
            "A * B / total_g1) - (k3 * A_g1_t1) - "
            "(k4_g1 * A_g1_t1)\n"
            "[A_g1_t2(t+Dt) - A_g1_t2(t)] / Dt = - (k1_g1_t2 * "
            "A * B / total_g1) - (k3 * A_g1_t2) + "
            "(k4_g1 * A_g1_t1)\n"
            "[A_g2_t1(t+Dt) - A_g2_t1(t)] / Dt = - (k1_g2_t1 * A * B "
            "/ total_g2) + (k3 * A_g1_t1) - (k4_g2 * "
            "A_g2_t1)\n"
            "[A_g2_t2(t+Dt) - A_g2_t2(t)] / Dt = - (0.3 * A * B / total) + "
            "(k3 * A_g1_t2) + (k4_g2 * A_g2_t1)\n"
            "[B_g1_t1(t+Dt) - B_g1_t1(t)] / Dt = (k1_g1_t1 * A "
            "* B / total_g1) - (k2_g1 * B) - (k3 * B_g1_t1) - "
            "(k4_g1 * B_g1_t1)\n"
            "[B_g1_t2(t+Dt) - B_g1_t2(t)] / Dt = (k1_g1_t2 * A "
            "* B / total_g1) - (k2_g1 * B) - (k3 * B_g1_t2) + "
            "(k4_g1 * B_g1_t1)\n"
            "[B_g2_t1(t+Dt) - B_g2_t1(t)] / Dt = (k1_g2_t1 * A * B / "
            "total_g2) - (0.1 * B) + (k3 * B_g1_t1) - "
            "(k4_g2 * B_g2_t1)\n"
            "[B_g2_t2(t+Dt) - B_g2_t2(t)] / Dt = (0.3 * A * B / total) - (0.1 "
            "* B) + (k3 * B_g1_t2) + (k4_g2 * "
            "B_g2_t1)\n"
            "[C_g1_t1(t+Dt) - C_g1_t1(t)] / Dt = (k2_g1 * B) - "
            "(k3 * C_g1_t1) - (k4_g1 * C_g1_t1)\n"
            "[C_g1_t2(t+Dt) - C_g1_t2(t)] / Dt = (k2_g1 * B) - "
            "(k3 * C_g1_t2) + (k4_g1 * C_g1_t1)\n"
            "[C_g2_t1(t+Dt) - C_g2_t1(t)] / Dt = (0.1 * B) + (k3 * "
            "C_g1_t1) - (k4_g2 * C_g2_t1)\n"
            "[C_g2_t2(t+Dt) - C_g2_t2(t)] / Dt = (0.1 * B) + (k3 * "
            "C_g1_t2) + (k4_g2 * C_g2_t1)"
        )
        assert output == expected

    def test_conditional_stratification_text(self) -> None:
        """
        ``status`` splits only ``group=g2``, and the move transition moves ``A_g1``
        to ``A_g2_s1`` through its ``to`` overrides.
        """
        output = equation_forms(_status_model())

        expected = (
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "ab (A -> B):\n"
            "  A_g1 -> B_g1: k1 * A_g1 * B_g1 / total\n"
            "  A_g2_s1 -> B_g2_s1: k1 * A_g2_s1 * B_g2_s1 / total\n"
            "  A_g2_s2 -> B_g2_s2: k1 * A_g2_s2 * B_g2_s2 / total\n"
            "\n"
            "bc (B -> C):\n"
            "  B_g1 -> C_g1: k2 * B_g1\n"
            "  B_g2_s1 -> C_g2_s1: k2 * B_g2_s1\n"
            "  B_g2_s2 -> C_g2_s2: k2 * B_g2_s2\n"
            "\n"
            "move_transition (A -> A):\n"
            "  A_g1 -> A_g2_s1: k3 * A_g1\n"
            "\n"
            "Total System: 9 coupled equations (3 bins × 2 group × 2 status)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "[A_g1(t+Dt) - A_g1(t)] / Dt = - (k1 * A_g1 * B_g1 / total) - (k3 * A_g1)\n"
            "[A_g2_s1(t+Dt) - A_g2_s1(t)] / Dt = - (k1 * A_g2_s1 * B_g2_s1 / total) + "
            "(k3 * A_g1)\n"
            "[A_g2_s2(t+Dt) - A_g2_s2(t)] / Dt = - (k1 * A_g2_s2 * B_g2_s2 / total)\n"
            "[B_g1(t+Dt) - B_g1(t)] / Dt = (k1 * A_g1 * B_g1 / total) - (k2 * B_g1)\n"
            "[B_g2_s1(t+Dt) - B_g2_s1(t)] / Dt = (k1 * A_g2_s1 * B_g2_s1 / total) - "
            "(k2 * B_g2_s1)\n"
            "[B_g2_s2(t+Dt) - B_g2_s2(t)] / Dt = (k1 * A_g2_s2 * B_g2_s2 / total) - "
            "(k2 * B_g2_s2)\n"
            "[C_g1(t+Dt) - C_g1(t)] / Dt = (k2 * B_g1)\n"
            "[C_g2_s1(t+Dt) - C_g2_s1(t)] / Dt = (k2 * B_g2_s1)\n"
            "[C_g2_s2(t+Dt) - C_g2_s2(t)] / Dt = (k2 * B_g2_s2)"
        )
        assert output == expected

    def test_cross_category_multiple_targets(self) -> None:
        """The move from ``g1`` splits between the two categories of ``level``."""
        builder = _per_compartment_abc(_conditional("level", ("l1", "l2")))
        builder.add_parameter(id="p_l2", value=0.2)
        builder = _move_to(builder, "move_l1", "k3 * (1 - p_l2) * A_g1", "level", "l1")
        builder = _move_to(builder, "move_l2", "k3 * p_l2 * A_g1", "level", "l2")

        output = equation_forms(build(builder))

        expected = (
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "ab (A -> B):\n"
            "  A_g1 -> B_g1: k1 * A_g1 * B_g1 / total\n"
            "  A_g2_l1 -> B_g2_l1: k1 * A_g2_l1 * B_g2_l1 / total\n"
            "  A_g2_l2 -> B_g2_l2: k1 * A_g2_l2 * B_g2_l2 / total\n"
            "\n"
            "bc (B -> C):\n"
            "  B_g1 -> C_g1: k2 * B_g1\n"
            "  B_g2_l1 -> C_g2_l1: k2 * B_g2_l1\n"
            "  B_g2_l2 -> C_g2_l2: k2 * B_g2_l2\n"
            "\n"
            "move_l1 (A -> A):\n"
            "  A_g1 -> A_g2_l1: k3 * (1 - p_l2) * A_g1\n"
            "\n"
            "move_l2 (A -> A):\n"
            "  A_g1 -> A_g2_l2: k3 * p_l2 * A_g1\n"
            "\n"
            "Total System: 9 coupled equations (3 bins × 2 group × 2 level)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "[A_g1(t+Dt) - A_g1(t)] / Dt = - (k1 * A_g1 * B_g1 / total) - (k3 * A_g1)\n"
            "[A_g2_l1(t+Dt) - A_g2_l1(t)] / Dt = - (k1 * A_g2_l1 * B_g2_l1 / total) + "
            "(k3 * (1 - p_l2) * A_g1)\n"
            "[A_g2_l2(t+Dt) - A_g2_l2(t)] / Dt = - (k1 * A_g2_l2 * B_g2_l2 / total) + "
            "(k3 * p_l2 * A_g1)\n"
            "[B_g1(t+Dt) - B_g1(t)] / Dt = (k1 * A_g1 * B_g1 / total) - (k2 * B_g1)\n"
            "[B_g2_l1(t+Dt) - B_g2_l1(t)] / Dt = (k1 * A_g2_l1 * B_g2_l1 / total) - "
            "(k2 * B_g2_l1)\n"
            "[B_g2_l2(t+Dt) - B_g2_l2(t)] / Dt = (k1 * A_g2_l2 * B_g2_l2 / total) - "
            "(k2 * B_g2_l2)\n"
            "[C_g1(t+Dt) - C_g1(t)] / Dt = (k2 * B_g1)\n"
            "[C_g2_l1(t+Dt) - C_g2_l1(t)] / Dt = (k2 * B_g2_l1)\n"
            "[C_g2_l2(t+Dt) - C_g2_l2(t)] / Dt = (k2 * B_g2_l2)"
        )
        assert output == expected


class TestLatex:
    def test_latex_with_stratification(self) -> None:
        model = build_ode(_move("Group-Stratified ABC Model"), settings=None)

        output = equation_forms(model, latex=True)

        expected = (
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "ab (A -> B):\n"
            "  $A_{g1} \\to B_{g1}: \\frac{k1 \\cdot A \\cdot B}{total}$\n"
            "  $A_{g2} \\to B_{g2}: \\frac{k1 \\cdot A \\cdot B}{total}$\n"
            "\n"
            "bc (B -> C):\n"
            "  $B_{g1} \\to C_{g1}: k2 \\cdot B$\n"
            "  $B_{g2} \\to C_{g2}: k2 \\cdot B$\n"
            "\n"
            "Group Stratification Transitions:\n"
            "move (g1 -> g2):\n"
            "  $A_{g1} \\to A_{g2}: k3 \\cdot A_{g1}$\n"
            "  $B_{g1} \\to B_{g2}: k3 \\cdot B_{g1}$\n"
            "  $C_{g1} \\to C_{g2}: k3 \\cdot C_{g1}$\n"
            "\n"
            "Total System: 6 coupled equations (3 bins × 2 group)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "\\[\\frac{dA_{g1}}{dt} = - (\\frac{k1 \\cdot A \\cdot B}{total}) - "
            "(k3 \\cdot A_{g1})\\]\n"
            "\\[\\frac{dA_{g2}}{dt} = - (\\frac{k1 \\cdot A \\cdot B}{total}) + "
            "(k3 \\cdot A_{g1})\\]\n"
            "\\[\\frac{dB_{g1}}{dt} = (\\frac{k1 \\cdot A \\cdot B}{total}) - "
            "(k2 "
            "\\cdot B) - (k3 \\cdot B_{g1})\\]\n"
            "\\[\\frac{dB_{g2}}{dt} = (\\frac{k1 \\cdot A \\cdot B}{total}) - "
            "(k2 "
            "\\cdot B) + (k3 \\cdot B_{g1})\\]\n"
            "\\[\\frac{dC_{g1}}{dt} = (k2 \\cdot B) - (k3 \\cdot "
            "C_{g1})\\]\n"
            "\\[\\frac{dC_{g2}}{dt} = (k2 \\cdot B) + (k3 \\cdot "
            "C_{g1})\\]"
        )
        assert output == expected

    def test_conditional_stratification_latex(self) -> None:
        output = equation_forms(_status_model(), latex=True)

        expected = (
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "ab (A -> B):\n"
            "  $A_{g1} \\to B_{g1}: \\frac{k1 \\cdot A_{g1} \\cdot B_{g1}}{total}$\n"
            "  $A_{g2,s1} \\to B_{g2,s1}: \\frac{k1 \\cdot A_{g2,s1} \\cdot "
            "B_{g2,s1}}{total}$\n"
            "  $A_{g2,s2} \\to B_{g2,s2}: \\frac{k1 \\cdot A_{g2,s2} \\cdot "
            "B_{g2,s2}}{total}$\n"
            "\n"
            "bc (B -> C):\n"
            "  $B_{g1} \\to C_{g1}: k2 \\cdot B_{g1}$\n"
            "  $B_{g2,s1} \\to C_{g2,s1}: k2 \\cdot B_{g2,s1}$\n"
            "  $B_{g2,s2} \\to C_{g2,s2}: k2 \\cdot B_{g2,s2}$\n"
            "\n"
            "move_transition (A -> A):\n"
            "  $A_{g1} \\to A_{g2,s1}: k3 \\cdot A_{g1}$\n"
            "\n"
            "Total System: 9 coupled equations (3 bins × 2 group × 2 status)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "\\[\\frac{A_{g1}(t+\\Delta t) - A_{g1}(t)}{\\Delta t} = - (\\frac{k1 "
            "\\cdot A_{g1} \\cdot B_{g1}}{total}) - (k3 \\cdot A_{g1})\\]\n"
            "\\[\\frac{A_{g2,s1}(t+\\Delta t) - A_{g2,s1}(t)}{\\Delta t} = - "
            "(\\frac{k1 \\cdot A_{g2,s1} \\cdot B_{g2,s1}}{total}) + (k3 \\cdot "
            "A_{g1})\\]\n"
            "\\[\\frac{A_{g2,s2}(t+\\Delta t) - A_{g2,s2}(t)}{\\Delta t} = - "
            "(\\frac{k1 "
            "\\cdot A_{g2,s2} \\cdot B_{g2,s2}}{total})\\]\n"
            "\\[\\frac{B_{g1}(t+\\Delta t) - B_{g1}(t)}{\\Delta t} = "
            "(\\frac{k1 \\cdot "
            "A_{g1} \\cdot B_{g1}}{total}) - (k2 \\cdot B_{g1})\\]\n"
            "\\[\\frac{B_{g2,s1}(t+\\Delta t) - B_{g2,s1}(t)}{\\Delta t} = "
            "(\\frac{k1 "
            "\\cdot A_{g2,s1} \\cdot B_{g2,s1}}{total}) - (k2 \\cdot B_{g2,s1})\\]\n"
            "\\[\\frac{B_{g2,s2}(t+\\Delta t) - B_{g2,s2}(t)}{\\Delta t} = (\\frac{k1 "
            "\\cdot A_{g2,s2} \\cdot B_{g2,s2}}{total}) - (k2 \\cdot B_{g2,s2})\\]\n"
            "\\[\\frac{C_{g1}(t+\\Delta t) - C_{g1}(t)}{\\Delta t} = (k2 \\cdot "
            "B_{g1})\\]\n"
            "\\[\\frac{C_{g2,s1}(t+\\Delta t) - C_{g2,s1}(t)}{\\Delta t} = "
            "(k2 \\cdot "
            "B_{g2,s1})\\]\n"
            "\\[\\frac{C_{g2,s2}(t+\\Delta t) - C_{g2,s2}(t)}{\\Delta t} = (k2 \\cdot "
            "B_{g2,s2})\\]"
        )
        assert output == expected
