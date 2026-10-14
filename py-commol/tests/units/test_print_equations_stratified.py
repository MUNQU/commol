"""print_equations of stratified models: one line per expanded flow."""

import pytest

from commol import ModelBuilder
from commol.context.model import Model
from tests.support import build, sir_builder


def _printed(model: Model, capsys: pytest.CaptureFixture[str], **options) -> str:
    model.print_equations(**options)
    return capsys.readouterr().out


def _aging(name: str) -> ModelBuilder:
    """SIR split by age, with an aging transition from young to old."""
    return sir_builder(
        name=name,
        parameters={"beta": 0.3, "gamma": 0.1, "aging_rate": 0.01},
        stratifications={"age": {"young": 0.7, "old": 0.3}},
    ).add_transition(id="aging", source=["young"], target=["old"], rate="aging_rate")


def _conditional(second: str, categories: tuple[str, str]) -> ModelBuilder:
    """SIR split by age, and by ``second`` only for the old (``o``)."""
    return sir_builder(
        name="SIR Conditional",
        parameters={"beta": 0.3, "gamma": 0.1, "aging": 0.01},
        infection=None,
        recovery=None,
        fractions={"S": 0.99, "I": 0.005, "R": 0.005},
        stratifications={
            "age": {"y": 0.7, "o": 0.3},
            second: {categories[0]: 0.6, categories[1]: 0.4},
        },
        conditions={second: {"age": "o"}},
    )


def _per_compartment_sir(builder: ModelBuilder) -> ModelBuilder:
    return builder.add_transition(
        id="infection",
        source=["S"],
        target=["I"],
        rate="beta * S * I / total",
        per_compartment=True,
    ).add_transition(
        id="recovery", source=["I"], target=["R"], rate="gamma", per_compartment=True
    )


def _aging_to(
    builder: ModelBuilder, transition_id: str, rate: str, second: str, category: str
) -> ModelBuilder:
    """Move ``S_y`` to ``S_o`` and into ``category`` of ``second``."""
    return builder.add_transition(
        id=transition_id,
        source=["S"],
        target=["S"],
        stratified_rates=[
            {
                "conditions": [
                    {"stratification": "age", "category": "y", "to": "o"},
                    {"stratification": second, "to": category},
                ],
                "rate": rate,
            }
        ],
    )


def _vaccination() -> Model:
    builder = _per_compartment_sir(_conditional("vax", ("nv", "v")))
    return build(_aging_to(builder, "aging_transition", "aging * S_y", "vax", "nv"))


class TestText:
    def test_print_equations_with_stratification(self, capsys) -> None:
        output = _printed(build(_aging("Age-Stratified SIR")), capsys)

        expected = (
            "========================================\n"
            "MODEL INFORMATION\n"
            "========================================\n"
            "Model: Age-Stratified SIR\n"
            "Model Type: DifferenceEquations\n"
            "Number of Bins: 3\n"
            "Number of Stratifications: 1\n"
            "Number of Parameters: 3\n"
            "Number of Transitions: 3\n"
            "Bins: S, I, R\n"
            "Stratifications:\n"
            "  - age: [young, old]\n"
            "\n"
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "infection (S -> I):\n"
            "  S_young -> I_young: beta * S * I / total\n"
            "  S_old -> I_old: beta * S * I / total\n"
            "\n"
            "recovery (I -> R):\n"
            "  I_young -> R_young: gamma * I\n"
            "  I_old -> R_old: gamma * I\n"
            "\n"
            "Age Stratification Transitions:\n"
            "aging (young -> old):\n"
            "  S_young -> S_old: aging_rate * S_young\n"
            "  I_young -> I_old: aging_rate * I_young\n"
            "  R_young -> R_old: aging_rate * R_young\n"
            "\n"
            "Total System: 6 coupled equations (3 bins × 2 age)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "[S_young(t+Dt) - S_young(t)] / Dt = - (beta * S * I / total) - "
            "(aging_rate * S_young)\n"
            "[S_old(t+Dt) - S_old(t)] / Dt = - (beta * S * I / total) + (aging_rate * "
            "S_young)\n"
            "[I_young(t+Dt) - I_young(t)] / Dt = (beta * S * I / total) - (gamma * I) "
            "- (aging_rate * I_young)\n"
            "[I_old(t+Dt) - I_old(t)] / Dt = (beta * S * I / total) - (gamma * I) + "
            "(aging_rate * I_young)\n"
            "[R_young(t+Dt) - R_young(t)] / Dt = (gamma * I) - (aging_rate * "
            "R_young)\n"
            "[R_old(t+Dt) - R_old(t)] / Dt = (gamma * I) + (aging_rate * R_young)\n"
        )
        assert output == expected

    def test_print_equations_with_multiple_stratifications(self, capsys) -> None:
        """
        Rates differ by category, use the totals ``total_young`` and ``total_old``, and
        fall back to the transition rate for an unlisted category.
        """
        young, old = (
            {"stratification": "age", "category": "young"},
            {"stratification": "age", "category": "old"},
        )
        urban, rural = (
            {"stratification": "location", "category": "urban"},
            {"stratification": "location", "category": "rural"},
        )
        model = build(
            sir_builder(
                name="Age-Location Stratified SIR",
                parameters={
                    "beta_young_urban": 0.4,
                    "beta_young_rural": 0.25,
                    "beta_old_urban": 0.35,
                    "gamma_young": 0.15,
                    "aging_rate": 0.01,
                    "migration_rate_young": 0.02,
                    "migration_rate_old": 0.01,
                },
                infection=None,
                recovery=None,
                stratifications={
                    "age": {"young": 0.7, "old": 0.3},
                    "location": {"urban": 0.6, "rural": 0.4},
                },
            )
            .add_transition(
                id="infection",
                source=["S"],
                target=["I"],
                rate="0.3 * S * I / total",
                stratified_rates=[
                    {
                        "conditions": [young, urban],
                        "rate": "beta_young_urban * S * I / total_young",
                    },
                    {
                        "conditions": [young, rural],
                        "rate": "beta_young_rural * S * I / total_young",
                    },
                    {
                        "conditions": [old, urban],
                        "rate": "beta_old_urban * S * I / total_old",
                    },
                ],
            )
            .add_transition(
                id="recovery",
                source=["I"],
                target=["R"],
                rate="0.1 * I",
                stratified_rates=[{"conditions": [young], "rate": "gamma_young * I"}],
            )
            .add_transition(
                id="aging", source=["young"], target=["old"], rate="aging_rate"
            )
            .add_transition(
                id="migration",
                source=["urban"],
                target=["rural"],
                rate="0.015",
                stratified_rates=[
                    {"conditions": [young], "rate": "migration_rate_young"},
                    {"conditions": [old], "rate": "migration_rate_old"},
                ],
            )
        )

        output = _printed(model, capsys)

        expected = (
            "========================================\n"
            "MODEL INFORMATION\n"
            "========================================\n"
            "Model: Age-Location Stratified SIR\n"
            "Model Type: DifferenceEquations\n"
            "Number of Bins: 3\n"
            "Number of Stratifications: 2\n"
            "Number of Parameters: 7\n"
            "Number of Transitions: 4\n"
            "Bins: S, I, R\n"
            "Stratifications:\n"
            "  - age: [young, old]\n"
            "  - location: [urban, rural]\n"
            "\n"
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "infection (S -> I):\n"
            "  S_young_urban -> I_young_urban: beta_young_urban * S * I / total_young\n"
            "  S_young_rural -> I_young_rural: beta_young_rural * S * I / total_young\n"
            "  S_old_urban -> I_old_urban: beta_old_urban * S * I / total_old\n"
            "  S_old_rural -> I_old_rural: 0.3 * S * I / total\n"
            "\n"
            "recovery (I -> R):\n"
            "  I_young_urban -> R_young_urban: gamma_young * I\n"
            "  I_young_rural -> R_young_rural: gamma_young * I\n"
            "  I_old_urban -> R_old_urban: 0.1 * I\n"
            "  I_old_rural -> R_old_rural: 0.1 * I\n"
            "\n"
            "Age Stratification Transitions:\n"
            "aging (young -> old):\n"
            "  S_young_urban -> S_old_urban: aging_rate * S_young_urban\n"
            "  S_young_rural -> S_old_rural: aging_rate * S_young_rural\n"
            "  I_young_urban -> I_old_urban: aging_rate * I_young_urban\n"
            "  I_young_rural -> I_old_rural: aging_rate * I_young_rural\n"
            "  R_young_urban -> R_old_urban: aging_rate * R_young_urban\n"
            "  R_young_rural -> R_old_rural: aging_rate * R_young_rural\n"
            "\n"
            "Location Stratification Transitions:\n"
            "migration (urban -> rural):\n"
            "  S_young_urban -> S_young_rural: migration_rate_young * S_young_urban\n"
            "  S_old_urban -> S_old_rural: migration_rate_old * S_old_urban\n"
            "  I_young_urban -> I_young_rural: migration_rate_young * I_young_urban\n"
            "  I_old_urban -> I_old_rural: migration_rate_old * I_old_urban\n"
            "  R_young_urban -> R_young_rural: migration_rate_young * R_young_urban\n"
            "  R_old_urban -> R_old_rural: migration_rate_old * R_old_urban\n"
            "\n"
            "Total System: 12 coupled equations (3 bins × 2 age × 2 location)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "[S_young_urban(t+Dt) - S_young_urban(t)] / Dt = - (beta_young_urban * "
            "S * I / total_young) - (aging_rate * S_young_urban) - "
            "(migration_rate_young * S_young_urban)\n"
            "[S_young_rural(t+Dt) - S_young_rural(t)] / Dt = - (beta_young_rural * "
            "S * I / total_young) - (aging_rate * S_young_rural) + "
            "(migration_rate_young * S_young_urban)\n"
            "[S_old_urban(t+Dt) - S_old_urban(t)] / Dt = - (beta_old_urban * S * I "
            "/ total_old) + (aging_rate * S_young_urban) - (migration_rate_old * "
            "S_old_urban)\n"
            "[S_old_rural(t+Dt) - S_old_rural(t)] / Dt = - (0.3 * S * I / total) + "
            "(aging_rate * S_young_rural) + (migration_rate_old * S_old_urban)\n"
            "[I_young_urban(t+Dt) - I_young_urban(t)] / Dt = (beta_young_urban * S "
            "* I / total_young) - (gamma_young * I) - (aging_rate * I_young_urban) - "
            "(migration_rate_young * I_young_urban)\n"
            "[I_young_rural(t+Dt) - I_young_rural(t)] / Dt = (beta_young_rural * S "
            "* I / total_young) - (gamma_young * I) - (aging_rate * I_young_rural) + "
            "(migration_rate_young * I_young_urban)\n"
            "[I_old_urban(t+Dt) - I_old_urban(t)] / Dt = (beta_old_urban * S * I / "
            "total_old) - (0.1 * I) + (aging_rate * I_young_urban) - "
            "(migration_rate_old * I_old_urban)\n"
            "[I_old_rural(t+Dt) - I_old_rural(t)] / Dt = (0.3 * S * I / total) - (0.1 "
            "* I) + (aging_rate * I_young_rural) + (migration_rate_old * "
            "I_old_urban)\n"
            "[R_young_urban(t+Dt) - R_young_urban(t)] / Dt = (gamma_young * I) - "
            "(aging_rate * R_young_urban) - (migration_rate_young * R_young_urban)\n"
            "[R_young_rural(t+Dt) - R_young_rural(t)] / Dt = (gamma_young * I) - "
            "(aging_rate * R_young_rural) + (migration_rate_young * R_young_urban)\n"
            "[R_old_urban(t+Dt) - R_old_urban(t)] / Dt = (0.1 * I) + (aging_rate * "
            "R_young_urban) - (migration_rate_old * R_old_urban)\n"
            "[R_old_rural(t+Dt) - R_old_rural(t)] / Dt = (0.1 * I) + (aging_rate * "
            "R_young_rural) + (migration_rate_old * R_old_urban)\n"
        )
        assert output == expected

    def test_print_equations_conditional_stratification_text(self, capsys) -> None:
        """
        ``vax`` splits only ``age=o``, and the aging transition moves ``S_y``
        to ``S_o_nv`` through its ``to`` overrides.
        """
        output = _printed(_vaccination(), capsys, format="text")

        expected = (
            "========================================\n"
            "MODEL INFORMATION\n"
            "========================================\n"
            "Model: SIR Conditional\n"
            "Model Type: DifferenceEquations\n"
            "Number of Bins: 3\n"
            "Number of Stratifications: 2\n"
            "Number of Parameters: 3\n"
            "Number of Transitions: 3\n"
            "Bins: S, I, R\n"
            "Stratifications:\n"
            "  - age: [y, o]\n"
            "  - vax: [nv, v]\n"
            "\n"
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "infection (S -> I):\n"
            "  S_y -> I_y: beta * S_y * I_y / total\n"
            "  S_o_nv -> I_o_nv: beta * S_o_nv * I_o_nv / total\n"
            "  S_o_v -> I_o_v: beta * S_o_v * I_o_v / total\n"
            "\n"
            "recovery (I -> R):\n"
            "  I_y -> R_y: gamma * I_y\n"
            "  I_o_nv -> R_o_nv: gamma * I_o_nv\n"
            "  I_o_v -> R_o_v: gamma * I_o_v\n"
            "\n"
            "aging_transition (S -> S):\n"
            "  S_y -> S_o_nv: aging * S_y\n"
            "\n"
            "Total System: 9 coupled equations (3 bins × 2 age × 2 vax)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "[S_y(t+Dt) - S_y(t)] / Dt = - (beta * S_y * I_y / total) - (aging * S_y)\n"
            "[S_o_nv(t+Dt) - S_o_nv(t)] / Dt = - (beta * S_o_nv * I_o_nv / total) + "
            "(aging * S_y)\n"
            "[S_o_v(t+Dt) - S_o_v(t)] / Dt = - (beta * S_o_v * I_o_v / total)\n"
            "[I_y(t+Dt) - I_y(t)] / Dt = (beta * S_y * I_y / total) - (gamma * I_y)\n"
            "[I_o_nv(t+Dt) - I_o_nv(t)] / Dt = (beta * S_o_nv * I_o_nv / total) - "
            "(gamma * I_o_nv)\n"
            "[I_o_v(t+Dt) - I_o_v(t)] / Dt = (beta * S_o_v * I_o_v / total) - (gamma * "
            "I_o_v)\n"
            "[R_y(t+Dt) - R_y(t)] / Dt = (gamma * I_y)\n"
            "[R_o_nv(t+Dt) - R_o_nv(t)] / Dt = (gamma * I_o_nv)\n"
            "[R_o_v(t+Dt) - R_o_v(t)] / Dt = (gamma * I_o_v)\n"
        )
        assert output == expected

    def test_print_equations_cross_category_multiple_targets(self, capsys) -> None:
        """Aging from ``y`` splits between the two comorbidity categories."""
        builder = _per_compartment_sir(_conditional("comorbidity", ("nc", "c")))
        builder.add_parameter(id="p_c", value=0.2)
        builder = _aging_to(
            builder, "aging_nc", "aging * (1 - p_c) * S_y", "comorbidity", "nc"
        )
        builder = _aging_to(builder, "aging_c", "aging * p_c * S_y", "comorbidity", "c")

        output = _printed(build(builder), capsys, format="text")

        expected = (
            "========================================\n"
            "MODEL INFORMATION\n"
            "========================================\n"
            "Model: SIR Conditional\n"
            "Model Type: DifferenceEquations\n"
            "Number of Bins: 3\n"
            "Number of Stratifications: 2\n"
            "Number of Parameters: 4\n"
            "Number of Transitions: 4\n"
            "Bins: S, I, R\n"
            "Stratifications:\n"
            "  - age: [y, o]\n"
            "  - comorbidity: [nc, c]\n"
            "\n"
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "infection (S -> I):\n"
            "  S_y -> I_y: beta * S_y * I_y / total\n"
            "  S_o_nc -> I_o_nc: beta * S_o_nc * I_o_nc / total\n"
            "  S_o_c -> I_o_c: beta * S_o_c * I_o_c / total\n"
            "\n"
            "recovery (I -> R):\n"
            "  I_y -> R_y: gamma * I_y\n"
            "  I_o_nc -> R_o_nc: gamma * I_o_nc\n"
            "  I_o_c -> R_o_c: gamma * I_o_c\n"
            "\n"
            "aging_nc (S -> S):\n"
            "  S_y -> S_o_nc: aging * (1 - p_c) * S_y\n"
            "\n"
            "aging_c (S -> S):\n"
            "  S_y -> S_o_c: aging * p_c * S_y\n"
            "\n"
            "Total System: 9 coupled equations (3 bins × 2 age × 2 comorbidity)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "[S_y(t+Dt) - S_y(t)] / Dt = - (beta * S_y * I_y / total) - (aging * S_y)\n"
            "[S_o_nc(t+Dt) - S_o_nc(t)] / Dt = - (beta * S_o_nc * I_o_nc / total) + "
            "(aging * (1 - p_c) * S_y)\n"
            "[S_o_c(t+Dt) - S_o_c(t)] / Dt = - (beta * S_o_c * I_o_c / total) + (aging "
            "* p_c * S_y)\n"
            "[I_y(t+Dt) - I_y(t)] / Dt = (beta * S_y * I_y / total) - (gamma * I_y)\n"
            "[I_o_nc(t+Dt) - I_o_nc(t)] / Dt = (beta * S_o_nc * I_o_nc / total) - "
            "(gamma * I_o_nc)\n"
            "[I_o_c(t+Dt) - I_o_c(t)] / Dt = (beta * S_o_c * I_o_c / total) - (gamma * "
            "I_o_c)\n"
            "[R_y(t+Dt) - R_y(t)] / Dt = (gamma * I_y)\n"
            "[R_o_nc(t+Dt) - R_o_nc(t)] / Dt = (gamma * I_o_nc)\n"
            "[R_o_c(t+Dt) - R_o_c(t)] / Dt = (gamma * I_o_c)\n"
        )
        assert output == expected


class TestLatex:
    def test_print_equations_latex_with_stratification(self, capsys) -> None:
        model = _aging("Age-Stratified SIR Model").build(engine="DifferentialEquations")

        output = _printed(model, capsys, format="latex")

        expected = (
            "========================================\n"
            "MODEL INFORMATION\n"
            "========================================\n"
            "Model: Age-Stratified SIR Model\n"
            "Model Type: DifferentialEquations\n"
            "Number of Bins: 3\n"
            "Number of Stratifications: 1\n"
            "Number of Parameters: 3\n"
            "Number of Transitions: 3\n"
            "Bins: S, I, R\n"
            "Stratifications:\n"
            "  - age: [young, old]\n"
            "\n"
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "infection (S -> I):\n"
            "  $S_{young} \\to I_{young}: \\frac{\\beta \\cdot S \\cdot I}{total}$\n"
            "  $S_{old} \\to I_{old}: \\frac{\\beta \\cdot S \\cdot I}{total}$\n"
            "\n"
            "recovery (I -> R):\n"
            "  $I_{young} \\to R_{young}: \\gamma \\cdot I$\n"
            "  $I_{old} \\to R_{old}: \\gamma \\cdot I$\n"
            "\n"
            "Age Stratification Transitions:\n"
            "aging (young -> old):\n"
            "  $S_{young} \\to S_{old}: aging_{rate} \\cdot S_{young}$\n"
            "  $I_{young} \\to I_{old}: aging_{rate} \\cdot I_{young}$\n"
            "  $R_{young} \\to R_{old}: aging_{rate} \\cdot R_{young}$\n"
            "\n"
            "Total System: 6 coupled equations (3 bins × 2 age)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "\\[\\frac{dS_{young}}{dt} = - (\\frac{\\beta \\cdot S \\cdot I}{total}) - "
            "(aging_{rate} \\cdot S_{young})\\]\n"
            "\\[\\frac{dS_{old}}{dt} = - (\\frac{\\beta \\cdot S \\cdot I}{total}) + "
            "(aging_{rate} \\cdot S_{young})\\]\n"
            "\\[\\frac{dI_{young}}{dt} = (\\frac{\\beta \\cdot S \\cdot I}{total}) - "
            "(\\gamma "
            "\\cdot I) - (aging_{rate} \\cdot I_{young})\\]\n"
            "\\[\\frac{dI_{old}}{dt} = (\\frac{\\beta \\cdot S \\cdot I}{total}) - "
            "(\\gamma "
            "\\cdot I) + (aging_{rate} \\cdot I_{young})\\]\n"
            "\\[\\frac{dR_{young}}{dt} = (\\gamma \\cdot I) - (aging_{rate} \\cdot "
            "R_{young})\\]\n"
            "\\[\\frac{dR_{old}}{dt} = (\\gamma \\cdot I) + (aging_{rate} \\cdot "
            "R_{young})\\]\n"
        )
        assert output == expected

    def test_print_equations_conditional_stratification_latex(self, capsys) -> None:
        output = _printed(_vaccination(), capsys, format="latex")

        expected = (
            "========================================\n"
            "MODEL INFORMATION\n"
            "========================================\n"
            "Model: SIR Conditional\n"
            "Model Type: DifferenceEquations\n"
            "Number of Bins: 3\n"
            "Number of Stratifications: 2\n"
            "Number of Parameters: 3\n"
            "Number of Transitions: 3\n"
            "Bins: S, I, R\n"
            "Stratifications:\n"
            "  - age: [y, o]\n"
            "  - vax: [nv, v]\n"
            "\n"
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "infection (S -> I):\n"
            "  $S_{y} \\to I_{y}: \\frac{\\beta \\cdot S_{y} \\cdot I_{y}}{total}$\n"
            "  $S_{o,nv} \\to I_{o,nv}: \\frac{\\beta \\cdot S_{o,nv} \\cdot "
            "I_{o,nv}}{total}$\n"
            "  $S_{o,v} \\to I_{o,v}: \\frac{\\beta \\cdot S_{o,v} \\cdot "
            "I_{o,v}}{total}$\n"
            "\n"
            "recovery (I -> R):\n"
            "  $I_{y} \\to R_{y}: \\gamma \\cdot I_{y}$\n"
            "  $I_{o,nv} \\to R_{o,nv}: \\gamma \\cdot I_{o,nv}$\n"
            "  $I_{o,v} \\to R_{o,v}: \\gamma \\cdot I_{o,v}$\n"
            "\n"
            "aging_transition (S -> S):\n"
            "  $S_{y} \\to S_{o,nv}: aging \\cdot S_{y}$\n"
            "\n"
            "Total System: 9 coupled equations (3 bins × 2 age × 2 vax)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "\\[\\frac{S_{y}(t+\\Delta t) - S_{y}(t)}{\\Delta t} = - (\\frac{\\beta "
            "\\cdot S_{y} \\cdot I_{y}}{total}) - (aging \\cdot S_{y})\\]\n"
            "\\[\\frac{S_{o,nv}(t+\\Delta t) - S_{o,nv}(t)}{\\Delta t} = - "
            "(\\frac{\\beta \\cdot S_{o,nv} \\cdot I_{o,nv}}{total}) + (aging \\cdot "
            "S_{y})\\]\n"
            "\\[\\frac{S_{o,v}(t+\\Delta t) - S_{o,v}(t)}{\\Delta t} = - "
            "(\\frac{\\beta "
            "\\cdot S_{o,v} \\cdot I_{o,v}}{total})\\]\n"
            "\\[\\frac{I_{y}(t+\\Delta t) - I_{y}(t)}{\\Delta t} = "
            "(\\frac{\\beta \\cdot "
            "S_{y} \\cdot I_{y}}{total}) - (\\gamma \\cdot I_{y})\\]\n"
            "\\[\\frac{I_{o,nv}(t+\\Delta t) - I_{o,nv}(t)}{\\Delta t} = "
            "(\\frac{\\beta "
            "\\cdot S_{o,nv} \\cdot I_{o,nv}}{total}) - (\\gamma \\cdot I_{o,nv})\\]\n"
            "\\[\\frac{I_{o,v}(t+\\Delta t) - I_{o,v}(t)}{\\Delta t} = (\\frac{\\beta "
            "\\cdot S_{o,v} \\cdot I_{o,v}}{total}) - (\\gamma \\cdot I_{o,v})\\]\n"
            "\\[\\frac{R_{y}(t+\\Delta t) - R_{y}(t)}{\\Delta t} = (\\gamma \\cdot "
            "I_{y})\\]\n"
            "\\[\\frac{R_{o,nv}(t+\\Delta t) - R_{o,nv}(t)}{\\Delta t} = "
            "(\\gamma \\cdot "
            "I_{o,nv})\\]\n"
            "\\[\\frac{R_{o,v}(t+\\Delta t) - R_{o,v}(t)}{\\Delta t} = (\\gamma \\cdot "
            "I_{o,v})\\]\n"
        )
        assert output == expected
