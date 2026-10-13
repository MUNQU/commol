"""Parameter ids cannot be names the engine provides."""

import pytest

from commol.api.model_builder import ModelBuilder
from commol.constants import ModelTypes


def _builder(parameter_id: str, stratifications: int = 1) -> ModelBuilder:
    builder = (
        ModelBuilder(name="Reserved")
        .add_bin(id="A", name="A")
        .add_bin(id="B", name="B")
        .add_parameter(id=parameter_id, value=0.1)
        .add_parameter(id="k", value=0.1)
        .add_stratification(id="group", categories=["g1", "g2"])
    )
    fractions = [
        {
            "stratification": "group",
            "fractions": [
                {"category": "g1", "fraction": 0.5},
                {"category": "g2", "fraction": 0.5},
            ],
        }
    ]
    if stratifications == 2:
        builder = builder.add_stratification(id="type", categories=["t1", "t2"])
        fractions.append(
            {
                "stratification": "type",
                "fractions": [
                    {"category": "t1", "fraction": 0.5},
                    {"category": "t2", "fraction": 0.5},
                ],
            }
        )
    return builder.add_transition(
        id="t_ab", source=["A"], target=["B"], rate="k"
    ).set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "A", "fraction": 1.0},
            {"bin": "B", "fraction": 0.0},
        ],
        stratification_fractions=fractions,
    )


@pytest.mark.parametrize("name", ["N", "step", "t", "pi", "e"])
def test_an_engine_variable_is_not_a_parameter_id(name: str) -> None:
    with pytest.raises(ValueError, match=f"engine variables: \\['{name}'\\]"):
        _builder(name).build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)


def test_a_bin_id_is_not_a_parameter_id() -> None:
    with pytest.raises(ValueError, match="reserved compartment names: \\['A'\\]"):
        _builder("A").build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)


def test_a_compartment_is_not_a_parameter_id() -> None:
    with pytest.raises(ValueError, match="'A_g1' \\(a compartment\\)"):
        _builder("A_g1").build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)


def test_a_subpopulation_is_not_a_parameter_id() -> None:
    with pytest.raises(ValueError, match="'N_g1' \\(a subpopulation total\\)"):
        _builder("N_g1").build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)


def test_a_partial_bin_total_is_not_a_parameter_id() -> None:
    with pytest.raises(ValueError, match="'A_t1' \\(a subpopulation total\\)"):
        _builder("A_t1", stratifications=2).build(
            typology=ModelTypes.DIFFERENCE_EQUATIONS.value
        )


def test_an_n_prefix_that_names_no_subpopulation_is_a_valid_id() -> None:
    _builder("N_max").build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
