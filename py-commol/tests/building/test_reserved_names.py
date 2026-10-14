"""Parameter ids cannot be names the engine provides."""

import pytest

from commol import ModelBuilder, Simulation
from tests.support import a_to_b_builder, build, sir_builder


def _builder(parameter_id: str, stratifications: int = 1) -> ModelBuilder:
    return a_to_b_builder(
        name="Reserved",
        parameters={parameter_id: 0.1, "k": 0.1},
        stratifications={
            "group": {"g1": 0.5, "g2": 0.5},
            **({"type": {"t1": 0.5, "t2": 0.5}} if stratifications == 2 else {}),
        },
    )


@pytest.mark.parametrize("name", ["total", "step", "t", "pi", "e"])
def test_an_engine_variable_is_not_a_parameter_id(name: str) -> None:
    with pytest.raises(ValueError, match=f"engine variables: \\['{name}'\\]"):
        build(_builder(name))


def test_a_bin_id_is_not_a_parameter_id() -> None:
    with pytest.raises(ValueError, match="reserved compartment names: \\['A'\\]"):
        build(_builder("A"))


def test_every_conflicting_parameter_id_is_reported() -> None:
    builder = sir_builder(parameters={"S": 0.5, "I": 0.3, "gamma": 0.1}, infection=None)

    with pytest.raises(ValueError, match="reserved compartment names: \\['I', 'S'\\]"):
        build(builder)


def test_a_compartment_is_not_a_parameter_id() -> None:
    with pytest.raises(ValueError, match="'A_g1' \\(a compartment\\)"):
        build(_builder("A_g1"))


def test_a_subgroup_total_is_not_a_parameter_id() -> None:
    with pytest.raises(ValueError, match="'total_g1' \\(a subgroup total\\)"):
        build(_builder("total_g1"))


def test_a_partial_bin_total_is_not_a_parameter_id() -> None:
    with pytest.raises(ValueError, match="'A_t1' \\(a subgroup total\\)"):
        build(_builder("A_t1", stratifications=2))


def test_a_total_prefix_that_names_no_subgroup_is_a_valid_id() -> None:
    model = build(_builder("total_max"))

    assert [parameter.id for parameter in model.parameters] == ["total_max", "k"]


def test_a_bin_named_n_builds_and_runs() -> None:
    builder = ModelBuilder(name="NPZ", version="1.0")
    for bin_id in ("N", "P"):
        builder.add_bin(id=bin_id, name=bin_id)
    builder.add_parameter(id="uptake", value=0.5)
    builder.add_transition(
        id="growth", source=["N"], target=["P"], rate="uptake * P / total"
    )
    builder.set_initial_conditions(
        total=2.5,
        bin_fractions=[{"bin": "N", "fraction": 0.8}, {"bin": "P", "fraction": 0.2}],
    )

    result = Simulation(build(builder)).run(1)

    # The rate reads P, so it is an absolute flow: 0.5 * 0.5 / 2.5 = 0.1.
    assert result["N"] == pytest.approx([2.0, 1.9])
    assert result["P"] == pytest.approx([0.5, 0.6])
