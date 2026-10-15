"""Parameter, bin and accumulator ids cannot be names the engine provides."""

import pytest

from commol import ModelBuilder, Simulation
from tests.support import a_to_b_builder, build, abc_builder


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


@pytest.mark.parametrize("name", ["total", "step", "t", "pi", "e"])
def test_an_engine_variable_is_not_a_bin_id(name: str) -> None:
    builder = ModelBuilder(name="Reserved", version="1.0")
    builder.add_bin(id=name, name=name).add_bin(id="B", name="B")
    builder.add_parameter(id="k", value=0.1)
    builder.add_transition(id="flow", source=[name], target=["B"], rate="k")
    builder.set_initial_conditions(
        total=1000,
        bin_fractions=[{"bin": name, "fraction": 1.0}, {"bin": "B", "fraction": 0.0}],
    )

    with pytest.raises(ValueError, match=f"bin '{name}'"):
        build(builder)


def test_an_engine_variable_is_not_an_accumulator_id() -> None:
    builder = a_to_b_builder(accumulator="total")

    with pytest.raises(ValueError, match="accumulator 'total'"):
        build(builder)


def test_a_bin_name_that_reads_as_a_subgroup_total_is_rejected() -> None:
    builder = ModelBuilder(name="Reserved", version="1.0")
    builder.add_bin(id="total_y", name="total_y").add_bin(id="B", name="B")
    builder.add_stratification(id="group", categories=["y", "o"])
    builder.add_stratification(id="type", categories=["y", "o"])
    builder.add_parameter(id="k", value=0.1)
    builder.add_transition(id="flow", source=["total_y"], target=["B"], rate="k")
    halves = [{"category": "y", "fraction": 0.5}, {"category": "o", "fraction": 0.5}]
    builder.set_initial_conditions(
        total=1000,
        bin_fractions=[
            {"bin": "total_y", "fraction": 1.0},
            {"bin": "B", "fraction": 0.0},
        ],
        stratification_fractions=[
            {"stratification": "group", "fractions": halves},
            {"stratification": "type", "fractions": halves},
        ],
    )

    with pytest.raises(ValueError, match="'total_y_y'"):
        build(builder)


def test_a_bin_id_is_not_a_parameter_id() -> None:
    with pytest.raises(ValueError, match="reserved compartment names: \\['A'\\]"):
        build(_builder("A"))


def test_every_conflicting_parameter_id_is_reported() -> None:
    builder = abc_builder(parameters={"A": 0.5, "B": 0.3, "k2": 0.1}, ab=None)

    with pytest.raises(ValueError, match="reserved compartment names: \\['A', 'B'\\]"):
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
