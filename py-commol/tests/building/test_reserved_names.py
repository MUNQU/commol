"""Parameter, bin and accumulator ids cannot be names the engine provides."""

import re

import pytest

from commol import Aggregate, ModelBuilder, Simulation
from tests.support import FactoryBuilder, a_to_b_builder, build, abc_builder


def _builder(parameter_id: str, stratifications: int = 1) -> ModelBuilder:
    return a_to_b_builder(
        name="Reserved",
        parameters={parameter_id: 0.1, "k": 0.1},
        stratifications={
            "group": {"g1": 0.5, "g2": 0.5},
            **({"type": {"t1": 0.5, "t2": 0.5}} if stratifications == 2 else {}),
        },
    )


@pytest.mark.parametrize(
    ("parameter_id", "conflict"),
    [
        *((name, f"parameter '{name}'") for name in ("total", "step", "t", "pi", "e")),
        ("A", "'A' (a subgroup total)"),
    ],
)
def test_a_reserved_parameter_id_gets_the_message_of_the_engine(
    parameter_id: str, conflict: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(conflict)) as error:
        build(_builder(parameter_id))

    assert "Rename" in str(error.value)


@pytest.mark.parametrize("name", ["total", "step", "t", "pi", "e"])
def test_an_engine_variable_is_not_a_bin_id(name: str) -> None:
    builder = FactoryBuilder(name="Reserved", version="1.0")
    builder.add_bin(id=name, name=name).add_bin(id="B", name="B")
    builder.add_parameter(id="k", value=0.1)
    builder.add_transition(id="flow", source=[name], target=["B"], rate="k")
    builder.set_structure(Aggregate(bin_amounts={name: 1000, "B": 0}))

    with pytest.raises(ValueError, match=f"bin '{name}'"):
        build(builder)


def test_an_engine_variable_is_not_an_accumulator_id() -> None:
    builder = a_to_b_builder(accumulator="total")

    with pytest.raises(ValueError, match="accumulator 'total'"):
        build(builder)


def test_a_bin_name_that_reads_as_a_subgroup_total_is_rejected() -> None:
    builder = FactoryBuilder(name="Reserved", version="1.0")
    builder.add_bin(id="total_y", name="total_y").add_bin(id="B", name="B")
    builder.add_stratification(id="group", categories=["y", "o"])
    builder.add_stratification(id="type", categories=["y", "o"])
    builder.add_parameter(id="k", value=0.1)
    builder.add_transition(id="flow", source=["total_y"], target=["B"], rate="k")
    halves = {"y": 0.5, "o": 0.5}
    builder.set_structure(
        Aggregate(
            bin_amounts={"total_y": 1000, "B": 0},
            stratification_fractions={"group": halves, "type": halves},
        )
    )

    with pytest.raises(ValueError, match="'total_y_y'"):
        build(builder)


def test_every_conflicting_parameter_id_is_reported() -> None:
    builder = abc_builder(parameters={"A": 0.5, "B": 0.3, "k2": 0.1}, ab=None)

    with pytest.raises(
        ValueError, match=re.escape("'A' (a compartment), 'B' (a compartment)")
    ):
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
    builder = FactoryBuilder(name="NPZ", version="1.0")
    for bin_id in ("N", "P"):
        builder.add_bin(id=bin_id, name=bin_id)
    builder.add_parameter(id="uptake", value=0.5)
    builder.add_transition(
        id="growth", source=["N"], target=["P"], rate="uptake * P / total"
    )
    builder.set_structure(Aggregate(bin_amounts={"N": 2, "P": 0.5}))

    result = Simulation(build(builder)).run(1)

    # The rate reads P, so it is an absolute flow: 0.5 * 0.5 / 2.5 = 0.1.
    assert result["N"] == pytest.approx([2.0, 1.9])
    assert result["P"] == pytest.approx([0.5, 0.6])
