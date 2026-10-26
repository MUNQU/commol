"""A condition and a stratified rate are classes that the builder takes."""

import subprocess
import sys
from pathlib import Path

from commol import Simulation, StratificationCondition, StratifiedRate
from tests.support import ENGINE_REL, abc_builder, assert_close, build, reference_run

BUILDER_CODE = """\
from commol import ModelBuilder, StratificationCondition, StratifiedRate

builder = (
    ModelBuilder(name="typed")
    .add_bin(id="A", name="A")
    .add_bin(id="B", name="B")
    .add_stratification(id="group", categories=["g1", "g2"])
    .add_stratification(
        id="status",
        categories=["s1", "s2"],
        conditions=[StratificationCondition(stratification="group", category="g2")],
    )
    .add_transition(
        id="ab",
        source=["A"],
        target=["B"],
        rate="k",
        stratified_rates=[
            StratifiedRate(
                conditions=[
                    StratificationCondition(stratification="group", category="g1")
                ],
                rate=0.5,
            )
        ],
    )
)
"""


def test_a_condition_is_one_class(tmp_path: Path) -> None:
    builder = abc_builder(
        parameters={"k1": 0.0, "k2": 0.1, "k2_s1": 0.4},
        bc=None,
        amounts={"A": 0, "B": 1000, "C": 0},
        stratifications={
            "group": {"g1": 0.5, "g2": 0.5},
            "status": {"s1": 0.5, "s2": 0.5},
        },
        conditions={"status": {"group": "g2"}},
    )
    builder.add_transition(
        id="bc",
        source=["B"],
        target=["C"],
        rate="k2 * B_g1",
        stratified_rates=[
            StratifiedRate(
                conditions=[
                    StratificationCondition(stratification="group", category="g2"),
                    StratificationCondition(stratification="status", category="s1"),
                ],
                rate="k2_s1 * B_g2_s1",
            )
        ],
    )

    result = Simulation(build(builder)).run(3)

    # The status splits g2 only (`stratify` gives the condition as a
    # StratificationCondition); g2_s1 takes its stratified rate, g1 and g2_s2
    # the fallback. A rate that reads a compartment is absolute.
    expected = reference_run(
        {
            "A_g1": 0.0,
            "A_g2_s1": 0.0,
            "A_g2_s2": 0.0,
            "B_g1": 500.0,
            "B_g2_s1": 250.0,
            "B_g2_s2": 250.0,
            "C_g1": 0.0,
            "C_g2_s1": 0.0,
            "C_g2_s2": 0.0,
        },
        [
            ("B_g1", "C_g1", lambda s, _: 0.1 * s["B_g1"]),
            ("B_g2_s1", "C_g2_s1", lambda s, _: 0.4 * s["B_g2_s1"]),
            ("B_g2_s2", "C_g2_s2", lambda s, _: 0.1 * s["B_g1"]),
        ],
        3,
    )
    assert_close(result, expected, rel=ENGINE_REL)

    source = tmp_path / "typed.py"
    source.write_text(BUILDER_CODE, encoding="utf-8")
    checked = subprocess.run(
        [Path(sys.executable).parent / "ty", "check", "--python", sys.prefix, source],
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode == 0, checked.stdout
