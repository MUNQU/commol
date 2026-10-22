"""The panels of a figure are computed without matplotlib: the label, unit,
positions and values of each series."""

import json
import subprocess
import sys
from pathlib import Path

from commol import Simulation
from commol.api.plotting.panels import run_panels
from tests.support import abc_builder, build

PANELS = """
import json, sys
from commol import Simulation
from commol.api.plotting.panels import run_panels
from tests.support import abc_builder, build

run = Simulation(build(abc_builder(bin_unit="item"))).run(5)
print(json.dumps({
    "panels": [
        [p.series, p.name, p.unit, p.line.positions, p.line.values]
        for p in run_panels(run)
    ],
    "matplotlib": "matplotlib" in sys.modules,
}))
"""


def test_the_panels_of_a_run_need_no_matplotlib() -> None:
    computed = subprocess.run(
        [sys.executable, "-c", PANELS],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
    )

    out = json.loads(computed.stdout)
    run = Simulation(build(abc_builder(bin_unit="item"))).run(5)
    assert out["matplotlib"] is False
    assert out["panels"] == [
        [name, f"State {name}", "item", run.positions, run[name]]
        for name in ("A", "B", "C")
    ]
    assert [panel.line.values for panel in run_panels(run)] == [
        run[name] for name in ("A", "B", "C")
    ]
