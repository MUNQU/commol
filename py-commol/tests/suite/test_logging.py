"""Check that commol configures no logging and that the application sees its messages.

The test suite sends the messages of commol to the standard error stream
(``tests/support/logs.py``), so each test runs its check in a fresh
interpreter. The script of a fresh interpreter cannot import ``tests``, whose
``__init__`` adds that handler, so it builds its model by hand.
"""

import logging
import subprocess
import sys
import textwrap

import pytest

from commol import Simulation
from commol.sensitivity import SensitivityAnalysis
from tests.support import abc_builder, abc_rates, build, flow_calibrator


def _run_python(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        check=True,
    )


def test_importing_commol_adds_only_a_null_handler() -> None:
    result = _run_python(
        """
        import logging

        import commol

        logger = logging.getLogger("commol")
        print([type(handler).__name__ for handler in logger.handlers])
        print(logger.level == logging.NOTSET)
        print(logger.propagate)
        """
    )

    assert result.stdout.splitlines() == ["['NullHandler']", "True", "True"]


def test_basic_config_shows_the_messages_of_commol() -> None:
    result = _run_python(
        """
        import logging

        from commol import Aggregate, ModelBuilder, Simulation

        logging.basicConfig(level=logging.INFO)
        model = (
            ModelBuilder(name="AB", version="1.0")
            .add_bin(id="A", name="A")
            .add_bin(id="B", name="B")
            .add_parameter(id="k1", value=0.1)
            .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1")
            .build(
                Aggregate(bin_amounts={"A": 100, "B": 0}),
                engine="DifferenceEquations",
            )
        )
        Simulation(model).run(num_steps=3)
        """
    )

    assert "INFO:commol:Running simulation for 3 steps." in (result.stderr.splitlines())


def test_a_record_names_the_public_module(caplog: pytest.LogCaptureFixture) -> None:
    """A calibration logs as ``commol``, the module of `Calibrator`, and a
    sensitivity analysis as ``commol.sensitivity``, never as a private path."""
    with caplog.at_level(logging.INFO, logger="commol"):
        flow_calibrator([10.0, 11.0, 9.0], windowed=True).run()
        SensitivityAnalysis(Simulation(build(abc_builder())), abc_rates())

    assert {record.name for record in caplog.records} == {
        "commol",
        "commol.sensitivity",
    }
