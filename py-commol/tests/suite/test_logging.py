"""Check that commol configures no logging and that the application sees its messages.

The test suite sends the messages of commol to the standard error stream
(``tests/support/logs.py``), so each test runs its check in a fresh
interpreter. The script of a fresh interpreter cannot import ``tests``, whose
``__init__`` adds that handler, so it builds its model by hand.
"""

import subprocess
import sys
import textwrap


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

    assert "INFO:commol.api.simulation:Running simulation for 3 steps." in (
        result.stderr.splitlines()
    )
