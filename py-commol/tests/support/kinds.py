"""The test model kind: a kind written only against `ModelStructure`.

Each output starts at a value and decays exponentially at the rate of a
parameter, ``x(t) = initial * exp(-rate * t)``. Its engine is ``TestEngine``
in the engine's list, which the extension has only when it is built with the
``test-kind`` feature (``maturin develop --release --features test-kind``).
"""

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, ConfigDict

from commol import Aggregate, Model, ModelStructure
from commol.commol_rs import _commol_rs as commol_rs
from commol.context.description import (
    Equation,
    EquationGroup,
    Equations,
    Section,
)
from commol.context.structure import ReportSections, SeriesLabel
from commol.context.parameter import Parameter
from commol.context.engine import ModelEngine

if TYPE_CHECKING:
    from commol.context.inference.sample import ParameterSummary

TEST_ENGINE = "TestEngine"
TEST_STOCHASTIC_ENGINE = "TestStochasticEngine"
"""Each unit of an output leaves after an exponential time; a run depends on
its seed, and its mean is the curve of `TEST_ENGINE`."""


def require_test_kind() -> None:
    """Fail with how to build the extension when it has no test kind."""
    if TEST_ENGINE not in commol_rs.engine.EngineType.names():
        raise RuntimeError(
            "The extension has no test model kind. Build it with "
            "`maturin develop --release --features test-kind`."
        )


class DecayOutput(BaseModel):
    """One output of a test model: its initial value and its rate parameter."""

    model_config = ConfigDict(extra="forbid")

    id: str
    initial: float
    rate: str


class DecayStructure(ModelStructure):
    """The structure of the test model kind."""

    kind: Literal["test"] = "test"
    outputs: list[DecayOutput]

    def check_model(self, model: "Model[Self]") -> None:
        """The process is empty, and every rate names a parameter of the model."""
        if not model.process.is_empty():
            raise ValueError(
                "A test model runs no process: its outputs decay on their own. "
                "Leave the process empty."
            )
        parameter_ids = {parameter.id for parameter in model.parameters}
        unknown = sorted(
            output.rate for output in self.outputs if output.rate not in parameter_ids
        )
        if unknown:
            raise ValueError(f"Unknown rate parameters: {unknown}")

    def outputs_by_source(
        self,
        model: "Model[Self]",  # noqa: ARG002
    ) -> dict[str, list[str]]:
        """Each output is its own source."""
        return {output.id: [output.id] for output in self.outputs}

    def series_label(
        self,
        model: "Model[Self]",  # noqa: ARG002
        series_id: str,
    ) -> SeriesLabel:
        """Outputs are named "Output <id>" and measured in "unit"."""
        return SeriesLabel(f"Output {series_id}", "unit")

    def describe(
        self,
        model: "Model[Self]",  # noqa: ARG002
    ) -> list[Section]:
        """A ``dynamics`` section: ``dx/dt = -k * x`` for each output."""
        return [
            Section(
                key="dynamics",
                title="Dynamics",
                blocks=[
                    Equations(
                        groups=[
                            EquationGroup(
                                equations=[
                                    Equation(
                                        left=f"d{output.id}/dt",
                                        right=f"-{output.rate} * {output.id}",
                                        left_latex=f"\\frac{{d{output.id}}}{{dt}}",
                                        right_latex=f"-{output.rate} {output.id}",
                                    )
                                    for output in self.outputs
                                ]
                            )
                        ]
                    )
                ],
            )
        ]

    def report_sections(
        self,
        model: "Model[Self]",  # noqa: ARG002
        calibrated_ids: set[str],
        parameter_statistics: "Mapping[str, ParameterSummary] | None",
    ) -> ReportSections:
        """A section ``outputs`` with the initial value of each output."""
        return {
            "outputs": {
                output.id: {
                    "initial": output.initial,
                    "rate_calibrated": output.rate in calibrated_ids,
                    "has_interval": output.rate in (parameter_statistics or {}),
                }
                for output in self.outputs
            }
        }


def decay_model(
    rate: float | None = 0.5,
    initial: float = 10.0,
    engine: str = TEST_ENGINE,
) -> Model[DecayStructure]:
    """A test model with one output ``x`` that decays at the parameter ``k``."""
    require_test_kind()
    return Model[DecayStructure](
        name="decay",
        parameters=[Parameter(id="k", value=rate)],
        engine=ModelEngine(type=engine),
        structure=DecayStructure(
            outputs=[DecayOutput(id="x", initial=initial, rate="k")]
        ),
    )


class UnstratifiedVariant(Aggregate):
    """
    The variant of the tests: an aggregate structure of a process with no
    stratifications, which the engine checks (``AggregateVariant::Test``).
    """

    variant: Literal["test"] = "test"


def decay(initial: float, rate: float, position: float) -> float:
    """The value of an output of a test model at a position."""
    return initial * math.exp(-rate * position)
