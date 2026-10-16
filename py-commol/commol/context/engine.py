"""The engine of a model and its settings.

The engine's list of engines (``commol_rs.engine.EngineType``) is the source
of truth: it knows every name, the kinds of structure each engine runs,
validates the settings and says what each engine means. Python keeps the
names of the engines as the ``Literal`` `EngineName`, for static typing of
`ModelBuilder.build`, and a test checks that it equals the list.
"""

from typing import Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

try:
    from commol.commol_rs import _commol_rs as commol_rs
except ImportError as e:
    raise ImportError(f"Error importing Rust extension: {e}") from e


EngineName: TypeAlias = Literal["DifferenceEquations", "DifferentialEquations"]
"""Names of the engines of the list. An engine names a method; the structure
of the model says which kinds of model it runs."""


class OdeSolver(BaseModel):
    """
    Settings of the ``DifferentialEquations`` engine.

    Attributes
    ----------
    method : Literal["tsit45", "bdf"]
        ``"tsit45"``: explicit Tsitouras 5(4) Runge-Kutta, for non-stiff
        models. ``"bdf"``: backward differentiation formulas, for stiff models.
    rtol : float
        Relative tolerance of each state component.
    atol : float
        Absolute tolerance of each state component, in the unit of the bins.
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["tsit45", "bdf"] = Field(
        default="tsit45", description="Integration method."
    )
    rtol: float = Field(default=1e-6, gt=0.0, description="Relative tolerance.")
    atol: float = Field(default=1e-6, gt=0.0, description="Absolute tolerance.")


EngineSettings: TypeAlias = OdeSolver
"""Settings of an engine: one class per engine that takes settings."""


class ModelEngine(BaseModel):
    """
    The engine that runs a model, and its settings.

    Attributes
    ----------
    type : str
        Name of an engine of the engine's list, such as
        ``"DifferenceEquations"`` (discrete time) or
        ``"DifferentialEquations"`` (continuous time). The engine must run the
        kind of the structure of the model.
    settings : EngineSettings | None
        Settings of that engine; None uses its defaults. Only
        ``DifferentialEquations`` takes settings (`OdeSolver`).
    """

    model_config = ConfigDict(extra="forbid")

    type: str = Field(..., description="Name of the engine.")
    settings: EngineSettings | None = Field(
        default=None, description="Settings of the engine; None uses its defaults."
    )

    @model_validator(mode="after")
    def check_settings(self) -> Self:
        """Ask the engine's list whether it has the engine and the settings."""
        settings_json = (
            None if self.settings is None else self.settings.model_dump_json()
        )
        commol_rs.engine.EngineType(self.type).check_settings(settings_json)
        return self
