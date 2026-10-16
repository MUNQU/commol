"""The engine of a model and its settings.

The engine's list of engines (``commol_rs.engine.EngineType``) is the source
of truth: it knows every name, the kinds of structure each engine runs,
validates the settings and says what each engine means. Python keeps the
names of the engines as the ``Literal`` `EngineName`, for static typing of
`ModelBuilder.build`, and a test checks that it equals the list.
"""

from typing import Any, Literal, Self, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

try:
    from commol.commol_rs import _commol_rs as commol_rs
except ImportError as e:
    raise ImportError(f"Error importing Rust extension: {e}") from e


EngineName: TypeAlias = Literal[
    "DifferenceEquations",
    "DifferentialEquations",
    "StochasticEvents",
    "StochasticSteps",
]
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


Rounding: TypeAlias = Literal["largest_remainder", "random"]
"""How a stochastic engine makes whole the initial amounts that are not:
``"largest_remainder"`` gives the same whole amounts in every run, which sum
to the total; ``"random"`` draws each run's amounts from a multinomial of the
total with the fractions of the model, so the mean start equals them."""


class StochasticEventsSettings(BaseModel):
    """
    Settings of the ``StochasticEvents`` engine.

    Attributes
    ----------
    method : Literal["direct", "tau_leaping"]
        ``"direct"``: Gillespie's direct method, one event at a time, exact.
        ``"tau_leaping"``: a Poisson number of events of each flow in each
        leap, an approximation that is faster when events are many.
    tau : float | None
        The length of a leap of ``"tau_leaping"``. None chooses each leap as
        Cao, Gillespie and Petzold (2006) do, from `epsilon`.
    epsilon : float
        How much a propensity may change in one chosen leap, relative to its
        value.
    tolerance : float
        Relative tolerance of the time of an event when a rate reads ``t``.
    rounding : Rounding | None
        How to make whole the initial amounts that are not. None requires
        whole amounts.
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["direct", "tau_leaping"] = Field(
        default="direct", description="Simulation method."
    )
    tau: float | None = Field(
        default=None, gt=0.0, description="Length of a leap of tau-leaping."
    )
    epsilon: float = Field(
        default=0.03, gt=0.0, lt=1.0, description="Bound of an adaptive leap."
    )
    tolerance: float = Field(
        default=1e-8, gt=0.0, description="Relative tolerance of an event time."
    )
    rounding: Rounding | None = Field(
        default=None, description="How to make the initial amounts whole."
    )


class StochasticStepsSettings(BaseModel):
    """
    Settings of the ``StochasticSteps`` engine.

    Attributes
    ----------
    rounding : Rounding | None
        How to make whole the initial amounts that are not. None requires
        whole amounts.
    """

    model_config = ConfigDict(extra="forbid")

    rounding: Rounding | None = Field(
        default=None, description="How to make the initial amounts whole."
    )


EngineSettings: TypeAlias = (
    OdeSolver | StochasticEventsSettings | StochasticStepsSettings
)
"""Settings of an engine: one class per engine that takes settings."""

_SETTINGS: dict[str, type[EngineSettings]] = {
    "DifferentialEquations": OdeSolver,
    "StochasticEvents": StochasticEventsSettings,
    "StochasticSteps": StochasticStepsSettings,
}
"""The class of the settings of each engine that takes settings."""


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
        Settings of that engine; None uses its defaults.
        ``DifferentialEquations`` takes `OdeSolver`, ``StochasticEvents``
        takes `StochasticEventsSettings` and ``StochasticSteps`` takes
        `StochasticStepsSettings`.
    """

    model_config = ConfigDict(extra="forbid")

    type: str = Field(..., description="Name of the engine.")
    settings: EngineSettings | None = Field(
        default=None, description="Settings of the engine; None uses its defaults."
    )

    @field_validator("settings", mode="before")
    @classmethod
    def settings_of_the_engine(cls, value: Any, info: ValidationInfo) -> Any:
        """Read settings given as a mapping with the class of their engine."""
        settings_class = _SETTINGS.get(info.data.get("type", ""))
        if isinstance(value, dict) and settings_class is not None:
            return settings_class.model_validate(value)
        return value

    @model_validator(mode="after")
    def check_settings(self) -> Self:
        """Ask the engine's list whether it has the engine and the settings."""
        settings_json = (
            None if self.settings is None else self.settings.model_dump_json()
        )
        commol_rs.engine.EngineType(self.type).check_settings(settings_json)
        return self
