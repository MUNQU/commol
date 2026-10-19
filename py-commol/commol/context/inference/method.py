"""The configuration of an inference method, generic in its result.

Three kinds of method share one base and one registry: an uncertainty method
(``Calibrator.uncertainty``), an identifiability method
(``Calibrator.identifiability``) and a sensitivity method
(``SensitivityAnalysis.run``). Each entry point takes only its kind.
"""

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

ResultT = TypeVar("ResultT", bound=BaseModel, covariant=True)


class InferenceMethod(BaseModel, Generic[ResultT]):
    """
    The configuration of a method that reads a calibration problem or a
    sensitivity analysis, whose result is `ResultT`.

    A method subclasses one of its kinds, `UncertaintyMethod`,
    `IdentifiabilityMethod` or `SensitivityMethod`, with its result type, and
    registers its runner (`commol.api.inference.registry.register`).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class UncertaintyMethod(InferenceMethod[ResultT], Generic[ResultT]):
    """
    The configuration of an uncertainty method, whose result is `ResultT`.

    A method subclasses it with its result type, for example
    ``class BayesianBootstrap(UncertaintyMethod[BootstrapResult])``, and
    registers its runner (`commol.api.inference.registry.register`).
    ``Calibrator.uncertainty(method)`` returns that result type, so a static
    type checker infers it from the configuration's type.
    """


class IdentifiabilityMethod(InferenceMethod[ResultT], Generic[ResultT]):
    """
    The configuration of an identifiability method, whose result is
    `ResultT`.

    ``Calibrator.identifiability(method)`` runs it, and returns its result
    type. The methods of commol return an `IdentifiabilityReport`.
    """


class SensitivityMethod(InferenceMethod[ResultT], Generic[ResultT]):
    """
    The configuration of a sensitivity method, whose result is `ResultT`.

    ``SensitivityAnalysis.run(method)`` runs it on the inputs of the
    analysis, with no data, and returns its result type.
    """
