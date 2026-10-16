"""What every uncertainty method takes and returns: the configuration base
and the weighted parameter sample."""

from commol.context.inference.method import UncertaintyMethod
from commol.context.inference.sample import (
    ParameterSample,
    ParameterSummary,
    SampleMethod,
)

__all__ = [
    "ParameterSample",
    "ParameterSummary",
    "SampleMethod",
    "UncertaintyMethod",
]
