"""The test uncertainty method, registered so that the entry point is tested
with no real method.

`StartsSample` runs a multi-start calibration and returns the ends of its
starts as a sample of equal weights. It is not a statistical method: it only
exercises `Calibrator.uncertainty` and the registry.
"""

from pydantic import BaseModel, ConfigDict, Field

from commol import (
    Calibrator,
    ParameterSample,
    SampleMethod,
    UncertaintyMethod,
)
from commol.api.inference.registry import register


class StartsResult(BaseModel):
    """The result of `StartsSample`: the ends of the starts, as a sample."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample: ParameterSample


class StartsSample(UncertaintyMethod[StartsResult]):
    """The ends of `starts` starts of the calibration."""

    starts: int = Field(default=3, ge=1)


def run_starts_sample(calibrator: Calibrator, method: StartsSample) -> StartsResult:
    """Run `method` on `calibrator`."""
    result = calibrator.run(starts=method.starts)
    parameters = calibrator.problem.parameters
    return StartsResult(
        sample=ParameterSample.equally_weighted(
            [p.id for p in parameters],
            [(p.min_bound, p.max_bound) for p in parameters],
            [list(start.end.values()) for start in result.starts],
            SampleMethod(name="starts", settings={"starts": method.starts}),
        )
    )


register(StartsSample, run_starts_sample)
