"""What every method that reads blocks does with the length of a block, and
with a fit it cannot run."""

import logging
import re
from collections.abc import Callable

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from commol import Calibrator
from commol._calibration.method import UncertaintyMethod
from commol._data.result_model import ResultModel
from commol.conformal import EnbPI, JackknifePlus
from commol.conformal.results import ConformalResult
from commol.distributions import Priors
from commol.local import Sandwich
from commol.local.results import SandwichResult
from commol.resampling import (
    BayesianBootstrap,
    BlockJackknife,
    ResidualBootstrap,
    WildBootstrap,
)
from commol.resampling.results import JackknifeResult, BootstrapResult
from commol.smc import BootstrapCoverageEta, GibbsPosterior, InformationMatchingEta
from commol.smc.results import GibbsResult
from tests.support import (
    flow_calibrator,
    no_run_calibrator,
    normal_values,
)

Build = Callable[[int], UncertaintyMethod[ResultModel]]

POSITIONS = (1, 2, 3, 4, 8, 9, 12, 13, 14, 15, 20, 21)


def _gibbs(eta: InformationMatchingEta | BootstrapCoverageEta) -> GibbsPosterior:
    return GibbsPosterior(
        priors=Priors.uniform_on_bounds(),
        eta=eta,
        particles=50,
        target_cess=0.5,
        resample_threshold=0.5,
        stay_probability=0.05,
    )


METHODS: dict[str, Build] = {
    "BayesianBootstrap": lambda n: BayesianBootstrap(replicates=20, block_length=n),
    "ResidualBootstrap": lambda n: ResidualBootstrap(
        replicates=20, block_length=n, residual_scale="additive", scheme="moving"
    ),
    "WildBootstrap": lambda n: WildBootstrap(
        replicates=20, block_length=n, multipliers="mammen"
    ),
    "BlockJackknife": lambda n: BlockJackknife(block_length=n),
    "JackknifePlus": lambda n: JackknifePlus(
        block_length=n,
        positions=[22.0],
        score="absolute",
        by="series",
        confidence_level=0.5,
    ),
    "EnbPI": lambda n: EnbPI(
        replicates=20,
        block_length=n,
        positions=[22.0],
        score="absolute",
        by="series",
        confidence_level=0.5,
    ),
    "Sandwich": lambda n: Sandwich(block_length=n, draws=2),
    "InformationMatchingEta": lambda n: _gibbs(InformationMatchingEta(block_length=n)),
    "BootstrapCoverageEta": lambda n: _gibbs(
        BootstrapCoverageEta(replicates=10, block_length=n, confidence_level=0.8)
    ),
}
"""Each method that reads blocks, of a given block length."""


def _gapped() -> Calibrator:
    """A constant flow observed at `POSITIONS`, where blocks of distinct
    positions are not windows of the axis."""
    calibrator = flow_calibrator(normal_values(21, 6, sd=1.0), windowed=True)
    data = [o for o in calibrator.problem.observed_data if o.at in POSITIONS]
    return Calibrator(
        calibrator.simulation,
        calibrator.problem.model_copy(update={"observed_data": data}),
    )


def _numbers(result: ResultModel) -> list[float]:
    """The numbers of `result` that its blocks move."""
    match result:
        case BootstrapResult():
            draws = np.array(result.sample.values)[:, 0]
            return [float(draws.mean()), float(draws.std())]
        case JackknifeResult():
            return [result.standard_error["c"]]
        case ConformalResult():
            series = result.intervals.series["B"]
            return [series.lower[0], series.upper[0]]
        case SandwichResult():
            return [result.covariance[0][0]]
        case GibbsResult():
            return [result.eta]
    raise AssertionError(type(result))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("BayesianBootstrap", [10.027407498552234, 0.14912443271199055]),
        ("ResidualBootstrap", [9.89863173597139, 0.22361735832265844]),
        ("WildBootstrap", [10.024741857517988, 0.18436298220720343]),
        ("BlockJackknife", [0.21859948734663937]),
        ("JackknifePlus", [8.948190508140305, 11.251761516542231]),
        ("EnbPI", [8.942835542278333, 11.127681572838481]),
        ("Sandwich", [0.0007440276420141215]),
        ("InformationMatchingEta", [0.6876683155579488]),
        ("BootstrapCoverageEta", [1.7255053050550624]),
    ],
    ids=METHODS,
)
def test_a_given_length_sets_the_blocks_a_method_draws(
    name: str, expected: list[float]
) -> None:
    calibrator = _gapped()

    given = _numbers(calibrator.uncertainty(METHODS[name](3)))
    ones = _numbers(calibrator.uncertainty(METHODS[name](1)))

    assert given == pytest.approx(expected, rel=1e-5)
    # Blocks of one position give another result, so these are not theirs.
    assert given != pytest.approx(ones, rel=1e-3)


@pytest.mark.parametrize("name", METHODS)
def test_a_block_length_is_required(name: str) -> None:
    method = METHODS[name](3)
    owner = method.eta if isinstance(method, GibbsPosterior) else method
    assert isinstance(owner, BaseModel)
    settings = {
        field: getattr(owner, field)
        for field in type(owner).model_fields
        if field not in ("block_length", "kind")
    }

    with pytest.raises(ValidationError) as raised:
        type(owner)(**settings)

    (error,) = raised.value.errors()
    assert (error["type"], error["loc"]) == ("missing", ("block_length",))


@pytest.mark.parametrize("name", METHODS)
def test_a_block_longer_than_the_positions_is_an_error(name: str) -> None:
    message = (
        "block_length is 13; it must be between 1 and the number of distinct "
        "positions of the observations, 12"
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        _gapped().uncertainty(METHODS[name](13))


@pytest.mark.parametrize(
    "name", ["BlockJackknife", "JackknifePlus", "EnbPI", "BootstrapCoverageEta"]
)
def test_a_block_longer_than_the_positions_is_an_error_before_the_fit(
    name: str, caplog: pytest.LogCaptureFixture
) -> None:
    calibrator = _gapped()
    caplog.clear()

    with (
        caplog.at_level(logging.INFO, logger="commol"),
        pytest.raises(ValueError, match="block_length is 13"),
    ):
        calibrator.uncertainty(METHODS[name](13))

    assert not [r for r in caplog.messages if r.startswith("Fitting")]


@pytest.mark.parametrize(
    "name", ["BayesianBootstrap", "ResidualBootstrap", "WildBootstrap"]
)
def test_a_bootstrap_checks_the_length_before_the_run_of_the_fit(name: str) -> None:
    with pytest.raises(ValueError, match="^block_length is 21"):
        no_run_calibrator().uncertainty(METHODS[name](21))


@pytest.mark.parametrize(
    "name", [n for n in METHODS if n not in ("Sandwich", "InformationMatchingEta")]
)
def test_a_fit_with_no_run_is_an_error_of_every_method_that_refits(name: str) -> None:
    calibrator = no_run_calibrator()
    assert calibrator.run().best_parameters["k"] > 1.0

    with pytest.raises(ValueError, match="^The fit has no run: .*compartment 'A'"):
        calibrator.uncertainty(METHODS[name](3))
