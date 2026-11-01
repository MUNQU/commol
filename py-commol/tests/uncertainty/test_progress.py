"""A long method logs each step it finishes: the replicates of a bootstrap,
the stages of a profile, the steps of every other long method and an
identifiability method; the records change no result, and a Ctrl-C stops the
steps."""

import _thread
import logging
import re
import signal
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import FrameType

import pytest

from commol import Simulation
from commol.conformal import EnbPI, JackknifePlus, RollingConformal
from commol.distributions import Normal, Priors
from commol.identifiability import Collinearity, StructuralRank
from commol.local import GibbsTarget, LaplaceMixture
from commol.observations import NormalObservations
from commol.parallelism import threads
from commol.profiles import ProfileLikelihood, Threshold
from commol.quantities import ValueAt
from commol.resampling import BayesianBootstrap, BlockJackknife
from commol.sensitivity import (
    PRCC,
    Morris,
    PolynomialChaos,
    Propagation,
    SensitivityAnalysis,
    Sobol,
)
from commol.smc import (
    ABCSMC,
    ABCRejection,
    BayesianPosterior,
    BootstrapCoverageEta,
    GibbsPosterior,
    LossDistance,
    SeriesDistance,
)
from commol.uncertainty import Predictive
from tests.support import abc_builder, abc_rates, build, line_calibrator

VALUES = [7.4, 8.6, 11.3, 12.8, 15.1, 16.4, 19.9, 20.6, 23.5, 24.7]

REPLICATE = re.compile(r"Replicate (\d+) (finished|failed) \((\d+)/5\)(.*)")
START_OF_REPLICATE = re.compile(
    r"Replicate (\d+): start (\d+) finished \((\d+)/2\): loss \S+, .*"
)


def _messages(
    caplog: pytest.LogCaptureFixture, name: str, level: int = logging.INFO
) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == name and record.levelno == level
    ]


def test_each_fitted_replicate_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    calibrator = line_calibrator(VALUES, seed=5)
    caplog.clear()

    with caplog.at_level(logging.DEBUG, logger="commol"):
        calibrator.uncertainty(
            BayesianBootstrap(replicates=5, block_length=1, starts=2)
        )

    info = _messages(caplog, "commol.resampling")
    replicates = [match for m in info if (match := REPLICATE.fullmatch(m))]
    assert [int(match[3]) for match in replicates] == [1, 2, 3, 4, 5]
    assert sorted(int(match[1]) for match in replicates) == [0, 1, 2, 3, 4]
    for match in replicates:
        assert re.fullmatch(r": loss \S+, (converged|did not converge)", match[4])
    assert info.index("Fitting 5 replicates") < info.index(replicates[0][0])
    # The fit of the original data is a stage of its own, at INFO.
    assert info.index("Fitting 2 starts") < info.index("Fitting 5 replicates")

    debug = _messages(caplog, "commol.resampling", logging.DEBUG)
    starts = [match for m in debug if (match := START_OF_REPLICATE.fullmatch(m))]
    assert len(starts) == 10
    for replicate in range(5):
        counts = [int(match[3]) for match in starts if int(match[1]) == replicate]
        assert counts == [1, 2]
        assert f"Replicate {replicate}: fitting 2 starts" in debug
    assert not any(START_OF_REPLICATE.fullmatch(m) for m in info)


def test_each_stage_of_a_profile_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    calibrator = line_calibrator(VALUES, seed=5)
    caplog.clear()

    with caplog.at_level(logging.INFO, logger="commol"):
        result = calibrator.uncertainty(
            ProfileLikelihood(
                parameters=["c", "B"],
                threshold=Threshold.bootstrap_calibrated(
                    BayesianBootstrap(replicates=5, block_length=1), 0.9
                ),
                max_steps=100,
            )
        )

    messages = _messages(caplog, "commol.profiles")
    fitted = len(result.profiles["c"].replicate_rises or [])
    stages = [
        "Fitting 1 start",
        "Fitting 5 replicates",
        f"Refitting {fitted} replicates with each quantity held",
        "Tracing 4 profile sides",
    ]
    positions = [messages.index(stage) for stage in stages]
    assert positions == sorted(positions)
    replicates = [m for m in messages if REPLICATE.fullmatch(m)]
    assert positions[1] < messages.index(replicates[0])
    assert messages.index(replicates[-1]) < positions[2]
    refits = [
        match
        for m in messages
        if (match := re.fullmatch(rf"Replicate (\d+) refitted \((\d+)/{fitted}\)", m))
    ]
    assert [int(match[2]) for match in refits] == list(range(1, fitted + 1))
    sides = [
        match
        for m in messages
        if (
            match := re.fullmatch(
                r"Profile of (c|B), (lower|upper) side: (\w+) after (\d+) points "
                r"\((\d)/4\)",
                m,
            )
        )
    ]
    assert positions[3] < messages.index(sides[0][0])
    assert [int(match[5]) for match in sides] == [1, 2, 3, 4]
    assert sorted((match[1], match[2]) for match in sides) == [
        ("B", "lower"),
        ("B", "upper"),
        ("c", "lower"),
        ("c", "upper"),
    ]
    for match in sides:
        profile = result.profiles[match[1]]
        side = 0 if match[2] == "lower" else 1
        assert match[3] == profile.ends[side]


@pytest.mark.parametrize("name", ["StructuralRank", "Collinearity"])
def test_an_identifiability_method_logs_its_start_and_end(
    caplog: pytest.LogCaptureFixture, name: str
) -> None:
    calibrator = line_calibrator(VALUES)
    method = (
        StructuralRank(points=3, design="observations")
        if name == "StructuralRank"
        else Collinearity(at=calibrator.run(), max_subset_size=2, threshold=10.0)
    )
    caplog.clear()

    with caplog.at_level(logging.INFO, logger="commol"):
        calibrator.identifiability(method)

    messages = _messages(caplog, "commol.identifiability")
    assert len(messages) == 2
    assert messages[0] == f"{name} started."
    assert re.fullmatch(rf"{name} finished in \d+\.\d\d s\.", messages[1])


def _runs() -> tuple[object, ...]:
    calibrator = line_calibrator(VALUES, seed=7)
    bootstrap = BayesianBootstrap(replicates=4, block_length=1, starts=2)
    return (
        calibrator.run(starts=3).best_parameters,
        calibrator.uncertainty(bootstrap).sample.values,
        calibrator.uncertainty(
            ProfileLikelihood(
                parameters=["c"],
                threshold=Threshold.bootstrap_calibrated(bootstrap, 0.9),
                max_steps=100,
            )
        )
        .profiles["c"]
        .model_dump(),
    )


def test_the_progress_changes_no_result(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="commol"):
        logged = _runs()
    with caplog.at_level(logging.WARNING, logger="commol"):
        quiet = _runs()

    assert logged == quiet


class _Raising(logging.Handler):
    """A handler that raises `error` at the first record of a replicate."""

    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.error = error

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage().startswith("Replicate"):
            raise self.error


@contextmanager
def _handler(error: BaseException) -> Iterator[None]:
    logger = logging.getLogger("commol.resampling")
    handler = _Raising(error)
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)


def test_a_failing_handler_keeps_the_result() -> None:
    calibrator = line_calibrator(VALUES, seed=7)
    bootstrap = BayesianBootstrap(replicates=4, block_length=1)
    expected = calibrator.uncertainty(bootstrap).sample.values

    with _handler(ValueError("a broken handler")):
        assert calibrator.uncertainty(bootstrap).sample.values == expected
    with _handler(KeyboardInterrupt()), pytest.raises(KeyboardInterrupt):
        calibrator.uncertainty(bootstrap)


PRIORS = Priors.uniform_on_bounds()
CONFORMAL = {"score": "absolute", "by": "series", "confidence_level": 0.9}
SMC = {"particles": 16, "target_cess": 0.5, "stay_probability": 0.5}


def _line(*, normal: bool = False) -> object:
    models = {"B": NormalObservations(sd=1.0)} if normal else None
    return line_calibrator(VALUES, seed=5, observation_models=models)


def _analysis() -> SensitivityAnalysis:
    return SensitivityAnalysis(Simulation(build(abc_builder())), abc_rates(), seed=7)


QUANTITY = [ValueAt(series="B", at=5)]


def _predictive() -> None:
    calibrator = line_calibrator(VALUES, seed=5)
    with caplog_off():
        sample = calibrator.uncertainty(
            BayesianBootstrap(replicates=5, block_length=1)
        ).sample
    Predictive(calibrator, sample, [1.0, 2.0])


@contextmanager
def caplog_off() -> Iterator[None]:
    """No record of the loggers of commol within the block."""
    logger = logging.getLogger("commol")
    level = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        logger.setLevel(level)


def _fits(stage: str, unit: str, total: int) -> Callable[[list[str]], None]:
    """A check of one record of the stage, then one per fit of a `unit`, its
    count rising to `total`."""

    def check(messages: list[str]) -> None:
        step = re.compile(rf"{unit} (\d+) (?:finished|failed) \((\d+)/{total}\).*")
        fits = [match for m in messages if (match := step.fullmatch(m))]
        assert [int(match[2]) for match in fits] == list(range(1, total + 1))
        assert sorted(int(match[1]) for match in fits) == list(range(total))
        assert messages.index(stage) < messages.index(fits[0][0])

    return check


def _points(total: int) -> Callable[[list[str]], None]:
    """A check of one record of the stage, then one at each 1 % of the
    `total` points."""

    def check(messages: list[str]) -> None:
        step = re.compile(rf"Ran (\d+) of {total} points \((\d+)%\)")
        runs = [match for m in messages if (match := step.fullmatch(m))]
        percents = [int(match[2]) for match in runs]
        assert len(runs) == min(total, 100)
        assert percents == sorted(set(percents))
        assert runs[-1][0] == f"Ran {total} of {total} points (100%)"
        stage = f"Running the model at {total} points"
        assert messages.index(stage) < messages.index(runs[0][0])

    return check


def _generations(last: str) -> Callable[[list[str]], None]:
    """A check of one record per generation, numbered from 1, on the way to
    the level `last`."""

    def check(messages: list[str]) -> None:
        step = re.compile(
            rf"Generation (\d+): level \S+ toward {last}, effective size \S+, "
            r"acceptance \S+"
        )
        generations = [int(match[1]) for m in messages if (match := step.fullmatch(m))]
        assert generations == list(range(1, len(generations) + 1))
        assert generations

    return check


def _calibrated_eta(messages: list[str]) -> None:
    replicates = [
        int(match[1])
        for m in messages
        if (match := re.fullmatch(r"Replicate \d+ (?:sampled|failed) \((\d)/3\)", m))
    ]
    assert replicates == [1, 2, 3]
    assert messages.index("Fitting 1 start") < messages.index("Sampling 3 replicates")
    _generations("1.000000e0")(messages)


STEPS: dict[str, tuple[str, Callable[[], object], Callable[[list[str]], None]]] = {
    "BlockJackknife": (
        "commol.resampling",
        lambda: _line().uncertainty(BlockJackknife(block_length=2)),
        _fits("Fitting 5 blocks", "Block", 5),
    ),
    "JackknifePlus": (
        "commol.conformal",
        lambda: _line().uncertainty(
            JackknifePlus(**CONFORMAL, block_length=2, positions=[11.0])
        ),
        _fits("Fitting 5 blocks", "Block", 5),
    ),
    "EnbPI": (
        "commol.conformal",
        lambda: _line().uncertainty(
            EnbPI(**CONFORMAL, replicates=10, block_length=2, positions=[11.0])
        ),
        _fits("Fitting 10 replicates", "Replicate", 10),
    ),
    "RollingConformal": (
        "commol.conformal",
        lambda: _line().uncertainty(
            RollingConformal(**CONFORMAL, origins=[5.0, 7.0], positions=[11.0])
        ),
        _fits("Fitting 2 origins", "Origin", 2),
    ),
    "GibbsPosterior": (
        "commol.smc",
        lambda: _line().uncertainty(
            GibbsPosterior(priors=PRIORS, eta=1.0, resample_threshold=0.5, **SMC)
        ),
        _generations("1.000000e0"),
    ),
    "GibbsPosterior with a calibrated eta": (
        "commol.smc",
        lambda: _line().uncertainty(
            GibbsPosterior(
                # Priors near the fit, whose intervals cover it.
                priors=Priors(
                    distributions={
                        "c": Normal(mean=2.0, sd=1.0),
                        "B": Normal(mean=5.0, sd=3.0),
                    }
                ),
                eta=BootstrapCoverageEta(
                    replicates=3, block_length=1, confidence_level=0.9
                ),
                resample_threshold=0.5,
                **SMC,
            )
        ),
        _calibrated_eta,
    ),
    "BayesianPosterior": (
        "commol.smc",
        lambda: _line(normal=True).uncertainty(
            BayesianPosterior(priors=PRIORS, resample_threshold=0.5, **SMC)
        ),
        _generations("1.000000e0"),
    ),
    "ABCSMC": (
        "commol.smc",
        lambda: _line().uncertainty(
            ABCSMC(
                priors=PRIORS,
                distance=SeriesDistance(metric="rmse"),
                min_acceptance=0.1,
                **SMC,
            )
        ),
        _generations(r"\S+"),
    ),
    "ABCRejection": (
        "commol.smc",
        lambda: _line().uncertainty(
            ABCRejection(
                priors=PRIORS, distance=LossDistance(), draws=250, quantile=0.5
            )
        ),
        _points(250),
    ),
    "LaplaceMixture": (
        "commol.local",
        lambda: _line().uncertainty(
            LaplaceMixture(draws=40, target=GibbsTarget(priors=PRIORS, eta=1.0))
        ),
        _points(40),
    ),
    "Propagation": (
        "commol.sensitivity",
        lambda: _analysis().run(Propagation(draws=12, quantities=QUANTITY)),
        _points(12),
    ),
    "PRCC": (
        "commol.sensitivity",
        lambda: _analysis().run(PRCC(draws=12, quantities=QUANTITY)),
        _points(12),
    ),
    "Morris": (
        "commol.sensitivity",
        lambda: _analysis().run(Morris(trajectories=2, levels=4, quantities=QUANTITY)),
        _points(2 * (len(abc_rates().names) + 1)),
    ),
    "Sobol": (
        "commol.sensitivity",
        lambda: _analysis().run(
            Sobol(draws=8, replicates=4, confidence_level=0.9, quantities=QUANTITY)
        ),
        _points(8 * (len(abc_rates().names) + 2)),
    ),
    "PolynomialChaos": (
        "commol.sensitivity",
        lambda: _analysis().run(
            PolynomialChaos(degree=1, draws=8, quantities=QUANTITY)
        ),
        _points(8),
    ),
    "Predictive": ("commol", _predictive, _points(5)),
}


@pytest.mark.parametrize("name", STEPS)
def test_every_long_method_logs_its_steps(
    caplog: pytest.LogCaptureFixture, name: str
) -> None:
    logger, run, check = STEPS[name]

    with caplog.at_level(logging.INFO, logger="commol"):
        run()

    messages = _messages(caplog, logger)
    assert messages
    check(messages)


@contextmanager
def _sigint() -> Iterator[None]:
    """A handler of SIGINT that raises `KeyboardInterrupt` within the block
    only, so a signal that comes after the method has ended raises
    nothing."""
    armed = threading.Event()
    armed.set()

    def interrupt(_signum: int, _frame: FrameType | None) -> None:
        if armed.is_set():
            raise KeyboardInterrupt

    previous = signal.signal(signal.SIGINT, interrupt)
    try:
        yield
    finally:
        armed.clear()
        signal.signal(signal.SIGINT, previous)


class _Interrupting(logging.Handler):
    """A handler that sends SIGINT to the main thread at the first record of
    a start of a replicate, and keeps the time it did."""

    def __init__(self) -> None:
        super().__init__()
        self.at: float | None = None

    def emit(self, record: logging.LogRecord) -> None:
        if self.at is None and re.fullmatch(
            r"Replicate \d+: start \d+ finished .*", record.getMessage()
        ):
            self.at = time.perf_counter()
            _thread.interrupt_main(signal.SIGINT)


def test_an_interrupt_stops_the_steps(caplog: pytest.LogCaptureFixture) -> None:
    calibrator = line_calibrator(VALUES, seed=5)
    # On one thread, each fit of 3000 starts takes about 3 s: a replicate
    # that ran its starts after the interrupt would take that long to stop.
    bootstrap = BayesianBootstrap(replicates=4, block_length=1, starts=3000)
    handler = _Interrupting()
    logger = logging.getLogger("commol.resampling")
    logger.addHandler(handler)
    try:
        with (
            caplog.at_level(logging.DEBUG, logger="commol.resampling"),
            threads(1),
            _sigint(),
            pytest.raises(KeyboardInterrupt),
        ):
            calibrator.uncertainty(bootstrap)
        stopped = time.perf_counter()
    finally:
        logger.removeHandler(handler)

    assert handler.at is not None
    assert stopped - handler.at < 0.3


def test_an_interrupt_stops_a_method_that_logs_nothing() -> None:
    calibrator = line_calibrator(VALUES, seed=5)
    # About 10 s on one thread: 2000 replicates of 8 starts.
    bootstrap = BayesianBootstrap(replicates=2000, block_length=1, starts=8)
    timer = threading.Timer(0.2, _thread.interrupt_main, (signal.SIGINT,))
    began = time.perf_counter()
    try:
        with threads(1), caplog_off(), _sigint(), pytest.raises(KeyboardInterrupt):
            timer.start()
            calibrator.uncertainty(bootstrap)
        elapsed = time.perf_counter() - began
    finally:
        timer.cancel()

    assert elapsed < 2.0
