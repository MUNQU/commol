"""A long method logs each step it finishes: the replicates of a bootstrap,
the stages of a profile and an identifiability method; the records change no
result."""

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from commol.identifiability import Collinearity, StructuralRank
from commol.profiles import ProfileLikelihood, Threshold
from commol.resampling import BayesianBootstrap
from tests.support import line_calibrator

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
