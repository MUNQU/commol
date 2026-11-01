"""A calibration logs each start that finishes."""

import logging
import re

import pytest

from tests.support import line_calibrator

VALUES = [7.4, 8.6, 11.3, 12.8, 15.1, 16.4, 19.9, 20.6, 23.5, 24.7]

START = re.compile(r"Start (\d+) (finished|failed) \((\d+)/(\d+)\)(.*)")


def test_each_finished_start_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    calibrator = line_calibrator(VALUES, seed=3)
    caplog.clear()

    with caplog.at_level(logging.INFO, logger="commol"):
        calibrator.run(starts=4)

    starts = [
        match
        for record in caplog.records
        if record.name == "commol"
        and record.levelno == logging.INFO
        and (match := START.fullmatch(record.getMessage()))
    ]
    assert [int(match[3]) for match in starts] == [1, 2, 3, 4]
    assert {int(match[4]) for match in starts} == {4}
    assert sorted(int(match[1]) for match in starts) == [0, 1, 2, 3]
    for match in starts:
        assert match[2] == "finished"
        assert re.fullmatch(r": loss \S+, (converged|did not converge)", match[5])
    messages = caplog.messages
    assert messages.index("Fitting 4 starts") < messages.index(starts[0][0])
    assert messages[0].startswith("Starting calibration")
    assert messages[-1].startswith("Calibration finished")
