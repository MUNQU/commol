"""Record how long each test takes, for the time budget of ``test_rules.py``."""

import pytest

DURATIONS: dict[str, float] = {}
LAST = "test_slow_tests_are_listed"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Run the check of the time budget after every other test."""
    items.sort(key=lambda item: item.name == LAST)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Add the time of each phase of a test: setup, call and teardown."""
    DURATIONS[report.nodeid] = DURATIONS.get(report.nodeid, 0.0) + report.duration
