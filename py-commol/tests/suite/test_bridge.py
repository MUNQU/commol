"""The two shapes of a value agree: each data class whose JSON the engine
reads crosses whole, and the engine reads back its ``model_dump()``."""

import json

import pytest

from commol._data.engine_json import engine_json
from commol.commol_rs import _commol_rs as commol_rs
from tests.suite.test_methods import calls_the_engine, method_classes
from tests.support.data_objects import EXAMPLES


@pytest.mark.parametrize("name", commol_rs.json.objects())
def test_every_data_class_crosses_as_json(name: str) -> None:
    value = EXAMPLES[name]()
    sent = engine_json(value)

    read = commol_rs.json.as_read(name, sent)

    assert type(value).__name__ == name
    assert json.loads(read) == json.loads(sent)


def test_every_method_crosses_as_json() -> None:
    read = set(commol_rs.json.objects())

    missing = sorted(
        method.__name__
        for method in method_classes()
        if method._runner is not None
        and calls_the_engine(method._runner)
        and method.__name__ not in read
    )

    assert not missing, "Give each method a settings type of the engine:\n" + (
        "\n".join(missing)
    )


def test_the_engine_reads_no_class_it_does_not_know() -> None:
    with pytest.raises(ValueError, match="The engine reads no object 'Unknown'"):
        commol_rs.json.as_read("Unknown", "{}")


def test_an_unknown_field_is_named_with_its_path() -> None:
    problem = json.loads(engine_json(EXAMPLES["CalibrationProblem"]()))
    problem["observed_data"][0]["extra"] = 1

    with pytest.raises(
        ValueError, match=r"^observed_data\[0\]\.extra: unknown field `extra`"
    ):
        commol_rs.json.as_read("CalibrationProblem", json.dumps(problem))
