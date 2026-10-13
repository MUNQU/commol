"""Every public Pydantic class rejects a field it does not define."""

import inspect

import pytest
from pydantic import BaseModel, ValidationError

import commol
from commol import CalibrationConstraint, ObservedDataPoint


def _public_models() -> list[type[BaseModel]]:
    return [
        obj
        for name in commol.__all__
        if inspect.isclass(obj := getattr(commol, name)) and issubclass(obj, BaseModel)
    ]


@pytest.mark.parametrize("model", _public_models(), ids=lambda model: model.__name__)
def test_a_public_model_forbids_unknown_fields(model: type[BaseModel]) -> None:
    assert model.model_config.get("extra") == "forbid"


def test_an_old_name_is_named_in_the_error() -> None:
    with pytest.raises(ValidationError, match="window_steps"):
        ObservedDataPoint(at=7, output="A", value=1.0, window_steps=7)


def test_a_misspelt_name_is_named_in_the_error() -> None:
    with pytest.raises(ValidationError, match="time_steps"):
        CalibrationConstraint(id="c", expression="1 - A", time_steps=[1])
