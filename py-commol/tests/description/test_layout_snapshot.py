"""The default and the ODD layouts of a description, of models that take
each branch of their blocks."""

from collections.abc import Callable

import pytest

from commol._model.model import Model
from tests.support.described import stochastic_with_settings, variant_with_a_series
from tests.support.description import matches_snapshot
from tests.support.kinds import decay_model


@pytest.mark.parametrize("layout", ["default", "odd"])
@pytest.mark.parametrize(
    "model", [variant_with_a_series, stochastic_with_settings, decay_model]
)
def test_a_layout_matches_its_snapshot(model: Callable[[], Model], layout: str) -> None:
    rendered = model().describe(layout).render("json") + "\n"  # ty: ignore[invalid-argument-type]

    matches_snapshot(f"layout_{model.__name__}_{layout}.json", rendered)
