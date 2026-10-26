"""The sections an aggregate model fills, of models that take each branch."""

from collections.abc import Callable

import pytest

from commol._model.description import Matrix
from commol._model.model import Model
from tests.support.described import identity_groups, weighted_groups
from tests.support.description import matches_snapshot

KEYS = ("entities", "initialization", "design_concepts")


@pytest.mark.parametrize("model", [weighted_groups, identity_groups])
def test_the_sections_of_an_aggregate_match_their_snapshot(
    model: Callable[[], Model],
) -> None:
    built = model()
    kind = {section.key: section for section in built.structure.describe(built)}
    neighbors = [
        block
        for block in kind["dynamics"].blocks
        if block.heading == "Neighbors" or isinstance(block, Matrix)
    ]

    rendered = "\n".join(
        [
            *(kind[key].model_dump_json(indent=1) for key in KEYS),
            *(block.model_dump_json(indent=1) for block in neighbors),
        ]
    )

    matches_snapshot(f"aggregate_{model.__name__}.json", rendered + "\n")
