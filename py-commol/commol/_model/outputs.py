"""The outputs a source id of a model stands for: the outputs of a bin or an
accumulator, as the structure of the model groups them."""

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from commol._model.model import Model


def outputs_for(model: "Model", id: str) -> list[str]:
    """
    Names of the outputs a source id expands into.

    The structure of the model groups its outputs by source
    (`ModelStructure.outputs_by_source`): for an aggregate model, by bin and
    accumulator.

    Parameters
    ----------
    model : Model
        The model whose outputs the results hold.
    id : str
        A source id, such as a bin id or an accumulator id.

    Returns
    -------
    list[str]
        Output names for that id, in the order the engine reports them.

    Raises
    ------
    KeyError
        If `id` is not a source of the model.
    """
    outputs = model.structure.outputs_by_source(model)
    if id not in outputs:
        raise KeyError(
            f"'{id}' is not a bin or accumulator of model '{model.name}'. "
            f"Available: {sorted(outputs)}"
        )
    return outputs[id]


def series_names(model: "Model", ids: Iterable[str]) -> list[str]:
    """
    The outputs the ids stand for, in the order given.

    A source id, such as a bin or accumulator id, stands for all its outputs
    (`outputs_for`); an output name stands for itself.

    Raises
    ------
    KeyError
        If an id is neither a source nor an output of the model.
    """
    by_source = model.structure.outputs_by_source(model)
    outputs = [name for names in by_source.values() for name in names]
    names: list[str] = []
    for id in ids:
        if id in by_source:
            names.extend(by_source[id])
        elif id in outputs:
            names.append(id)
        else:
            raise KeyError(
                f"'{id}' is not a bin, accumulator or output of model "
                f"'{model.name}'. Bins and accumulators: {sorted(by_source)}. "
                f"Outputs: {outputs}"
            )
    return names
