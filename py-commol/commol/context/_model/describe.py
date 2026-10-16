"""The sections of a description that every kind of model shares, and layouts.

`describe_model` builds the sections of `Model` (overview, schedule,
parameters, outputs), adds the sections of the kind
(`ModelStructure.describe`) and orders them by a layout:

- ``"default"``: the seven sections of a description, after the "Details"
  part of the ODD protocol: overview, entities, dynamics, schedule,
  initialization, parameters, outputs.
- ``"odd"``: the ODD protocol for describing individual- and agent-based
  models (Grimm et al. 2006, updated 2010 and 2020): purpose, entities, process
  overview and scheduling, design concepts, initialization, input data and
  submodels. An element commol cannot fill is marked `TO_BE_WRITTEN`.
"""

from typing import TYPE_CHECKING

try:
    from commol.commol_rs import _commol_rs as commol_rs
except ImportError as e:
    raise ImportError(f"Error importing Rust extension: {e}") from e

from commol.context.description import (
    Block,
    Layout,
    ModelDescription,
    Section,
    Table,
    Text,
)
from commol.context.parameter import SeriesValue

if TYPE_CHECKING:
    from commol.commol_rs._commol_rs import EngineTypeProtocol
    from commol.context.model import Model

TO_BE_WRITTEN = "To be written by the modeller."
"""The marker of an element of the ODD protocol that commol cannot fill."""

DEFAULT_SECTIONS: tuple[tuple[str, str], ...] = (
    ("overview", "Overview"),
    ("entities", "Entities and state variables"),
    ("dynamics", "Dynamics"),
    ("schedule", "Schedule"),
    ("initialization", "Initialization"),
    ("parameters", "Parameters"),
    ("outputs", "Outputs"),
)
"""The keys and titles of the default layout, in order."""

ODD_SECTIONS: tuple[tuple[str, str], ...] = (
    ("purpose", "Purpose and patterns"),
    ("entities", "Entities, state variables and scales"),
    ("process_overview", "Process overview and scheduling"),
    ("design_concepts", "Design concepts"),
    ("initialization", "Initialization"),
    ("input_data", "Input data"),
    ("submodels", "Submodels"),
)
"""The keys and titles of the ODD layout, in the order of the protocol."""

DESIGN_CONCEPTS: tuple[str, ...] = (
    "Basic principles",
    "Emergence",
    "Adaptation",
    "Objectives",
    "Learning",
    "Prediction",
    "Sensing",
    "Interaction",
    "Stochasticity",
    "Collectives",
    "Observation",
)
"""The elements of the design concepts of the ODD protocol (Grimm et al. 2020)."""


def describe_model(model: "Model", layout: Layout) -> ModelDescription:
    """
    The description of a model in a layout.

    Parameters
    ----------
    model : Model
        The model to describe.
    layout : {"default", "odd"}
        The sections and their order.

    Returns
    -------
    ModelDescription
        The description.

    Raises
    ------
    ValueError
        If the layout is not one of these.
    """
    kind_sections = {
        section.key: section for section in model.structure.describe(model)
    }
    if layout == "default":
        sections = _default_layout(model, kind_sections)
    elif layout == "odd":
        sections = _odd_layout(model, kind_sections)
    else:
        raise ValueError(
            f"Unknown layout '{layout}'. The layouts are ['default', 'odd']"
        )
    return ModelDescription(title=model.name, layout=layout, sections=sections)


# ----------------------------------------------------------------------
# Default layout
# ----------------------------------------------------------------------


def _default_layout(model: "Model", kind: dict[str, Section]) -> list[Section]:
    generic: dict[str, list[Block]] = {
        "overview": overview_blocks(model),
        "schedule": [Text(text=_engine(model).schedule)],
        "parameters": parameter_blocks(model),
        "outputs": output_blocks(model),
    }
    return [
        Section(
            key=key,
            title=title,
            blocks=[
                *generic.get(key, []),
                *(kind[key].blocks if key in kind else []),
            ],
        )
        for key, title in DEFAULT_SECTIONS
    ]


def overview_blocks(model: "Model") -> list[Block]:
    """The name, description, kind, engine, settings and axis of a model."""
    structure = model.structure
    kind = structure.kind
    if structure.variant is not None:
        kind = f"{kind} ({structure.variant})"
    settings = (
        "defaults"
        if model.engine.settings is None
        else model.engine.settings.model_dump_json()
    )
    rows: list[list[str | None]] = [
        ["Name", model.name],
        ["Description", model.description],
        ["Version", model.version],
        ["Kind", kind],
        ["Engine", model.engine.type],
        ["Engine settings", settings],
        ["Dynamics", _engine(model).dynamics_form.replace("_", " ")],
        ["Step unit", model.step_unit or "step"],
    ]
    return [
        Table(
            columns=["Field", "Value"],
            rows=[[field, value] for field, value in rows if value is not None],
        )
    ]


def parameter_blocks(model: "Model") -> list[Block]:
    """Every parameter, with its value, unit, description and calibration."""
    if not model.parameters:
        return [Text(text="The model has no parameters.")]
    return [
        Table(
            columns=["Parameter", "Value", "Unit", "Set by calibration", "Description"],
            rows=[
                [
                    parameter.id,
                    _parameter_value(parameter.value),
                    parameter.unit,
                    parameter.value is None,
                    parameter.description,
                ]
                for parameter in model.parameters
            ],
        )
    ]


def _parameter_value(value: float | str | SeriesValue | None) -> str | float | None:
    if isinstance(value, SeriesValue):
        return f"series ({value.mode}, {len(value.data)} points)"
    return value


def output_blocks(model: "Model") -> list[Block]:
    """The outputs a run returns, grouped by the id they come from."""
    outputs = model.structure.outputs_by_source(model)
    return [
        Table(
            columns=["Source", "Outputs"],
            rows=[[source, ", ".join(names)] for source, names in outputs.items()],
        )
    ]


# ----------------------------------------------------------------------
# ODD layout
# ----------------------------------------------------------------------


def _odd_layout(model: "Model", kind: dict[str, Section]) -> list[Section]:
    def blocks_of(key: str) -> list[Block]:
        return list(kind[key].blocks) if key in kind else []

    elements: dict[str, list[Block]] = {
        "purpose": [Text(text=model.description or TO_BE_WRITTEN)],
        "entities": [*blocks_of("entities"), *_scales(model)],
        "process_overview": [
            *_process_overview(model),
            Text(heading="Scheduling", text=_engine(model).schedule),
        ],
        "design_concepts": _design_concepts(model, kind.get("design_concepts")),
        "initialization": blocks_of("initialization") or [Text(text=TO_BE_WRITTEN)],
        "input_data": _input_data(model),
        "submodels": [
            *blocks_of("dynamics"),
            *[
                block.model_copy(update={"heading": "Parameters"})
                for block in parameter_blocks(model)
            ],
            *[
                block.model_copy(update={"heading": "Outputs"})
                for block in output_blocks(model)
            ],
        ],
    }
    return [
        Section(key=key, title=title, blocks=elements[key])
        for key, title in ODD_SECTIONS
    ]


def _scales(model: "Model") -> list[Block]:
    form = _engine(model).dynamics_form
    time = "continuous" if form == "differential_equation" else "discrete steps"
    return [
        Table(
            heading="Scales",
            columns=["Scale", "Value"],
            rows=[
                ["Axis", f"one step is 1 {model.step_unit or 'step'}"],
                ["Advance", f"{time} ({model.engine.type})"],
            ],
        )
    ]


def _process_overview(model: "Model") -> list[Block]:
    transitions = model.process.transitions
    if not transitions:
        return []
    return [
        Table(
            heading="Processes",
            columns=["Process", "From", "To", "When"],
            rows=[
                [
                    transition.id,
                    ", ".join(transition.source) or "none",
                    ", ".join(transition.target) or "none",
                    "impulse" if transition.is_impulse else "every step",
                ]
                for transition in transitions
            ],
        )
    ]


def _design_concepts(model: "Model", kind: Section | None) -> list[Block]:
    """The elements of the design concepts; the kind fills what it knows."""
    filled: dict[str, str] = {}
    if kind is not None:
        for block in kind.blocks:
            if isinstance(block, Table):
                filled.update(
                    {str(row[0]): str(row[1]) for row in block.rows if len(row) >= 2}
                )
    if "Stochasticity" not in filled:
        stochastic = _engine(model).dynamics_form == "stochastic_events"
        filled["Stochasticity"] = (
            "The engine draws the events at random."
            if stochastic
            else (
                f"None: the {model.engine.type} engine is deterministic, so a "
                "run depends only on the model."
            )
        )
    if "Observation" not in filled:
        filled["Observation"] = (
            "A run returns the value of every output at every position it "
            "reports; the submodels list the outputs."
        )
    return [
        Table(
            columns=["Element", "Description"],
            rows=[
                [element, filled.get(element, TO_BE_WRITTEN)]
                for element in DESIGN_CONCEPTS
            ],
        )
    ]


def _input_data(model: "Model") -> list[Block]:
    series = [
        parameter
        for parameter in model.parameters
        if isinstance(parameter.value, SeriesValue)
    ]
    if not series:
        return [Text(text="None: the model reads no series of values.")]
    return [
        Table(
            columns=["Parameter", "Mode", "Points"],
            rows=[
                [
                    parameter.id,
                    parameter.value.mode,
                    ", ".join(
                        f"({at:g}, {value:g})" for at, value in parameter.value.data
                    ),
                ]
                for parameter in series
                if isinstance(parameter.value, SeriesValue)
            ],
        )
    ]


def _engine(model: "Model") -> "EngineTypeProtocol":
    return commol_rs.engine.EngineType(model.engine.type)
