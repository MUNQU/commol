"""How the process of a model is laid out, and what every kind gives.

A `Model` holds a header, a `Process` written once, an engine and one
`ModelStructure`, which changes between kinds of model. The generic layers
(simulation, calibration, uncertainty, reports, plots) use only `Model` and the
methods of `ModelStructure`, so a new kind of model is a new subclass with its
engines in the engine's list (``commol_rs.engine.EngineType``).

A structure may have variants: a subclass that keeps the kind, the fields
and the engines of its structure, narrows the constructor and may add checks.
Its ``variant`` is a ``Literal`` default; the registry holds one class per
``(kind, variant)``, so a model loads back as the class it was saved with.
"""

import inspect
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar, NamedTuple, Self, TypedDict

from pydantic import BaseModel, ConfigDict

from commol.commol_rs import _commol_rs as commol_rs
from commol._data.engine_json import engine_json


if TYPE_CHECKING:
    from commol._model.report import CategorySplitEntry, InitialAmountEntry
    from commol.commol_rs._commol_rs import EngineProtocol
    from commol._problem.problem import CalibrationParameter, CalibrationProblem
    from commol._model.description import Section
    from commol._model.model import Model
    from commol.uncertainty.sample import ParameterSummary


class ReportSections(TypedDict, total=False):
    """
    The sections a kind adds to a report of applied values, by key: an
    aggregate model adds ``initial_amounts`` and ``category_splits``, and a
    new kind adds the keys of its sections here.
    """

    initial_amounts: "dict[str, InitialAmountEntry]"
    category_splits: "dict[str, CategorySplitEntry]"


class SeriesLabel(NamedTuple):
    """
    How a plot names a series and its unit.

    Attributes
    ----------
    name : str
        Display name of the series.
    unit : str | None
        Unit of its values, or None when it has none.
    """

    name: str
    unit: str | None


class ModelStructure(BaseModel, ABC):
    """
    How the process of a model is laid out: the part that depends on its kind.

    Each kind subclasses it and sets ``kind`` to a ``Literal`` default, its
    name in the engine's list of kinds. A variant subclasses the structure of
    its kind and sets ``variant`` to a ``Literal`` default. A subclass that
    does not implement an abstract method cannot be made: the error names
    the method.

    Attributes
    ----------
    kind : str
        Name of the model kind, the tag of the structure in JSON.
    variant : str | None
        Name of the variant, or None for the structure of the kind itself.
    """

    model_config = ConfigDict(extra="forbid")

    kind: str
    variant: str | None = None

    _kinds: ClassVar[dict[tuple[str, str | None], type["ModelStructure"]]] = {}

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        """Register a concrete subclass under its ``kind`` and ``variant``."""
        super().__pydantic_init_subclass__(**kwargs)
        kind = cls.model_fields["kind"].default
        variant = cls.model_fields["variant"].default
        if inspect.isabstract(cls) or not isinstance(kind, str):
            return
        key = (kind, variant)
        known = ModelStructure._kinds.get(key)
        if known is not None and known.__qualname__ != cls.__qualname__:
            raise TypeError(
                f"Model kind '{kind}' with variant {variant!r} is already the "
                f"structure of {known.__qualname__}"
            )
        if variant is not None and (kind, None) not in ModelStructure._kinds:
            raise TypeError(
                f"The variant {variant!r} of {cls.__qualname__} needs the "
                f"structure of its kind '{kind}' first"
            )
        ModelStructure._kinds[key] = cls

    @classmethod
    def of_kind(cls, kind: str, variant: str | None = None) -> type["ModelStructure"]:
        """
        The structure class of a model kind, or of one of its variants.

        Parameters
        ----------
        kind : str
            Name of the kind, such as ``"aggregate"``.
        variant : str | None, default=None
            Name of a variant of the kind, or None for the kind itself.

        Returns
        -------
        type[ModelStructure]
            The subclass registered for them.

        Raises
        ------
        ValueError
            If no structure class has that kind and variant.
        """
        structure = cls._kinds.get((kind, variant))
        if structure is not None:
            return structure
        kinds = sorted({known for known, _ in cls._kinds})
        if kind not in kinds:
            raise ValueError(f"Unknown model kind '{kind}'. Known kinds: {kinds}")
        variants = sorted(
            known for known_kind, known in cls._kinds if known_kind == kind and known
        )
        raise ValueError(
            f"Unknown variant {variant!r} of the {kind} kind. Known variants: "
            f"{variants}"
        )

    @abstractmethod
    def check_model(self, model: "Model[Self]") -> None:
        """
        Check the structure against the rest of its model.

        The model calls it when it is made, so a model that cannot be run as
        written is rejected then.

        Parameters
        ----------
        model : Model
            The model that holds this structure.

        Raises
        ------
        ValueError
            If the structure does not agree with the parameters or the engine.
        """

    @abstractmethod
    def outputs_by_source(self, model: "Model[Self]") -> dict[str, list[str]]:
        """
        The output names of the model, grouped by the id they come from.

        Reports and plots name a group of outputs by this id and sum them.

        Parameters
        ----------
        model : Model
            The model that holds this structure.

        Returns
        -------
        dict[str, list[str]]
            Each id mapped to its output names, in engine order.
        """

    def series_label(
        self,
        model: "Model[Self]",  # noqa: ARG002
        series_id: str,
    ) -> SeriesLabel:
        """
        How a plot names a series and its unit.

        Parameters
        ----------
        model : Model
            The model that holds this structure.
        series_id : str
            An output name or an id of `outputs_by_source`.

        Returns
        -------
        SeriesLabel
            The id itself and no unit, unless the kind knows better.
        """
        return SeriesLabel(series_id, None)

    def calibration_value_types(self) -> frozenset[str]:
        """
        Types of value of this kind that calibration may set.

        ``parameter`` and ``factor`` are not here: every kind accepts them.
        The engine's list of kinds declares the others.

        Returns
        -------
        frozenset[str]
            For example ``{"initial_condition"}`` for an aggregate model.
        """
        return frozenset(
            name
            for name in commol_rs.engine.ModelKind(engine_json(self.kind)).value_types()
        )

    def uncalibrated_values(self) -> list[str]:
        """
        Ids of the values of this kind that need calibration before a run.

        Returns
        -------
        list[str]
            Empty unless the kind has values left unset.
        """
        return []

    def check_calibration_values(
        self,
        model: "Model[Self]",  # noqa: ARG002
        engine: "EngineProtocol",  # noqa: ARG002
        parameters: Sequence["CalibrationParameter"],  # noqa: ARG002
        remainder: str | None,  # noqa: ARG002
    ) -> None:
        """
        Check the values of this kind that a calibration or a sensitivity
        analysis sets.

        The engine has resolved each value and the remainder when this runs;
        a kind checks here what only the Python definition knows, such as the
        units of the bins. The default checks nothing.

        Parameters
        ----------
        model : Model
            The model the values are set on.
        engine : EngineProtocol
            The engine built from it.
        parameters : Sequence[CalibrationParameter]
            The values set, with their types and bounds.
        remainder : str | None
            The id that takes what the initial conditions leave of the sum
            the kind keeps, if any.
        """

    def with_calibrated_values(
        self,
        model: "Model[Self]",  # noqa: ARG002
        problem: "CalibrationProblem",  # noqa: ARG002
        values: Mapping[str, float],  # noqa: ARG002
    ) -> Self:
        """
        The structure with the calibrated values of its kind written back.

        Parameters
        ----------
        model : Model
            The model the problem was built on. It is not changed.
        problem : CalibrationProblem
            The problem, for the types of its parameters.
        values : Mapping[str, float]
            Calibrated value per parameter id.

        Returns
        -------
        Self
            A structure; this one when the kind sets no values of its own.
        """
        return self

    def report_sections(
        self,
        model: "Model[Self]",  # noqa: ARG002
        calibrated_ids: set[str],  # noqa: ARG002
        parameter_statistics: "Mapping[str, ParameterSummary] | None",  # noqa: ARG002
    ) -> ReportSections:
        """
        Sections of this kind in a report of applied values.

        Parameters
        ----------
        model : Model
            The model that holds the applied values.
        calibrated_ids : set[str]
            Ids that calibration set.
        parameter_statistics : Mapping[str, ParameterSummary] | None
            Statistics of each id from a sample, or None without one.

        Returns
        -------
        ReportSections
            Each section by its key in the report; none by default.
        """
        return {}

    def describe(
        self,
        model: "Model[Self]",  # noqa: ARG002
    ) -> "list[Section]":
        """
        The sections of a description that this kind fills.

        `Model.describe` adds them to the sections every kind shares
        (overview, schedule, parameters, outputs). A kind fills the sections
        it knows, by key: ``entities`` (what holds the state), ``dynamics``
        (how it changes), ``initialization`` (the state it starts from) and
        ``design_concepts`` (a `Table` of elements of the ODD protocol and
        their text). A section with no blocks is left out of a rendering.

        Parameters
        ----------
        model : Model
            The model that holds this structure.

        Returns
        -------
        list[Section]
            The sections of the kind; none by default.
        """
        return []

    def check_unit_consistency(self, model: "Model[Self]", verbose: bool) -> None:
        """
        Check the units of the model.

        Raises
        ------
        NotImplementedError
            If the kind has no unit check.
        """
        raise NotImplementedError(f"A {self.kind} model has no unit check")
