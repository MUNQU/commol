"""The part of a model that depends on its kind, and what every kind gives.

A `Model` holds a name, parameters, a step unit, an engine and one
`ModelStructure`. The generic layers (simulation, calibration, ensembles,
reports, plots) use only `Model` and the methods of `ModelStructure`, so a new
kind of model is a new subclass with its engines in the engine's list
(``commol_rs.engine.EngineType``).
"""

import inspect
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar, NamedTuple, Self

from pydantic import BaseModel, ConfigDict

try:
    from commol.commol_rs import _commol_rs as commol_rs
except ImportError as e:
    raise ImportError(f"Error importing Rust extension: {e}") from e

from commol.context.constants import CalibrationParameterType

if TYPE_CHECKING:
    from commol.context.calibration import CalibrationProblem
    from commol.context.model import Model
    from commol.context.probabilistic_calibration import ParameterSetStatistics


type ReportSections = dict[str, Mapping[str, Mapping[str, object]]]
"""Sections a kind adds to a report of applied values, by key."""


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
    The part of a model that depends on its kind.

    Each kind subclasses it and sets ``kind`` to a ``Literal`` default, its
    name in the engine's list of kinds. A subclass that does not implement an
    abstract method cannot be made: the error names the method.

    Attributes
    ----------
    kind : str
        Name of the model kind, the tag of the structure in JSON.
    """

    model_config = ConfigDict(extra="forbid")

    kind: str

    _kinds: ClassVar[dict[str, type["ModelStructure"]]] = {}

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        """Register a concrete subclass under its ``kind``."""
        super().__pydantic_init_subclass__(**kwargs)
        kind = cls.model_fields["kind"].default
        if inspect.isabstract(cls) or not isinstance(kind, str):
            return
        known = ModelStructure._kinds.get(kind)
        if known is not None and known.__qualname__ != cls.__qualname__:
            raise TypeError(
                f"Model kind '{kind}' is already the kind of {known.__qualname__}"
            )
        ModelStructure._kinds[kind] = cls

    @classmethod
    def of_kind(cls, kind: str) -> type["ModelStructure"]:
        """
        The structure class of a model kind.

        Parameters
        ----------
        kind : str
            Name of the kind, such as ``"compartmental"``.

        Returns
        -------
        type[ModelStructure]
            The subclass registered for it.

        Raises
        ------
        ValueError
            If no structure class has that kind.
        """
        structure = cls._kinds.get(kind)
        if structure is None:
            raise ValueError(
                f"Unknown model kind '{kind}'. Known kinds: {sorted(cls._kinds)}"
            )
        return structure

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
    def outputs_by_source(self) -> dict[str, list[str]]:
        """
        The output names of the model, grouped by the id they come from.

        Reports and plots name a group of outputs by this id and sum them.

        Returns
        -------
        dict[str, list[str]]
            Each id mapped to its output names, in engine order.
        """

    def series_label(self, series_id: str) -> SeriesLabel:
        """
        How a plot names a series and its unit.

        Parameters
        ----------
        series_id : str
            An output name or an id of `outputs_by_source`.

        Returns
        -------
        SeriesLabel
            The id itself and no unit, unless the kind knows better.
        """
        return SeriesLabel(series_id, None)

    def calibration_value_types(self) -> frozenset[CalibrationParameterType]:
        """
        Types of value of this kind that calibration may set.

        ``parameter`` and ``factor`` are not here: every kind accepts them.
        The engine's list of kinds declares the others.

        Returns
        -------
        frozenset[CalibrationParameterType]
            For example ``{"initial_condition"}`` for a compartmental model.
        """
        return frozenset(
            CalibrationParameterType(name)
            for name in commol_rs.engine.ModelKind(self.kind).value_types()
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
        parameter_statistics: "Mapping[str, ParameterSetStatistics] | None",  # noqa: ARG002
    ) -> ReportSections:
        """
        Sections of this kind in a report of applied values.

        Parameters
        ----------
        model : Model
            The model that holds the applied values.
        calibrated_ids : set[str]
            Ids that calibration set.
        parameter_statistics : Mapping[str, ParameterSetStatistics] | None
            Ensemble spread per id, or None without an ensemble.

        Returns
        -------
        ReportSections
            Each section by its key in the report; none by default.
        """
        return {}

    def print_equations(
        self, model: "Model[Self]", output_file: str | None, format: str
    ) -> None:
        """
        Print the equations of the model.

        Raises
        ------
        NotImplementedError
            If the kind has no equations to print.
        """
        raise NotImplementedError(f"A {self.kind} model has no equations to print")

    def check_unit_consistency(self, model: "Model[Self]", verbose: bool) -> None:
        """
        Check the units of the model.

        Raises
        ------
        NotImplementedError
            If the kind has no unit check.
        """
        raise NotImplementedError(f"A {self.kind} model has no unit check")
