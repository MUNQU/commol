"""The distributions of commol and the priors of a sampling method.

A distribution is truncated to the bounds of the parameter it belongs to, so
each method takes those bounds: ``Uniform``, ``LogUniform`` and ``Beta`` live
on them, the others are truncated to them (and ``LogNormal`` and ``Gamma``
also to ``(0, ∞)``). Every value is computed in Rust, and a draw inverts the
distribution function of a number of the commol generator, so a seed gives
the same draws on every platform.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from pydantic_core import to_json

from commol._data.engine_json import engine_json
from commol._data.ranges import RangeChecked
from commol.commol_rs import _commol_rs as commol_rs

if TYPE_CHECKING:
    from commol.commol_rs._commol_rs import DistributionProtocol


class DistributionBase(BaseModel):
    """
    A family of distributions, truncated to the bounds of a parameter.

    Each method takes the bounds `(min, max)` of the parameter.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    def _rust(self, bounds: tuple[float, float]) -> "DistributionProtocol":
        return commol_rs.inference.Distribution(engine_json(self), bounds[0], bounds[1])

    def log_density(self, x: float, bounds: tuple[float, float]) -> float:
        """
        The log density at `x`.

        Parameters
        ----------
        x : float
            The value.
        bounds : tuple[float, float]
            The bounds of the parameter.

        Returns
        -------
        float
            ``-inf`` outside the bounds.

        Raises
        ------
        ValueError
            If the bounds do not fit the distribution.
        """
        return self._rust(bounds).log_density(x)

    def cdf(self, x: float, bounds: tuple[float, float]) -> float:
        """
        The distribution function at `x`.

        Parameters
        ----------
        x : float
            The value.
        bounds : tuple[float, float]
            The bounds of the parameter.

        Returns
        -------
        float
            0 at and below the lower bound, 1 at and above the upper one.

        Raises
        ------
        ValueError
            If the bounds do not fit the distribution.
        """
        return self._rust(bounds).cdf(x)

    def quantile(self, p: float, bounds: tuple[float, float]) -> float:
        """
        The quantile at level `p`.

        Parameters
        ----------
        p : float
            The level, between 0 and 1.
        bounds : tuple[float, float]
            The bounds of the parameter.

        Returns
        -------
        float
            The lower bound at 0, the upper one at 1.

        Raises
        ------
        ValueError
            If `p` is not between 0 and 1, or the bounds do not fit the
            distribution.
        """
        return self._rust(bounds).quantile(p)

    def sample(self, count: int, bounds: tuple[float, float], seed: int) -> list[float]:
        """
        Draw `count` values.

        Parameters
        ----------
        count : int
            The number of draws.
        bounds : tuple[float, float]
            The bounds of the parameter.
        seed : int
            The seed of the generator.

        Returns
        -------
        list[float]
            The draws, each inside the bounds.

        Raises
        ------
        ValueError
            If the bounds do not fit the distribution.
        """
        return self._rust(bounds).sample(count, seed)


class Uniform(DistributionBase):
    """The uniform distribution on the bounds; they must be finite."""

    kind: Literal["uniform"] = "uniform"


class LogUniform(DistributionBase):
    """
    The log-uniform distribution on the bounds: ``log x`` is uniform, the
    density is proportional to ``1/x``. The bounds must be finite and the
    lower one above 0.
    """

    kind: Literal["log_uniform"] = "log_uniform"


class Beta(DistributionBase, RangeChecked):
    """
    The beta distribution of shapes `a` and `b`, scaled to the bounds; they
    must be finite.

    Attributes
    ----------
    a : float
        The first shape, above 0.
    b : float
        The second shape, above 0.
    """

    kind: Literal["beta"] = "beta"
    a: float = Field(description="The first shape.")
    b: float = Field(description="The second shape.")


class Normal(DistributionBase, RangeChecked):
    """
    The normal distribution, truncated to the bounds.

    Attributes
    ----------
    mean : float
        The mean before truncation.
    sd : float
        The standard deviation before truncation, above 0.
    """

    kind: Literal["normal"] = "normal"
    mean: float = Field(description="The mean.")
    sd: float = Field(description="The standard deviation.")


class LogNormal(DistributionBase, RangeChecked):
    """
    The log-normal distribution, truncated to the bounds and to ``(0, ∞)``:
    ``log x`` is normal. The upper bound must be above 0.

    Attributes
    ----------
    mu : float
        The mean of ``log x``.
    sigma : float
        The standard deviation of ``log x``, above 0.
    """

    kind: Literal["log_normal"] = "log_normal"
    mu: float = Field(description="The mean of log x.")
    sigma: float = Field(description="The standard deviation of log x.")


class Gamma(DistributionBase, RangeChecked):
    """
    The gamma distribution of shape `shape` and rate `rate` (mean
    ``shape / rate``), truncated to the bounds and to ``(0, ∞)``. The upper
    bound must be above 0.

    Attributes
    ----------
    shape : float
        The shape, above 0.
    rate : float
        The rate, above 0.
    """

    kind: Literal["gamma"] = "gamma"
    shape: float = Field(description="The shape.")
    rate: float = Field(description="The rate.")


type _Distribution = Annotated[
    Uniform | LogUniform | Beta | Normal | LogNormal | Gamma,
    Field(discriminator="kind"),
]
"""One of the distributions of commol."""


class Priors(BaseModel):
    """
    One prior per calibrated parameter.

    A sampling method takes its priors with no default: what is known about
    each parameter before the data is a modelling choice. Give a mapping of
    parameter to distribution, or `uniform_on_bounds` for ``Uniform()`` on
    every parameter.

    Attributes
    ----------
    distributions : dict[str, distribution] | None
        The prior of each parameter; None gives ``Uniform()`` to every
        parameter (`uniform_on_bounds`).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    distributions: dict[str, _Distribution] | None = Field(
        description="The prior of each parameter, or None for uniform on the bounds."
    )

    @classmethod
    def uniform_on_bounds(cls) -> Self:
        """
        ``Uniform()`` on the bounds of every parameter.

        Returns
        -------
        Priors
            The priors; every parameter needs finite bounds.
        """
        return cls(distributions=None)


def coerce_priors(data: JsonValue) -> JsonValue:
    """
    Read a mapping of parameter to distribution as `Priors`.

    Parameters
    ----------
    data : JsonValue
        A `Priors`, or a mapping of parameter to distribution.

    Returns
    -------
    JsonValue
        The value for the ``priors`` field.
    """
    if isinstance(data, Mapping) and set(data) != {"distributions"}:
        return {"distributions": dict(data)}
    return data


def check_required(cls: type[BaseModel], data: JsonValue, name: str) -> None:
    """
    Raise the message of the engine when the field `name` is missing or None.

    The engine holds the rule and its message, which names the field and the
    fix: it checks the fields, without `name`, as the JSON of the first class
    of the bases of `cls` whose JSON it reads, so a subclass keeps the rule.

    Parameters
    ----------
    cls : type[BaseModel]
        The class the engine reads.
    data : JsonValue
        The fields as given.
    name : str
        The field the engine requires.

    Raises
    ------
    ValueError
        The message of the engine, if the field is missing or None.
    """
    if isinstance(data, Mapping) and data.get(name) is None:
        given = {key: value for key, value in data.items() if key != name}
        fields = to_json(given, inf_nan_mode="strings").decode()
        read = set(commol_rs.json.objects())
        engine_class = next(base for base in cls.__mro__ if base.__name__ in read)
        commol_rs.json.check(engine_class.__name__, fields)


class WithPriors(BaseModel):
    """
    A configuration whose ``priors`` field is required.

    The engine refuses a configuration with no priors (``settings/mod.rs``,
    ``PriorsSettings``), so Python holds no copy of the rule.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    priors: Priors = Field(description="The prior of each parameter.")

    @model_validator(mode="before")
    @classmethod
    def _coerce_priors(cls, data: JsonValue) -> JsonValue:
        """A mapping of distributions is the `Priors` of those distributions."""
        check_required(cls, data, "priors")
        if isinstance(data, Mapping):
            return {**data, "priors": coerce_priors(data["priors"])}
        return data
