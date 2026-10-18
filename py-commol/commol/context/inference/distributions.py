"""The distributions of commol and the priors of a sampling method.

A distribution is truncated to the bounds of the parameter it belongs to, so
each method takes those bounds: ``Uniform``, ``LogUniform`` and ``Beta`` live
on them, the others are truncated to them (and ``LogNormal`` and ``Gamma``
also to ``(0, ∞)``). Every value is computed in Rust, and a draw inverts the
distribution function of a number of the commol generator, so a seed gives
the same draws on every platform.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from commol.commol_rs import _commol_rs as commol_rs

if TYPE_CHECKING:
    from commol.commol_rs._commol_rs import DistributionProtocol

type Bounds = tuple[float, float]
"""The ``(min, max)`` bounds of a parameter."""


class DistributionBase(BaseModel):
    """
    A family of distributions, truncated to the bounds of a parameter.

    Each method takes the bounds `(min, max)` of the parameter.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rust_kind: ClassVar[str]

    def rust_parameters(self) -> list[float]:
        """The parameters of the family, in the order Rust reads them."""
        return []

    def rust_prior(self, parameter: str) -> tuple[str, str, list[float]]:
        """The prior of `parameter`, as the Rust runners read it."""
        return (parameter, self.rust_kind, self.rust_parameters())

    def _rust(self, bounds: Bounds) -> "DistributionProtocol":
        return commol_rs.inference.Distribution(
            self.rust_kind, self.rust_parameters(), bounds[0], bounds[1]
        )

    def log_density(self, x: float, bounds: Bounds) -> float:
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

    def cdf(self, x: float, bounds: Bounds) -> float:
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

    def quantile(self, p: float, bounds: Bounds) -> float:
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

    def sample(self, count: int, bounds: Bounds, seed: int) -> list[float]:
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

    rust_kind: ClassVar[str] = "uniform"
    kind: Literal["uniform"] = "uniform"


class LogUniform(DistributionBase):
    """
    The log-uniform distribution on the bounds: ``log x`` is uniform, the
    density is proportional to ``1/x``. The bounds must be finite and the
    lower one above 0.
    """

    rust_kind: ClassVar[str] = "log_uniform"
    kind: Literal["log_uniform"] = "log_uniform"


class Beta(DistributionBase):
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

    rust_kind: ClassVar[str] = "beta"
    kind: Literal["beta"] = "beta"
    a: float = Field(gt=0.0, allow_inf_nan=False, description="The first shape.")
    b: float = Field(gt=0.0, allow_inf_nan=False, description="The second shape.")

    def rust_parameters(self) -> list[float]:
        """The shapes `a` and `b`."""
        return [self.a, self.b]


class Normal(DistributionBase):
    """
    The normal distribution, truncated to the bounds.

    Attributes
    ----------
    mean : float
        The mean before truncation.
    sd : float
        The standard deviation before truncation, above 0.
    """

    rust_kind: ClassVar[str] = "normal"
    kind: Literal["normal"] = "normal"
    mean: float = Field(allow_inf_nan=False, description="The mean.")
    sd: float = Field(
        gt=0.0, allow_inf_nan=False, description="The standard deviation."
    )

    def rust_parameters(self) -> list[float]:
        """The mean and the standard deviation."""
        return [self.mean, self.sd]


class LogNormal(DistributionBase):
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

    rust_kind: ClassVar[str] = "log_normal"
    kind: Literal["log_normal"] = "log_normal"
    mu: float = Field(allow_inf_nan=False, description="The mean of log x.")
    sigma: float = Field(
        gt=0.0, allow_inf_nan=False, description="The standard deviation of log x."
    )

    def rust_parameters(self) -> list[float]:
        """`mu` and `sigma`."""
        return [self.mu, self.sigma]


class Gamma(DistributionBase):
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

    rust_kind: ClassVar[str] = "gamma"
    kind: Literal["gamma"] = "gamma"
    shape: float = Field(gt=0.0, allow_inf_nan=False, description="The shape.")
    rate: float = Field(gt=0.0, allow_inf_nan=False, description="The rate.")

    def rust_parameters(self) -> list[float]:
        """The shape and the rate."""
        return [self.shape, self.rate]


type Distribution = Annotated[
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
    distributions : dict[str, Distribution] | None
        The prior of each parameter; None gives ``Uniform()`` to every
        parameter (`uniform_on_bounds`).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    distributions: dict[str, Distribution] | None = Field(
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

    def rust_priors(self) -> list[tuple[str, str, list[float]]] | None:
        """The priors as the Rust runners read them; None is uniform."""
        if self.distributions is None:
            return None
        return [
            distribution.rust_prior(name)
            for name, distribution in self.distributions.items()
        ]


def coerce_priors(data: object) -> object:
    """
    Read a mapping of parameter to distribution as `Priors`.

    Parameters
    ----------
    data : object
        A `Priors`, or a mapping of parameter to distribution.

    Returns
    -------
    object
        The value for the ``priors`` field.
    """
    if isinstance(data, Mapping) and set(data) != {"distributions"}:
        return {"distributions": dict(data)}
    return data


class WithPriors(BaseModel):
    """A configuration whose ``priors`` field is required."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    priors: Priors = Field(description="The prior of each parameter.")

    @model_validator(mode="before")
    @classmethod
    def require_priors(cls, data: object) -> object:
        """Name `Priors.uniform_on_bounds()` when the priors are missing."""
        if isinstance(data, Mapping):
            if data.get("priors") is None:
                raise ValueError(
                    "priors is required: give a prior per calibrated parameter, or "
                    "Priors.uniform_on_bounds()."
                )
            return {**data, "priors": coerce_priors(data["priors"])}
        return data
