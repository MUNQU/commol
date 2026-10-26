"""The runners of the sensitivity methods, registered for
``SensitivityAnalysis.run``.

Each runner sends its method whole to the engine, with the forward analysis
of the inputs (`ForwardAnalysis`); the engine builds the design, runs it
through the forward model with no observations and estimates, and the runner
builds the result from the values it returns.
"""

from typing import TYPE_CHECKING

from commol._data.engine_json import engine_json
from commol.commol_rs import _commol_rs as commol_rs
from commol.quantities.quantities import quantity_labels
from commol.sensitivity.results import (
    ChaosResult,
    MorrisResult,
    PRCCResult,
    PropagationResult,
    SobolResult,
)
from commol.uncertainty.sample import ParameterSample, SampleMethod

if TYPE_CHECKING:
    from commol.sensitivity.analysis import SensitivityAnalysis
    from commol.sensitivity.methods import (
        PRCC,
        Morris,
        PolynomialChaos,
        Propagation,
        Sobol,
    )


def _sample(
    analysis: "SensitivityAnalysis",
    name: str,
    draws: tuple[list[list[float]], list[float]],
) -> ParameterSample:
    """
    The draws a method ran, as a `ParameterSample`: the sample of the inputs
    as it is, or the Latin hypercube Rust drew, named after the method.
    """
    inputs = analysis.inputs
    if inputs.sample is not None:
        return inputs.sample
    values, log_weights = draws
    return ParameterSample(
        names=inputs.names,
        bounds=[(p.min_bound, p.max_bound) for p in inputs.parameters],
        values=values,
        log_weights=log_weights,
        method=SampleMethod(
            name=name, settings={"draws": len(values), "seed": analysis.seed}
        ),
    )


def run_propagation(
    analysis: "SensitivityAnalysis", method: "Propagation"
) -> PropagationResult:
    """
    Run `method` on the inputs of `analysis`.

    Parameters
    ----------
    analysis : SensitivityAnalysis
        The analysis.
    method : Propagation
        The settings.

    Returns
    -------
    PropagationResult
        The draws and the quantities of each run.

    Raises
    ------
    ValueError
        If independent inputs have no `draws`, inputs from a sample have
        `draws`, or a quantity does not fit the model.
    """
    labels = quantity_labels(method.quantities) if method.quantities else []
    draws, values, weights, draw_seeds = commol_rs.inference.propagation(
        analysis._rust, engine_json(method)
    )
    return PropagationResult(
        sample=_sample(analysis, "propagation", draws),
        labels=labels,
        values=values,
        weights=weights,
        draw_seeds=draw_seeds,
    )


def run_prcc(analysis: "SensitivityAnalysis", method: "PRCC") -> PRCCResult:
    """
    Run `method` on the inputs of `analysis`.

    Parameters
    ----------
    analysis : SensitivityAnalysis
        The analysis.
    method : PRCC
        The settings.

    Returns
    -------
    PRCCResult
        The coefficients and their p-values.

    Raises
    ------
    ValueError
        If the draws do not fit the inputs, a quantity does not fit the
        model or is NaN at a draw, there are too few draws for the inputs, or
        their ranks are collinear.
    """
    labels = quantity_labels(method.quantities)
    draws, coefficients, p_values, effective_size = commol_rs.inference.prcc(
        analysis._rust, engine_json(method)
    )
    axes = [pair for quantity in method.quantities for pair in quantity.axes()]
    return PRCCResult(
        sample=_sample(analysis, "prcc", draws),
        inputs=analysis.inputs.names,
        labels=labels,
        series=[series for series, _ in axes],
        positions=[position for _, position in axes],
        coefficients=coefficients,
        p_values=p_values,
        effective_size=effective_size,
    )


def run_morris(analysis: "SensitivityAnalysis", method: "Morris") -> MorrisResult:
    """
    Run `method` on the inputs of `analysis`.

    Parameters
    ----------
    analysis : SensitivityAnalysis
        The analysis.
    method : Morris
        The settings.

    Returns
    -------
    MorrisResult
        The indices of each input for each quantity.

    Raises
    ------
    ValueError
        If the inputs come from a sample, the levels are not even, or a
        quantity does not fit the model or is NaN at a point.
    """
    labels = quantity_labels(method.quantities)
    mu, mu_star, sigma, runs = commol_rs.inference.morris(
        analysis._rust, engine_json(method)
    )
    return MorrisResult(
        inputs=analysis.inputs.names,
        labels=labels,
        mu=mu,
        mu_star=mu_star,
        sigma=sigma,
        runs=runs,
    )


def run_sobol(analysis: "SensitivityAnalysis", method: "Sobol") -> SobolResult:
    """
    Run `method` on the inputs of `analysis`.

    Parameters
    ----------
    analysis : SensitivityAnalysis
        The analysis.
    method : Sobol
        The settings.

    Returns
    -------
    SobolResult
        The indices of each input for each quantity, with their intervals.

    Raises
    ------
    ValueError
        If the inputs come from a sample, or a quantity does not fit the
        model or is NaN at a point.
    """
    labels = quantity_labels(method.quantities)
    run = commol_rs.inference.sobol(analysis._rust, engine_json(method))
    return SobolResult(
        inputs=analysis.inputs.names,
        labels=labels,
        confidence_level=method.confidence_level,
        first=run.first,
        total=run.total,
        first_lower=run.first_lower,
        first_upper=run.first_upper,
        total_lower=run.total_lower,
        total_upper=run.total_upper,
        runs=run.runs,
    )


def run_chaos(
    analysis: "SensitivityAnalysis", method: "PolynomialChaos"
) -> ChaosResult:
    """
    Run `method` on the inputs of `analysis`.

    Parameters
    ----------
    analysis : SensitivityAnalysis
        The analysis.
    method : PolynomialChaos
        The settings.

    Returns
    -------
    ChaosResult
        The expansion of each quantity and its Sobol indices.

    Raises
    ------
    ValueError
        If the inputs come from a sample, there are not more draws than terms
        (checked before any run), or a quantity does not fit the model or is
        NaN at a point.
    """
    labels = quantity_labels(method.quantities)
    run = commol_rs.inference.polynomial_chaos(analysis._rust, engine_json(method))
    return ChaosResult(
        inputs=analysis.inputs.names,
        input_space=analysis.inputs,
        labels=labels,
        degree=method.degree,
        q_norm=method.q_norm,
        max_interaction=method.max_interaction,
        indices=run.indices,
        coefficients=run.coefficients,
        loo_error=run.loo_error,
        mean=run.mean,
        variance=run.variance,
        first=run.first,
        total=run.total,
        runs=run.runs,
    )
