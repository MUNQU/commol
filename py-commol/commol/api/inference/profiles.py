"""The runners of the profile likelihood, registered for
``Calibrator.uncertainty``.

Each runner fits the problem with ``Calibrator.run`` and hands the fit and
the threshold to the Rust tracer, which traces every profile in parallel.
"""

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from commol.api.inference.registry import register
from commol.api.inference.resampling import fit_values, rust_block_length
from commol.commol_rs import _commol_rs as commol_rs
from commol.context._model import engine_state
from commol.context.calibration import CalibrationResult
from commol.context.inference.profiles import (
    PredictionProfile,
    Profile,
    ProfileBootstrap,
    ProfileInterval,
    ProfileLikelihood,
    ProfileResult,
    Threshold,
)
from commol.context.inference.resampling import (
    BayesianBootstrap,
    ParametricBootstrap,
    ResidualBootstrap,
)

if TYPE_CHECKING:
    from commol.api.calibrator import Calibrator
    from commol.commol_rs._commol_rs import (
        BootstrapSettingsProtocol,
        ObservedDataPointProtocol,
        ProfileRunProtocol,
    )

logger = logging.getLogger(__name__)


def _bootstrap_settings(bootstrap: ProfileBootstrap) -> "BootstrapSettingsProtocol":
    settings = commol_rs.inference.BootstrapSettings
    if isinstance(bootstrap, ParametricBootstrap):
        return settings("parametric", bootstrap.replicates, bootstrap.starts)
    length = rust_block_length(bootstrap.block_length)
    if isinstance(bootstrap, BayesianBootstrap):
        return settings("bayesian", bootstrap.replicates, bootstrap.starts, length)
    if isinstance(bootstrap, ResidualBootstrap):
        return settings(
            "residual",
            bootstrap.replicates,
            bootstrap.starts,
            length,
            residual_scale=bootstrap.residual_scale,
            scheme=bootstrap.scheme,
        )
    return settings(
        "wild",
        bootstrap.replicates,
        bootstrap.starts,
        length,
        multipliers=bootstrap.multipliers,
    )


def _trace(
    calibrator: "Calibrator",
    fit: CalibrationResult,
    parameters: list[int],
    predictions: list["ObservedDataPointProtocol"],
    threshold: Threshold,
    max_steps: int,
) -> "ProfileRunProtocol":
    kind = "bootstrap" if threshold.kind == "bootstrap_calibrated" else threshold.kind
    return commol_rs.inference.trace_profiles(
        calibrator.rust_problem,
        calibrator.rust_optimizer,
        fit_values(calibrator, fit),
        parameters,
        predictions,
        kind,
        threshold.level,
        max_steps,
        calibrator.seed,
        bootstrap=(
            None
            if threshold.bootstrap is None
            else _bootstrap_settings(threshold.bootstrap)
        ),
    )


def _result(
    calibrator: "Calibrator",
    fit: CalibrationResult,
    quantity: str,
    names: Sequence[str],
    threshold: Threshold,
    run: "ProfileRunProtocol",
) -> ProfileResult:
    parameter_names = calibrator.parameter_names
    profiles: dict[str, Profile] = {}
    intervals: dict[str, ProfileInterval] = {}
    verdicts: dict[str, str | None] = {}
    notes: list[str] = []
    for name, profile in zip(names, run.profiles, strict=True):
        verdicts[name] = profile.verdict
        if profile.verdict is None:
            notes.append(
                f"'{name}': the threshold {profile.threshold:.3g} is below the "
                f"numerical tolerance {run.tolerance:.3g}, so no rise can be told "
                "from numerical error; no verdict."
            )
        exits = [
            f"at {value:.4g} {side} the fit, where '{parameter_names[index]}' "
            "reaches its bound"
            for side, exit in zip(("below", "above"), profile.box_exits, strict=True)
            if exit is not None
            for value, index in [exit]
        ]
        if profile.verdict == "structurally_non_identifiable" and exits:
            notes.append(
                f"'{name}': the ridge leaves the box {' and '.join(exits)}; the "
                "profile is flat inside the box, and its rise past it comes from "
                "the bounds, not the data."
            )
        if profile.lowest < -run.tolerance:
            notes.append(
                f"'{name}': a refit is {-profile.lowest:.3g} below the fit; the fit "
                "is not the optimum, so fit again with more starts."
            )
        profiles[name] = Profile.model_validate(
            {
                "center": profile.center,
                "threshold": profile.threshold,
                "values": profile.values,
                "statistics": profile.statistics,
                "parameters": [
                    dict(zip(parameter_names, row, strict=True))
                    for row in profile.parameters
                ],
                "multipliers": profile.multipliers,
                "converged": profile.converged,
                "ends": profile.ends,
                "evaluations": profile.evaluations,
                "replicate_rises": profile.rises,
                "moving": [parameter_names[index] for index in profile.moving],
            }
        )
        lower, upper = profile.interval
        intervals[name] = ProfileInterval(lower=lower, upper=upper)
    failed = [f"Replicate {index} failed: {reason}" for index, reason in run.failed]
    for reason in failed:
        logger.warning(reason)
    return ProfileResult.model_validate(
        {
            "fit": fit,
            "quantity": quantity,
            "threshold": threshold,
            "profiles": profiles,
            "intervals": intervals,
            "verdicts": verdicts,
            "tolerance": run.tolerance,
            "notes": notes,
            "failed_replicates": failed,
        }
    )


def run_profile_likelihood(
    calibrator: "Calibrator", method: ProfileLikelihood
) -> ProfileResult:
    """
    Run the profile likelihood of parameters.

    Parameters
    ----------
    calibrator : Calibrator
        The calibration problem.
    method : ProfileLikelihood
        The settings.

    Returns
    -------
    ProfileResult
        The fit and the profile, interval and verdict of each parameter.

    Raises
    ------
    ValueError
        If a name is not a parameter of the problem, or the threshold does not
        fit the problem.
    """
    names = calibrator.parameter_names
    for name in method.parameters:
        if name not in names:
            raise ValueError(
                f"'{name}' is not a parameter of the problem; its parameters are "
                f"{names}."
            )
    fit = calibrator.run(starts=method.starts)
    run = _trace(
        calibrator,
        fit,
        [names.index(name) for name in method.parameters],
        [],
        method.threshold,
        method.max_steps,
    )
    return _result(
        calibrator, fit, "parameter", method.parameters, method.threshold, run
    )


def run_prediction_profile(
    calibrator: "Calibrator", method: PredictionProfile
) -> ProfileResult:
    """
    Run the profile likelihood of predictions.

    Parameters
    ----------
    calibrator : Calibrator
        The calibration problem.
    method : PredictionProfile
        The settings.

    Returns
    -------
    ProfileResult
        The fit and the profile and interval of each prediction, by label.

    Raises
    ------
    ValueError
        If a prediction does not fit the model, or the threshold does not fit
        the problem.
    """
    observations = [spec.as_observation() for spec in method.specs]
    fit = calibrator.run(starts=method.starts)
    run = _trace(
        calibrator,
        fit,
        [],
        engine_state.rust_observed_data(observations),
        method.threshold,
        method.max_steps,
    )
    return _result(
        calibrator,
        fit,
        "prediction",
        [observation.label for observation in observations],
        method.threshold,
        run,
    )


register(ProfileLikelihood, run_profile_likelihood)
register(PredictionProfile, run_prediction_profile)
