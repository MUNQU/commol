"""The runners of the profile likelihood, registered for
``Calibrator.uncertainty``.

Each runner fits the problem with ``Calibrator.run`` and hands the fit and
the threshold to the Rust tracer, which traces every profile in parallel.
"""

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from commol.api.inference.registry import register
from commol.api.inference.resampling import (
    fit_values,
    log_failed,
    rust_block_length,
)
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
    ProfileStructure,
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
        ProfileStructureProtocol,
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
        threshold.confidence_level,
        max_steps,
        calibrator.seed,
        bootstrap=(
            None
            if threshold.bootstrap is None
            else _bootstrap_settings(threshold.bootstrap)
        ),
    )


def _structure(
    run: "ProfileStructureProtocol", parameter_names: Sequence[str]
) -> ProfileStructure:
    return ProfileStructure.model_validate(
        {
            "verdict": run.verdict,
            "largest_rise": run.largest_rise,
            "compensated_by": [parameter_names[i] for i in run.compensating],
            "joint_rise": run.joint_rise,
            "undetermined": run.undetermined,
            "on_bound": None if run.on_bound is None else parameter_names[run.on_bound],
        }
    )


def _structure_notes(
    name: str, structure: ProfileStructure, threshold: float, flat: bool
) -> list[str]:
    """What a reader should know of a profile with no structural verdict."""
    rise = structure.largest_rise
    if structure.undetermined in {"below_fit", "undecided"} and not flat:
        return []
    why = {
        "below_fit": "the profile goes below the fit by more than the "
        "evaluation floor, so the fit is not the optimum",
        "no_noise_floor": "the evaluation floor cannot be computed for this objective",
        "stochastic": "a stochastic engine runs replicates, which are not a "
        "smooth function of the parameters",
        "on_bound": f"'{structure.on_bound}' sits on its bound at the fit, so a "
        "reset cannot show what lies past it",
        "undecided": "the resets of the other parameters raise it by up to "
        + ("no feasible value" if rise is None else f"{rise:.3g}")
        + f", against the threshold {threshold:.3g}",
    }.get(structure.undetermined or "")
    if why is None:
        return []
    return [f"'{name}': the profile gives no structural verdict: {why}."]


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
    structure: dict[str, ProfileStructure] = {}
    notes: list[str] = []
    for name, profile in zip(names, run.profiles, strict=True):
        verdicts[name] = profile.verdict
        if profile.verdict is None:
            notes.append(
                f"'{name}': the threshold {profile.threshold:.3g} is at or below "
                f"the numerical tolerance {run.tolerance:.3g}, so no rise can be "
                "told from numerical error; no verdict."
            )
        exits = [
            f"at {value:.4g} {side} the fit, where '{parameter_names[index]}' "
            "reaches its bound"
            for side, exit in zip(("below", "above"), profile.box_exits, strict=True)
            if exit is not None
            for value, index in [exit]
        ]
        flat = profile.verdict == "structurally_non_identifiable"
        if profile.structure is not None:
            structure[name] = _structure(profile.structure, parameter_names)
            notes.extend(
                _structure_notes(name, structure[name], profile.threshold, flat)
            )
        if flat and exits:
            notes.append(
                f"'{name}': the ridge leaves the box {' and '.join(exits)}; the "
                "profile is flat inside the box, and its rise past it comes from "
                "the bounds, not the data."
            )
        crossings = [
            parameter_names[index] if index is not None else None
            for index in profile.box_crossings
        ]
        for side, by, end in zip(
            ("below", "above"), crossings, profile.interval, strict=True
        ):
            if by is not None:
                notes.append(
                    f"'{name}': the profile crosses the threshold {side} the fit "
                    f"at {end:.4g} where '{by}' reaches a bound, so that side counts "
                    "as open."
                )
        if profile.refit_error is not None and profile.refit_error > run.tolerance:
            notes.append(
                f"'{name}': a refit at the fit lands {profile.refit_error:.3g} from "
                f"it, above the tolerance {run.tolerance:.3g}: the refits are less "
                "accurate than the tolerance."
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
                "box_crossings": crossings,
                "refit_error": profile.refit_error,
            }
        )
        lower, upper = profile.interval
        intervals[name] = ProfileInterval(lower=lower, upper=upper)
    return ProfileResult.model_validate(
        {
            "fit": fit,
            "quantity": quantity,
            "threshold": threshold,
            "profiles": profiles,
            "intervals": intervals,
            "verdicts": verdicts,
            "tolerance": run.tolerance,
            "evaluation_floor": run.evaluation_floor,
            "structure": structure,
            "notes": notes,
            "failed": log_failed(run.failed, logger),
            "not_converged": run.not_converged,
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
