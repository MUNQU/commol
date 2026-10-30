//! The settings of a calibration read from the JSON of the Python API: each
//! field by its Python name, an unknown or a missing field rejected with
//! its name, and an invalid value with its path.

use commol_calibration::{
    FitRecord, InertiaWeightStrategy, LossConfig, MutationApplication, MutationStrategy,
    Observation, ObservationModel, OptimizationConfig, OptimizerSettings, ProblemSettings,
    Progress, ReplicateSummary,
};
use commol_core::json;

/// The JSON of a problem with a particle swarm, with `swarm` merged into the
/// fields of the swarm and `problem` into the fields of the problem.
fn problem_json(swarm: &str, problem: &str) -> String {
    format!(
        r#"{{
        "observed_data": [{{"at": 1.0, "output": "B", "outputs": null, "label": "B",
            "value": 2.0, "weight": 1.0, "factor_id": null, "scale": 1.0, "window": null, "trials": null}}],
        "parameters": [{{"id": "k", "parameter_type": "parameter", "min_bound": 0.0,
            "max_bound": 1.0, "initial_guess": null}}],
        "constraints": [],
        "loss_function": "sse",
        "observation_models": null,
        "normalize_observations": false,
        "optimization_config": {{"kind": "particle_swarm", "particles": 4,
            "max_iterations": 10, "verbose": false, "header_interval": 100,
            "initialization": "latin_hypercube", "inertia": null, "inertia_factor": null,
            "inertia_min": null, "inertia_max": null, "acceleration": null,
            "cognitive": null, "social": null, "cognitive_initial": null,
            "cognitive_final": null, "social_initial": null, "social_final": null,
            "mutation": null, "mutation_scale": null, "mutation_probability": null,
            "mutation_application": null, "velocity_clamp_factor": null,
            "velocity_mutation_threshold": null, "improvement_window": null,
            "improvement_tol_abs": null, "improvement_tol_rel": null,
            "radius_tol": null, "polish": null {swarm}}},
        "initial_condition_remainder": null,
        "replicates": null,
        "replicate_summary": null,
        "seed": 3 {problem}
        }}"#
    )
}

fn read(swarm: &str, problem: &str) -> Result<ProblemSettings, String> {
    json::read(&problem_json(swarm, problem))
}

/// The problem with each `(field, value)` of `fields` in place of `null`.
fn read_with(fields: &[(&str, &str)]) -> Result<ProblemSettings, String> {
    let text = fields
        .iter()
        .fold(problem_json("", ""), |text, (field, value)| {
            with(
                &text,
                &format!(r#""{field}": null"#),
                &format!(r#""{field}": {value}"#),
            )
        });
    json::read(&text)
}

/// Replace the value of a field of `text`, which must be there.
fn with(text: &str, from: &str, to: &str) -> String {
    assert!(text.contains(from), "{from}");
    text.replace(from, to)
}

#[test]
fn a_problem_reads_each_field_by_its_python_name() {
    let settings = read("", "").unwrap();

    assert_eq!(settings.loss_function, Some(LossConfig::SumSquaredError));
    assert_eq!(settings.parameters[0].id, "k");
    assert_eq!(settings.seed, Some(3));
    let OptimizationConfig::ParticleSwarm(config) = settings.optimization_config.config().unwrap()
    else {
        panic!("the optimizer is a particle swarm");
    };
    assert_eq!(config.particles, 4);
    assert!(config.inertia_strategy.is_none());
    assert_eq!(config.mutation_strategy, MutationStrategy::None);
}

#[test]
fn what_a_problem_reads_writes_back() {
    let text = with(
        &problem_json("", ""),
        r#""inertia": null"#,
        r#""inertia": "constant""#,
    );
    let settings: ProblemSettings = json::read(&text).unwrap();

    let written = json::write(&settings);

    assert_eq!(json::read::<ProblemSettings>(&written).unwrap(), settings);
    let back: json::Value = serde_json::from_str(&written).unwrap();
    let original: json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(back, original);
}

#[test]
fn an_unknown_field_is_rejected() {
    let message = read("", r#", "loss": "sse""#).unwrap_err();

    assert!(message.contains("unknown field `loss`"), "{message}");
    let message = read(r#", "inertia_weight": 0.5"#, "").unwrap_err();

    assert!(
        message.starts_with("optimization_config.inertia_weight: unknown field"),
        "{message}"
    );
}

#[test]
fn a_missing_field_is_named() {
    let text = with(
        &problem_json("", ""),
        r#""normalize_observations": false,"#,
        "",
    );

    let message = json::read::<ProblemSettings>(&text).unwrap_err();

    assert_eq!(message, "missing field `normalize_observations`");
}

#[test]
fn an_invalid_setting_names_its_path_through_the_union() {
    let message = read_with(&[("inertia", r#""constant""#), ("inertia_factor", "-1")]).unwrap_err();

    assert!(
        message.starts_with("optimization_config.inertia_factor: inertia_factor is"),
        "{message}"
    );
    let text = with(
        &problem_json("", ""),
        r#""particles": 4"#,
        r#""particles": 0"#,
    );
    let message = json::read::<ProblemSettings>(&text).unwrap_err();
    assert!(
        message.starts_with("optimization_config.particles: particles is a whole number"),
        "{message}"
    );
}

#[test]
fn an_invalid_observation_names_its_index() {
    let text = with(&problem_json("", ""), r#""value": 2.0"#, r#""value": -2.0"#);

    let message = json::read::<ProblemSettings>(&text).unwrap_err();

    assert!(
        message.starts_with("observed_data[0].value: value is"),
        "{message}"
    );
}

#[test]
fn a_strategy_takes_its_numbers_and_their_defaults() {
    let swarm = |fields: &[(&str, &str)]| match read_with(fields)
        .unwrap()
        .optimization_config
        .config()
        .unwrap()
    {
        OptimizationConfig::ParticleSwarm(config) => config,
        OptimizationConfig::NelderMead(_) => panic!("a swarm"),
    };

    let constant = swarm(&[("inertia", r#""constant""#)]);
    assert!(matches!(
        constant.inertia_strategy,
        Some(InertiaWeightStrategy::Constant(factor))
            if factor == InertiaWeightStrategy::DEFAULT_FACTOR
    ));
    let mutated = swarm(&[
        ("mutation", r#""cauchy""#),
        ("mutation_scale", "0.2"),
        ("mutation_probability", "0.5"),
        ("mutation_application", r#""below_average""#),
    ]);
    assert_eq!(mutated.mutation_strategy, MutationStrategy::Cauchy(0.2));
    assert_eq!(mutated.mutation_probability, 0.5);
    assert_eq!(
        mutated.mutation_application,
        MutationApplication::BelowAverage
    );
}

#[test]
fn a_number_without_its_strategy_is_rejected() {
    for (fields, message) in [
        (
            &[("inertia_factor", "0.5")][..],
            "optimization_config: inertia_factor is read only with another inertia",
        ),
        (
            &[("inertia", r#""chaotic""#), ("inertia_min", "0.4")],
            "optimization_config: inertia='chaotic' needs inertia_max",
        ),
        (
            &[
                ("inertia", r#""chaotic""#),
                ("inertia_min", "0.9"),
                ("inertia_max", "0.4"),
            ],
            "optimization_config: inertia_max (0.4) must be greater than inertia_min (0.9)",
        ),
        (
            &[("mutation", r#""gaussian""#), ("mutation_scale", "0.1")],
            "optimization_config: mutation='gaussian' needs mutation_probability",
        ),
        (
            &[("improvement_window", "5")],
            "optimization_config: The improvement stop rule needs improvement_window, \
             improvement_tol_abs and improvement_tol_rel, all three",
        ),
    ] {
        assert_eq!(read_with(fields).unwrap_err(), message, "{fields:?}");
    }
}

#[test]
fn an_unknown_optimizer_is_an_error_at_its_kind() {
    let text = with(
        &problem_json("", ""),
        r#""kind": "particle_swarm""#,
        r#""kind": "annealing""#,
    );

    let message = json::read::<ProblemSettings>(&text).unwrap_err();

    assert!(
        message.starts_with("optimization_config.kind: unknown variant `annealing`"),
        "{message}"
    );
}

#[test]
fn a_quantile_summary_reads_its_level() {
    let summary = read_with(&[("replicate_summary", r#"{"level": 0.25}"#)])
        .unwrap()
        .replicate_summary;

    assert_eq!(summary, Some(ReplicateSummary::Quantile(0.25)));
    let message = read_with(&[("replicate_summary", r#""mode""#)]).unwrap_err();
    assert!(
        message.starts_with("replicate_summary: Unknown"),
        "{message}"
    );
}

#[test]
fn a_nelder_mead_optimizer_reads_its_verbosity() {
    let settings: OptimizerSettings =
        json::read(&format!(r#"{{"kind": "nelder_mead", {}}}"#, NELDER_MEAD)).unwrap();

    assert_eq!(
        settings.progress(),
        Progress {
            nelder_mead: Some(7),
            particle_swarm: None,
        }
    );
    assert!(settings.progress().any());
}

/// The fields of a verbose Nelder-Mead of 5 iterations, with a header every
/// 7 iterations.
const NELDER_MEAD: &str = r#""max_iterations": 5, "sd_tolerance": 1e-6,
    "initial_simplex_size": 0.1, "alpha": null, "gamma": null, "rho": null,
    "sigma": null, "verbose": true, "header_interval": 7"#;

#[test]
fn a_swarm_reads_its_polish_and_the_progress_of_each_stage() {
    let polish = format!(r#"{{"kind": "nelder_mead", {NELDER_MEAD}}}"#);
    let settings = read_with(&[("polish", &polish)])
        .unwrap()
        .optimization_config;

    let OptimizationConfig::ParticleSwarm(config) = settings.config().unwrap() else {
        panic!("the optimizer is a particle swarm");
    };
    assert_eq!(config.polish.unwrap().max_iterations, 5);
    assert_eq!(
        settings.progress(),
        Progress {
            nelder_mead: Some(7),
            particle_swarm: None,
        }
    );
    let quiet = read_with(&[]).unwrap().optimization_config.progress();
    assert!(!quiet.any());
}

#[test]
fn a_polish_is_a_nelder_mead_with_its_rules() {
    let swarm = r#"{"kind": "particle_swarm", "particles": 4}"#;
    let message = read_with(&[("polish", swarm)]).unwrap_err();
    assert!(
        message.starts_with("optimization_config.polish"),
        "{message}"
    );

    let polish = format!(r#"{{"kind": "nelder_mead", {NELDER_MEAD}}}"#)
        .replace(r#""sd_tolerance": 1e-6"#, r#""sd_tolerance": -1.0"#);
    let message = read_with(&[("polish", &polish)]).unwrap_err();
    assert!(
        message.starts_with("optimization_config.polish.sd_tolerance"),
        "{message}"
    );
}

#[test]
fn a_summary_of_replicates_writes_back_its_name() {
    for name in ["mean", "median", "crps"] {
        let summary = read_with(&[("replicate_summary", &format!(r#""{name}""#))])
            .unwrap()
            .replicate_summary
            .unwrap();

        assert_eq!(json::write(&summary), format!(r#""{name}""#));
    }
    let message = read_with(&[("replicate_summary", "true")]).unwrap_err();
    assert!(
        message.starts_with("replicate_summary: a replicate summary is"),
        "{message}"
    );
}

#[test]
fn a_noise_argument_is_a_number_or_a_parameter() {
    let read =
        |sd: &str| json::read::<ObservationModel>(&format!(r#"{{"kind": "normal", "sd": {sd}}}"#));

    assert_eq!(read("2").unwrap(), read("2.0").unwrap());
    let by_name = read(r#""sigma""#).unwrap();
    assert_eq!(
        json::read::<ObservationModel>(&json::write(&by_name)).unwrap(),
        by_name
    );
    let message = read("true").unwrap_err();
    assert!(
        message.contains("a noise argument is a number or the id of a noise parameter"),
        "{message}"
    );
}

/// An observation of `B` at 1 with `output`, `outputs` and `label`.
fn observation(output: &str, outputs: &str, label: &str) -> Result<(), String> {
    let text = format!(
        r#"{{"at": 1.0, "output": {output}, "outputs": {outputs}, "label": "{label}",
        "value": 2.0, "weight": 1.0, "factor_id": null, "scale": 1.0, "window": null, "trials": null}}"#
    );
    json::read::<Observation>(&text).map(|_| ())
}

#[test]
fn an_observation_has_one_output_and_a_label() {
    for ((output, outputs, label), message) in [
        (
            ("null", "[]", "B"),
            "outputs must be non-empty when provided",
        ),
        (
            (r#""B""#, r#"["B"]"#, "B"),
            "Give exactly one of 'output' and 'outputs'.",
        ),
        (
            (r#""B""#, "null", ""),
            "An observation needs a 'label' that names its series.",
        ),
    ] {
        assert_eq!(
            observation(output, outputs, label),
            Err(message.to_string()),
            "{output} {outputs} {label}"
        );
    }
    assert_eq!(observation("null", r#"["B"]"#, "B"), Ok(()));
}

#[test]
fn a_fit_gives_the_values_of_the_parameters_it_has() {
    let fit: FitRecord = json::read(
        r#"{"best_parameters": {"k": 0.5}, "final_loss": 1.0, "iterations": 3,
        "converged": true, "termination_reason": "done", "starts": [], "failed": [],
        "seed": 1}"#,
    )
    .unwrap();

    assert_eq!(fit.values(&["k".to_string()]), Ok(vec![0.5]));
    assert_eq!(
        fit.values(&["k".to_string(), "m".to_string()]),
        Err(r#"The fit is not a fit of this problem: it lacks ["m"]"#.to_string())
    );
}
