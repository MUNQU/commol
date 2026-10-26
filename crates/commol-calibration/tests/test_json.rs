//! The settings of a calibration read from the JSON of the Python API: each
//! field by its Python name, an unknown or a missing field rejected with
//! its name, and an invalid value with its path.

use commol_calibration::{
    InertiaWeightStrategy, LossConfig, MutationApplication, MutationStrategy, OptimizationConfig,
    OptimizerSettings, ProblemSettings, ReplicateSummary,
};
use commol_core::json;

/// The JSON of a problem with a particle swarm, with `swarm` merged into the
/// fields of the swarm and `problem` into the fields of the problem.
fn problem_json(swarm: &str, problem: &str) -> String {
    format!(
        r#"{{
        "observed_data": [{{"at": 1.0, "output": "B", "outputs": null, "label": "B",
            "value": 2.0, "weight": 1.0, "factor_id": null, "window": null, "trials": null}}],
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
            "radius_tol": null {swarm}}},
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
    let back: json::Value = json5::from_str(&written).unwrap();
    let original: json::Value = json5::from_str(&text).unwrap();
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
    let settings: OptimizerSettings = json::read(
        r#"{"kind": "nelder_mead", "max_iterations": 5, "sd_tolerance": 1e-6,
        "initial_simplex_size": 0.1, "alpha": null, "gamma": null, "rho": null,
        "sigma": null, "verbose": true, "header_interval": 7}"#,
    )
    .unwrap();

    assert!(settings.verbose());
    assert_eq!(settings.header_interval(), 7);
}
