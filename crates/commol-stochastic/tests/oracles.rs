//! The stochastic engines against their closed forms and against the
//! deterministic engines, their mean-field limits.

use commol_core::{Model, SimulationEngine, SimulationError, run_replicates};
use commol_difference::DifferenceEquations;
use commol_differential::DifferentialEquations;
use commol_random::replicate_seed;
use commol_stochastic::{StochasticEvents, StochasticSteps};

/// A model of the given bins (id and initial amount), parameters and
/// transitions, run by `engine` with `settings` (a JSON object or `null`).
fn model(
    engine: &str,
    settings: &str,
    bins: &[(&str, f64)],
    parameters: &[(&str, f64)],
    transitions: &[String],
) -> Model {
    let total: f64 = bins.iter().map(|(_, amount)| amount).sum();
    let bin_json: Vec<String> = bins
        .iter()
        .map(|(id, _)| format!(r#"{{"id":"{id}","name":"{id}"}}"#))
        .collect();
    let fractions: Vec<String> = bins
        .iter()
        .map(|(id, amount)| format!(r#""{id}":{}"#, amount / total))
        .collect();
    let parameter_json: Vec<String> = parameters
        .iter()
        .map(|(id, value)| format!(r#"{{"id":"{id}","value":{value}}}"#))
        .collect();
    let json = format!(
        r#"{{"name":"m","parameters":[{}],
        "engine":{{"type":"{engine}","settings":{settings}}},
        "process":{{"bins":[{}],"accumulators":[],"stratifications":[],
        "transitions":[{}]}},
        "structure":{{"kind":"aggregate","total":{total},"bin_fractions":{{{}}}}}}}"#,
        parameter_json.join(","),
        bin_json.join(","),
        transitions.join(","),
        fractions.join(","),
    );
    serde_json::from_str(&json).expect("valid model JSON")
}

fn transition(id: &str, source: &[&str], target: &[&str], rate: &str) -> String {
    transition_with(id, source, target, rate, "")
}

fn transition_with(id: &str, source: &[&str], target: &[&str], rate: &str, extra: &str) -> String {
    let quote = |ids: &[&str]| {
        ids.iter()
            .map(|id| format!(r#""{id}""#))
            .collect::<Vec<_>>()
            .join(",")
    };
    format!(
        r#"{{"id":"{id}","source":[{}],"target":[{}],"rate":"{rate}"{extra}}}"#,
        quote(source),
        quote(target)
    )
}

fn events(model: &Model) -> StochasticEvents {
    StochasticEvents::try_from_model(model).expect("the model compiles")
}

fn steps(model: &Model) -> StochasticSteps {
    StochasticSteps::try_from_model(model).expect("the model compiles")
}

/// The output rows of `count` replicates of `engine` at `positions`.
fn replicates<E: SimulationEngine + Send + Sync>(
    engine: &E,
    positions: &[f64],
    count: u64,
) -> Vec<Vec<Vec<f64>>> {
    let seeds: Vec<u64> = (0..count).map(|index| replicate_seed(11, index)).collect();
    run_replicates(engine, positions, &seeds).expect("every replicate runs")
}

/// Mean and standard error of the mean of `values`.
fn mean_and_error(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, (variance / n).sqrt())
}

/// The largest distance between the empirical distributions of two samples
/// (the two-sample Kolmogorov–Smirnov statistic).
fn ks_distance(first: &[f64], second: &[f64]) -> f64 {
    let mut values: Vec<f64> = first.iter().chain(second).copied().collect();
    values.sort_by(f64::total_cmp);
    values.dedup();
    let share = |sample: &[f64], at: f64| {
        sample.iter().filter(|&&value| value <= at).count() as f64 / sample.len() as f64
    };
    values
        .iter()
        .map(|&at| (share(first, at) - share(second, at)).abs())
        .fold(0.0, f64::max)
}

fn death(engine: &str, settings: &str, units: f64, rate: &str) -> Model {
    model(
        engine,
        settings,
        &[("A", units), ("D", 0.0)],
        &[("k", 0.5)],
        &[transition("death", &["A"], &["D"], rate)],
    )
}

// ----------------------------------------------------------------------
// Continuous time
// ----------------------------------------------------------------------

#[test]
fn a_pure_death_goes_extinct_after_the_sum_of_its_mean_waits() {
    let units = 20;
    let k = 0.5;
    let model = death("StochasticEvents", "null", f64::from(units), "k");
    let grid = 0.05;
    let positions: Vec<f64> = (0..=1200).map(|i| f64::from(i) * grid).collect();
    let runs = replicates(&events(&model), &positions, 2000);
    let times: Vec<f64> = runs
        .iter()
        .map(|rows| {
            let first = rows.iter().position(|row| row[0] == 0.0).expect("extinct");
            positions[first] - 0.5 * grid
        })
        .collect();
    let expected: f64 = (1..=units).map(|i| 1.0 / (k * f64::from(i))).sum();
    let (mean, error) = mean_and_error(&times);
    assert!(
        (mean - expected).abs() < 4.0 * error + grid,
        "{mean} against {expected} ± {error}"
    );
}

#[test]
fn a_rate_that_reads_t_keeps_each_unit_with_its_integrated_hazard() {
    // Five units at the hazard k (1 + sin t), so the events are far apart
    // and a rate frozen at the last event would be far from right.
    let model = model(
        "StochasticEvents",
        "null",
        &[("A", 5.0), ("D", 0.0)],
        &[("k", 0.2)],
        &[transition("death", &["A"], &["D"], "k * (1 + sin(t))")],
    );
    let end: f64 = 3.0;
    let runs = replicates(&events(&model), &[end], 20_000);
    let survivors: f64 = runs.iter().map(|rows| rows[0][0]).sum();
    let share = survivors / (5.0 * runs.len() as f64);
    let expected = (-0.2 * (end + 1.0 - end.cos())).exp();
    let error = (expected * (1.0 - expected) / (5.0 * runs.len() as f64)).sqrt();
    assert!(
        (share - expected).abs() < 4.0 * error,
        "{share} against {expected} ± {error}"
    );
}

#[test]
fn a_birth_death_process_goes_extinct_with_the_probability_of_kendall() {
    let (birth, death, start) = (1.0, 0.5, 2);
    let model = model(
        "StochasticEvents",
        "null",
        &[("A", f64::from(start))],
        &[("b", birth), ("d", death)],
        &[
            transition("birth", &["A"], &["A", "A"], "b"),
            transition("death", &["A"], &[], "d"),
        ],
    );
    let runs = replicates(&events(&model), &[12.0], 4000);
    let extinct = runs.iter().filter(|rows| rows[0][0] == 0.0).count() as f64 / 4000.0;
    let expected = (death / birth).powi(start);
    let error = (expected * (1.0 - expected) / 4000.0).sqrt();
    assert!(
        (extinct - expected).abs() < 4.0 * error,
        "{extinct} against {expected} ± {error}"
    );
}

/// A chain with an inflow, two proportional flows and an outflow: linear
/// rates, so the mean of the process is the solution of the ODE.
fn chain(engine: &str, settings: &str) -> Model {
    model(
        engine,
        settings,
        &[("A", 800.0), ("B", 200.0), ("C", 0.0)],
        &[("k1", 0.3), ("k2", 0.1), ("r", 5.0)],
        &[
            transition("in", &[], &["A"], "r"),
            transition("ab", &["A"], &["B"], "k1"),
            transition("bc", &["B"], &["C"], "k2"),
            transition("out", &["C"], &[], "k2"),
        ],
    )
}

#[test]
fn the_mean_of_many_runs_is_the_solution_of_the_ode() {
    let positions = [1.0, 5.0, 10.0];
    let runs = replicates(&events(&chain("StochasticEvents", "null")), &positions, 400);
    let settings = r#"{"method":"tsit45","rtol":1e-10,"atol":1e-8}"#;
    let ode = DifferentialEquations::try_from_model(&chain("DifferentialEquations", settings))
        .unwrap()
        .run_at(&positions)
        .unwrap();
    for (row, expected) in ode.iter().enumerate() {
        for (compartment, &value) in expected.iter().enumerate() {
            let sample: Vec<f64> = runs.iter().map(|rows| rows[row][compartment]).collect();
            let (mean, error) = mean_and_error(&sample);
            assert!(
                (mean - value).abs() < 4.5 * error.max(1e-9),
                "compartment {compartment} at {}: {mean} against {value} ± {error}",
                positions[row]
            );
        }
    }
}

#[test]
fn tau_leaping_agrees_with_the_direct_method_in_distribution() {
    let positions = [4.0];
    let direct = replicates(
        &events(&chain("StochasticEvents", "null")),
        &positions,
        2000,
    );
    for settings in [
        r#"{"method":"tau_leaping"}"#,
        r#"{"method":"tau_leaping","tau":0.01}"#,
    ] {
        let leaping = replicates(
            &events(&chain("StochasticEvents", settings)),
            &positions,
            2000,
        );
        for compartment in 0..3 {
            let sample = |runs: &[Vec<Vec<f64>>]| -> Vec<f64> {
                runs.iter().map(|rows| rows[0][compartment]).collect()
            };
            let distance = ks_distance(&sample(&direct), &sample(&leaping));
            // The critical value of the two-sample test at a level of 0.001.
            let critical = 1.95 * (2.0_f64 / 2000.0).sqrt();
            assert!(
                distance < critical,
                "{settings}, compartment {compartment}: {distance}"
            );
        }
    }
}

#[test]
fn a_run_depends_on_its_seed_and_not_on_where_it_reports() {
    for settings in ["null", r#"{"method":"tau_leaping"}"#] {
        let model = chain("StochasticEvents", settings);
        let mut engine = events(&model);
        engine.set_seed(5);
        engine.reset();
        let at_once = engine.run_at(&[10.0]).unwrap();
        engine.reset();
        let grid: Vec<f64> = (0..=40).map(|i| f64::from(i) * 0.25).collect();
        let on_grid = engine.run_at(&grid).unwrap();
        assert_eq!(on_grid.last(), at_once.last(), "{settings}");
        engine.reset();
        let first = engine.run_at(&[3.25]).unwrap();
        let rest = engine.run_at(&[10.0]).unwrap();
        assert_eq!(first[0], on_grid[13].clone(), "{settings}");
        assert_eq!(rest.last(), at_once.last(), "{settings}");
        engine.set_seed(6);
        engine.reset();
        assert_ne!(engine.run_at(&[10.0]).unwrap(), at_once, "{settings}");
    }
}

#[test]
fn impulses_move_whole_units_at_their_positions() {
    let model = model(
        "StochasticEvents",
        "null",
        &[("A", 1000.0), ("B", 0.0), ("C", 0.0)],
        &[("p", 0.25), ("n", 30.0)],
        &[
            transition_with("share", &["A"], &["B"], "p", r#","impulse_at":[1.5]"#),
            // A rate that reads a compartment is absolute.
            transition_with(
                "fixed",
                &["A"],
                &["C"],
                "n + 0 * A",
                r#","impulse_at":[1.5]"#,
            ),
        ],
    );
    let runs = replicates(&events(&model), &[1.0, 1.5, 2.0], 400);
    let moved: Vec<f64> = runs.iter().map(|rows| rows[1][1]).collect();
    for rows in &runs {
        assert_eq!(rows[0], vec![1000.0, 0.0, 0.0]);
        assert_eq!(rows[1][2], 30.0);
        assert_eq!(rows[1].iter().sum::<f64>(), 1000.0);
        assert_eq!(rows[1], rows[2]);
        assert_eq!(rows[1][1].fract(), 0.0);
    }
    let (mean, error) = mean_and_error(&moved);
    assert!((mean - 250.0).abs() < 4.0 * error, "{mean} ± {error}");
}

#[test]
fn an_impulse_that_is_not_a_probability_or_a_whole_number_fails() {
    for (rate, absolute) in [("1.5", false), ("2.5", true)] {
        let mut transitions = vec![transition_with(
            "jump",
            &["A"],
            &["B"],
            rate,
            r#","impulse_at":[1.0]"#,
        )];
        if absolute {
            // A rate that reads a compartment is absolute.
            transitions[0] = transition_with(
                "jump",
                &["A"],
                &["B"],
                &format!("{rate} + 0 * A"),
                r#","impulse_at":[1.0]"#,
            );
        }
        let model = model(
            "StochasticEvents",
            "null",
            &[("A", 10.0), ("B", 0.0)],
            &[],
            &transitions,
        );
        let error = events(&model).run_at(&[2.0]).unwrap_err();
        let SimulationError::Infeasible { message, .. } = error else {
            panic!("expected an infeasible run");
        };
        assert!(message.contains("'jump'"), "{message}");
    }
}

// ----------------------------------------------------------------------
// Discrete time
// ----------------------------------------------------------------------

#[test]
fn the_mean_of_many_chain_binomial_runs_is_the_difference_engine() {
    let positions = [1.0, 5.0, 10.0];
    let runs = replicates(&steps(&chain("StochasticSteps", "null")), &positions, 400);
    let expected = DifferenceEquations::try_from_model(&chain("DifferenceEquations", "null"))
        .unwrap()
        .run_at(&positions)
        .unwrap();
    for (row, expected) in expected.iter().enumerate() {
        for (compartment, &value) in expected.iter().enumerate() {
            let sample: Vec<f64> = runs.iter().map(|rows| rows[row][compartment]).collect();
            let (mean, error) = mean_and_error(&sample);
            assert!(
                (mean - value).abs() < 4.5 * error.max(1e-9),
                "compartment {compartment} at {}: {mean} against {value} ± {error}",
                positions[row]
            );
        }
    }
}

#[test]
fn one_step_of_an_outflow_has_the_binomial_variance() {
    let (units, k) = (1000.0, 0.3);
    let model = death("StochasticSteps", "null", units, "0.3");
    let runs = replicates(&steps(&model), &[1.0], 4000);
    let left: Vec<f64> = runs.iter().map(|rows| units - rows[0][0]).collect();
    let (mean, _) = mean_and_error(&left);
    let variance = left.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 3999.0;
    let expected = units * k * (1.0 - k);
    // The variance of a sample variance is about 2 σ⁴ / n.
    let error = expected * (2.0_f64 / 4000.0).sqrt();
    assert!(
        (variance - expected).abs() < 4.0 * error,
        "{variance} against {expected} ± {error}"
    );
    assert!((mean - units * k).abs() < 4.0 * (expected / 4000.0).sqrt());
}

#[test]
fn a_unit_never_leaves_twice_and_an_absolute_flow_is_capped() {
    let model = model(
        "StochasticSteps",
        "null",
        &[("A", 50.0), ("B", 0.0), ("C", 0.0)],
        &[],
        &[
            transition("ab", &["A"], &["B"], "0.6"),
            transition("ac", &["A"], &["C"], "0.4"),
            transition("drain", &["B"], &["C"], "1000 + 0 * B"),
        ],
    );
    for rows in replicates(&steps(&model), &[1.0, 2.0, 3.0], 50) {
        for row in rows {
            assert!(row.iter().all(|&amount| amount >= 0.0), "{row:?}");
            assert_eq!(row.iter().sum::<f64>(), 50.0);
        }
    }
}

#[test]
fn a_negative_rate_or_probabilities_above_one_fail_at_their_step() {
    let cases = [
        (
            vec![transition("ab", &["A"], &["B"], "if(step >= 3, -0.1, 0.1)")],
            3.0,
            vec!["'ab'", "'A'", "negative"],
        ),
        (
            vec![
                transition("ab", &["A"], &["B"], "if(step >= 2, 0.6, 0.1)"),
                transition("ac", &["A"], &["C"], "if(step >= 2, 0.6, 0.1)"),
            ],
            2.0,
            vec!["'ab' 0.6", "'ac' 0.6", "1.2", "'A'"],
        ),
    ];
    for (transitions, at, names) in cases {
        let model = model(
            "StochasticSteps",
            "null",
            &[("A", 100.0), ("B", 0.0), ("C", 0.0)],
            &[],
            &transitions,
        );
        let error = steps(&model).run_at(&[5.0]).unwrap_err();
        let SimulationError::Infeasible { position, message } = error else {
            panic!("expected an infeasible run");
        };
        assert_eq!(position, at);
        assert!(message.contains(&format!("step {at}")), "{message}");
        for name in names {
            assert!(message.contains(name), "{name} in {message}");
        }
    }
}

// ----------------------------------------------------------------------
// Whole units
// ----------------------------------------------------------------------

/// Bins A and B over ten groups that each start with half a unit of B.
fn seeded_groups(engine: &str, settings: &str) -> Model {
    let groups: Vec<String> = (0..10).map(|g| format!(r#""g{g}""#)).collect();
    let fractions: Vec<String> = (0..10).map(|g| format!(r#""g{g}":0.1"#)).collect();
    let json = format!(
        r#"{{"name":"m","parameters":[{{"id":"k","value":0.1}}],
        "engine":{{"type":"{engine}","settings":{settings}}},
        "process":{{"bins":[{{"id":"A","name":"A"}},{{"id":"B","name":"B"}}],
        "accumulators":[],
        "stratifications":[{{"id":"group","categories":[{}]}}],
        "transitions":[{}]}},
        "structure":{{"kind":"aggregate","total":1000,
        "bin_fractions":{{"A":0.995,"B":0.005}},
        "stratification_fractions":{{"group":{{{}}}}}}}}}"#,
        groups.join(","),
        transition("ab", &["A"], &["B"], "k * B / total"),
        fractions.join(","),
    );
    serde_json::from_str(&json).expect("valid model JSON")
}

#[test]
fn amounts_that_are_not_whole_need_a_rounding() {
    for engine in ["StochasticEvents", "StochasticSteps"] {
        let model = seeded_groups(engine, "null");
        let error = if engine == "StochasticEvents" {
            StochasticEvents::try_from_model(&model).err()
        } else {
            StochasticSteps::try_from_model(&model).err()
        }
        .unwrap();
        for part in [engine, "largest_remainder", "random", "'A_g0' (99.5)"] {
            assert!(error.contains(part), "{part} in {error}");
        }
    }
}

#[test]
fn whole_amounts_need_no_rounding_and_a_total_must_be_whole() {
    assert!(StochasticEvents::try_from_model(&chain("StochasticEvents", "null")).is_ok());
    let mut model = chain("StochasticSteps", r#"{"rounding":"random"}"#);
    model.aggregate_mut().unwrap().initial_conditions.total = 1000.5;
    let error = StochasticSteps::try_from_model(&model).err().unwrap();
    assert!(error.contains("StochasticSteps"), "{error}");
    assert!(error.contains("1000.5"), "{error}");
}

#[test]
fn the_largest_remainder_starts_every_run_from_the_same_whole_amounts() {
    let model = seeded_groups("StochasticSteps", r#"{"rounding":"largest_remainder"}"#);
    let runs = replicates(&steps(&model), &[0.0], 20);
    for rows in &runs {
        assert_eq!(rows[0], runs[0][0]);
        assert!(rows[0].iter().all(|amount| amount.fract() == 0.0));
        assert_eq!(rows[0].iter().sum::<f64>(), 1000.0);
    }
}

#[test]
fn a_random_rounding_starts_on_average_from_the_amounts_of_the_model() {
    let model = seeded_groups("StochasticEvents", r#"{"rounding":"random"}"#);
    let engine = events(&model);
    let amounts = engine.model().initial_state().to_vec();
    let runs = replicates(&engine, &[0.0], 4000);
    for rows in &runs {
        assert!(rows[0].iter().all(|amount| amount.fract() == 0.0));
        assert_eq!(rows[0].iter().sum::<f64>(), 1000.0);
    }
    for (compartment, &amount) in amounts.iter().enumerate() {
        let sample: Vec<f64> = runs.iter().map(|rows| rows[0][compartment]).collect();
        let (mean, error) = mean_and_error(&sample);
        assert!(
            (mean - amount).abs() < 4.5 * error.max(1e-3),
            "compartment {compartment}: {mean} against {amount} ± {error}"
        );
    }
}

#[test]
fn a_rate_that_switches_on_at_a_threshold_waits_for_it() {
    // No unit can leave before t = 2, and from there each leaves at hazard k.
    let model = death("StochasticEvents", "null", 5.0, "if(t >= 2, k, 0)");
    let runs = replicates(&events(&model), &[1.999, 4.0], 10_000);
    assert!(runs.iter().all(|rows| rows[0][0] == 5.0));
    let survivors: f64 = runs.iter().map(|rows| rows[1][0]).sum();
    let share = survivors / (5.0 * runs.len() as f64);
    let expected = (-0.5_f64 * 2.0).exp();
    let error = (expected * (1.0 - expected) / (5.0 * runs.len() as f64)).sqrt();
    assert!(
        (share - expected).abs() < 4.0 * error,
        "{share} against {expected} ± {error}"
    );
}

#[test]
fn calibrated_initial_conditions_go_through_the_rounding() {
    use commol_core::{ModelValue, ModelValueType};
    let values = [ModelValue {
        value_type: ModelValueType::InitialCondition,
        id: "B".to_string(),
    }];
    // 0.0123 of the total is 12.3 units of B; A takes the rest.
    let rounded = chain("StochasticSteps", r#"{"rounding":"largest_remainder"}"#);
    let mut engine = steps(&rounded);
    let plan = engine.plan_values(&values, Some("A")).unwrap();
    assert_eq!(engine.apply_values(&plan, &[0.0123]).unwrap(), 0.0);
    let start = &engine.run_at(&[0.0]).unwrap()[0];
    assert_eq!(start, &vec![988.0, 12.0, 0.0]);

    let mut whole_only = steps(&chain("StochasticSteps", "null"));
    let plan = whole_only.plan_values(&values, Some("A")).unwrap();
    let error = whole_only.apply_values(&plan, &[0.0123]).unwrap_err();
    for part in ["StochasticSteps", "largest_remainder", "random"] {
        assert!(error.contains(part), "{part} in {error}");
    }
}
