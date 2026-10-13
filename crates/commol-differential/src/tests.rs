//! Closed-form checks of the differential engine.

use std::time::Instant;

use commol_core::{Model, SimulationEngine};
use commol_difference::DifferenceEquations;

use super::DifferentialEquations;

/// A model with one compartment per bin, built from JSON.
///
/// `bins` holds `(id, initial population)`, `parameters` `(id, JSON value)`,
/// `transitions` the JSON objects of the transitions.
fn model(
    typology: &str,
    bins: &[(&str, f64)],
    parameters: &[(&str, &str)],
    accumulators: &[&str],
    transitions: &[String],
    solver: &str,
) -> Model {
    let size: f64 = bins.iter().map(|(_, population)| population).sum();
    let bin_json: Vec<String> = bins
        .iter()
        .map(|(id, _)| format!(r#"{{"id":"{id}","name":"{id}"}}"#))
        .collect();
    let fractions: Vec<String> = bins
        .iter()
        .map(|(id, population)| format!(r#"{{"bin":"{id}","fraction":{}}}"#, population / size))
        .collect();
    let parameter_json: Vec<String> = parameters
        .iter()
        .map(|(id, value)| format!(r#"{{"id":"{id}","value":{value}}}"#))
        .collect();
    let accumulator_json: Vec<String> = accumulators
        .iter()
        .map(|id| format!(r#"{{"id":"{id}","name":"{id}"}}"#))
        .collect();
    let json = format!(
        r#"{{"name":"m","population":{{"bins":[{}],"accumulators":[{}],
        "stratifications":[],"transitions":[],
        "initial_conditions":{{"population_size":{size},"bin_fractions":[{}],
        "stratification_fractions":[]}}}},"parameters":[{}],
        "dynamics":{{"typology":"{typology}","transitions":[{}]{solver}}}}}"#,
        bin_json.join(","),
        accumulator_json.join(","),
        fractions.join(","),
        parameter_json.join(","),
        transitions.join(","),
    );
    serde_json::from_str(&json).expect("valid model JSON")
}

fn transition(id: &str, source: &[&str], target: &[&str], rate: &str, extra: &str) -> String {
    let quote = |ids: &[&str]| {
        ids.iter()
            .map(|id| format!(r#""{id}""#))
            .collect::<Vec<_>>()
            .join(",")
    };
    format!(
        r#"{{"id":"{id}","source":[{}],"target":[{}],"rate":"{rate}",
        "stratified_rates":null,"per_compartment":null{extra}}}"#,
        quote(source),
        quote(target)
    )
}

const TIGHT: &str = r#","solver":{"method":"tsit45","rtol":1e-10,"atol":1e-8}"#;
const TIGHT_BDF: &str = r#","solver":{"method":"bdf","rtol":1e-10,"atol":1e-8}"#;

fn engine(model: &Model) -> DifferentialEquations {
    DifferentialEquations::try_from_model(model).expect("the model compiles")
}

fn assert_close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    let error = (actual - expected).abs();
    assert!(
        error <= tolerance * expected.abs().max(1.0),
        "{what}: got {actual}, expected {expected}, error {error}"
    );
}

fn decay_model(solver: &str) -> Model {
    model(
        "DifferentialEquations",
        &[("A", 1000.0), ("B", 0.0)],
        &[("k", "0.3")],
        &["moved"],
        &[transition(
            "decay",
            &["A"],
            &["B"],
            "k",
            r#","accumulators":["moved"]"#,
        )],
        solver,
    )
}

#[test]
fn exponential_decay_matches_the_closed_form_with_both_methods() {
    for solver in [TIGHT, TIGHT_BDF] {
        let rows = engine(&decay_model(solver)).run(10).unwrap();
        for (t, row) in rows.iter().enumerate() {
            let expected = 1000.0 * (-0.3 * t as f64).exp();
            assert_close(row[0], expected, 1e-7, &format!("A({t}) {solver}"));
            assert_close(row[0] + row[1], 1000.0, 1e-9, "conservation");
        }
    }
}

#[test]
fn an_accumulator_is_the_integral_of_its_flows() {
    let rows = engine(&decay_model(TIGHT)).run(10).unwrap();
    for (t, row) in rows.iter().enumerate() {
        let expected = 1000.0 * (1.0 - (-0.3 * t as f64).exp());
        assert_close(row[2], expected, 1e-7, &format!("moved({t})"));
    }
}

#[test]
fn outputs_at_any_positions_come_from_the_dense_output() {
    let positions = [0.0, 0.5, 2.25, 3.5];
    let rows = engine(&decay_model(TIGHT)).run_at(&positions).unwrap();
    for (position, row) in positions.iter().zip(&rows) {
        let expected = 1000.0 * (-0.3 * position).exp();
        assert_close(row[0], expected, 1e-7, &format!("A({position})"));
    }
    // A window over [0.5, 3.5] is the integral of the flow.
    let window = rows[3][2] - rows[1][2];
    let expected = 1000.0 * ((-0.3_f64 * 0.5).exp() - (-0.3_f64 * 3.5).exp());
    assert_close(window, expected, 1e-7, "window [0.5, 3.5]");
}

#[test]
fn positions_must_be_sorted_and_not_in_the_past() {
    let mut engine = engine(&decay_model(TIGHT));
    let error = engine.run_at(&[1.0, 0.5]).unwrap_err();
    assert!(error.contains("sorted"), "{error}");
    engine.run_at(&[2.0]).unwrap();
    let error = engine.run_at(&[1.0]).unwrap_err();
    assert!(error.contains("before the current time"), "{error}");
}

#[test]
fn logistic_growth_matches_the_closed_form() {
    let model = model(
        "DifferentialEquations",
        &[("X", 10.0)],
        &[("r", "0.5"), ("K", "1000.0")],
        &[],
        &[transition("growth", &[], &["X"], "r * X * (1 - X / K)", "")],
        TIGHT,
    );
    let rows = engine(&model).run(20).unwrap();
    for (t, row) in rows.iter().enumerate() {
        let expected = 1000.0 / (1.0 + 99.0 * (-0.5 * t as f64).exp());
        assert_close(row[0], expected, 1e-7, &format!("X({t})"));
    }
}

#[test]
fn a_linear_two_by_two_system_matches_the_closed_form() {
    let model = model(
        "DifferentialEquations",
        &[("A", 800.0), ("B", 200.0)],
        &[("a", "0.4"), ("b", "0.1")],
        &[],
        &[
            transition("ab", &["A"], &["B"], "a", ""),
            transition("ba", &["B"], &["A"], "b", ""),
        ],
        TIGHT,
    );
    let rows = engine(&model).run(15).unwrap();
    let equilibrium = 1000.0 * 0.1 / 0.5;
    for (t, row) in rows.iter().enumerate() {
        let expected = equilibrium + (800.0 - equilibrium) * (-0.5 * t as f64).exp();
        assert_close(row[0], expected, 1e-7, &format!("A({t})"));
    }
}

fn sir_model(solver: &str) -> Model {
    model(
        "DifferentialEquations",
        &[("S", 990.0), ("I", 10.0), ("R", 0.0)],
        &[("beta", "0.3"), ("gamma", "0.1")],
        &[],
        &[
            transition("infection", &["S"], &["I"], "beta * S * I / N", ""),
            transition("recovery", &["I"], &["R"], "gamma", ""),
        ],
        solver,
    )
}

#[test]
fn sir_keeps_the_final_size_relation() {
    // ln(S(t) / S(0)) = -(beta / gamma) R(t) / N holds at every t.
    let rows = engine(&sir_model(TIGHT)).run(300).unwrap();
    for row in rows.iter().step_by(25) {
        let residual = (row[0] / 990.0).ln() + 3.0 * row[2] / 1000.0;
        assert!(residual.abs() < 1e-7, "final-size residual {residual}");
    }
    let last = rows.last().unwrap();
    assert!(last[1] < 1e-3, "the epidemic is over: I = {}", last[1]);
}

#[test]
fn sis_reaches_its_endemic_equilibrium() {
    let model = model(
        "DifferentialEquations",
        &[("S", 990.0), ("I", 10.0)],
        &[("beta", "0.4"), ("gamma", "0.1")],
        &[],
        &[
            transition("infection", &["S"], &["I"], "beta * S * I / N", ""),
            transition("recovery", &["I"], &["S"], "gamma", ""),
        ],
        TIGHT,
    );
    let rows = engine(&model).run(400).unwrap();
    assert_close(rows[400][1], 1000.0 * (1.0 - 0.1 / 0.4), 1e-7, "I*");
}

fn stiff_model(solver: &str) -> Model {
    // Rates 1e3 and 1e-2: a stiffness ratio of 1e5.
    model(
        "DifferentialEquations",
        &[("A", 1000.0), ("B", 0.0), ("C", 0.0)],
        &[("fast", "1000.0"), ("slow", "0.01")],
        &[],
        &[
            transition("ab", &["A"], &["B"], "fast", ""),
            transition("bc", &["B"], &["C"], "slow", ""),
        ],
        solver,
    )
}

#[test]
fn bdf_solves_a_stiff_model_within_the_tolerance() {
    let model = stiff_model(r#","solver":{"method":"bdf","rtol":1e-8,"atol":1e-8}"#);
    let started = Instant::now();
    let rows = engine(&model).run(100).unwrap();
    let elapsed = started.elapsed();
    let (k1, k2) = (1000.0_f64, 0.01_f64);
    for (t, row) in rows.iter().enumerate().skip(1) {
        let t = t as f64;
        let b = 1000.0 * k1 / (k2 - k1) * ((-k1 * t).exp() - (-k2 * t).exp());
        assert_close(row[1], b, 1e-6, &format!("B({t})"));
    }
    eprintln!("stiff bdf, 100 time units: {elapsed:?}");
}

/// A per-capita rate `p` on step 2 only, as a time-series pulse.
fn pulse_model(typology: &str) -> Model {
    model(
        typology,
        &[("A", 1000.0), ("B", 0.0)],
        &[("p", r#"{"data":[[2,0.2]],"mode":"pulse"}"#)],
        &[],
        &[transition("pulse", &["A"], &["B"], "p", "")],
        if typology == "DifferentialEquations" {
            TIGHT
        } else {
            ""
        },
    )
}

#[test]
fn a_pulse_rate_is_a_constant_rate_on_its_unit_interval() {
    let rows = engine(&pulse_model("DifferentialEquations"))
        .run(4)
        .unwrap();
    assert_close(rows[2][0], 1000.0, 1e-12, "A before the pulse");
    let moved = rows[2][0] - rows[3][0];
    assert_close(moved, 1000.0 * (1.0 - (-0.2_f64).exp()), 1e-8, "ODE pulse");
    assert_close(rows[4][0], rows[3][0], 1e-12, "A after the pulse");

    // The difference engine moves `p` times the source in one step.
    let mut difference = DifferenceEquations::try_from_model(&pulse_model("DifferenceEquations"))
        .expect("the model compiles");
    let rows = difference.run(4).unwrap();
    assert_close(rows[2][0] - rows[3][0], 200.0, 1e-12, "difference pulse");
}

/// One per-capita transition A → B with `rate`.
fn rate_model(rate: &str) -> Model {
    model(
        "DifferentialEquations",
        &[("A", 1000.0), ("B", 0.0)],
        &[("k", "0.1")],
        &[],
        &[transition("move", &["A"], &["B"], rate, "")],
        TIGHT,
    )
}

#[test]
fn a_rate_reads_the_step_of_its_unit_interval() {
    // On [n, n + 1) the rate is k (n + 1), so A(n) = 1000 exp(−k n (n + 1) / 2).
    let rows = engine(&rate_model("k * (step + 1)")).run(5).unwrap();
    for (n, row) in rows.iter().enumerate() {
        let n = n as f64;
        let expected = 1000.0 * (-0.1 * n * (n + 1.0) / 2.0).exp();
        assert_close(row[0], expected, 1e-7, &format!("A({n})"));
    }
}

#[test]
fn a_rate_that_switches_on_the_step_acts_only_on_its_window() {
    let rows = engine(&rate_model("if(step >= 2, if(step < 4, 0.5, 0), 0)"))
        .run(6)
        .unwrap();
    assert_close(rows[2][0], 1000.0, 1e-12, "A before the window");
    assert_close(
        rows[4][0],
        1000.0 * (-1.0_f64).exp(),
        1e-7,
        "A after two units",
    );
    assert_close(rows[6][0], rows[4][0], 1e-12, "A after the window");
}

#[test]
fn a_formula_parameter_that_reads_the_step_restarts_the_integration() {
    let model = model(
        "DifferentialEquations",
        &[("A", 1000.0), ("B", 0.0)],
        &[("k", "0.1"), ("r", r#""k * (step + 1)""#)],
        &[],
        &[transition("move", &["A"], &["B"], "r", "")],
        TIGHT,
    );
    let rows = engine(&model).run(3).unwrap();
    assert_close(rows[3][0], 1000.0 * (-0.6_f64).exp(), 1e-7, "A(3)");
}

#[test]
fn a_rate_that_switches_on_t_acts_only_on_its_window() {
    // The rate is 0.5 on [12.5, 14) only: A keeps exp(−0.75) of its start.
    // Before the window the rate is 0, so without a restart at 12.5 the
    // solver grows its step and can step over the window.
    let rows = engine(&rate_model("if(t >= 12.5, if(t < 14, 0.5, 0), 0)"))
        .run(40)
        .unwrap();
    assert_close(rows[12][0], 1000.0, 1e-12, "A before the window");
    let after = 1000.0 * (-0.75_f64).exp();
    assert_close(rows[14][0], after, 1e-7, "A at the end of the window");
    assert_close(rows[40][0], after, 1e-7, "A after the window");
}

#[test]
fn a_time_threshold_reads_the_current_parameter_values() {
    let model = model(
        "DifferentialEquations",
        &[("A", 1000.0), ("B", 0.0)],
        &[("k", "0.1"), ("t_on", "10.0")],
        &[],
        &[transition(
            "move",
            &["A"],
            &["B"],
            "if(t >= t_on, if(t < t_on + 1.5, k, 0), 0)",
            "",
        )],
        TIGHT,
    );
    let mut engine = engine(&model);
    engine.set_parameter("t_on", 15.5).unwrap();
    let rows = engine.run(40).unwrap();
    assert_close(rows[15][0], 1000.0, 1e-12, "A before t_on");
    assert_close(rows[40][0], 1000.0 * (-0.15_f64).exp(), 1e-7, "A(40)");
}

fn impulse_model(typology: &str, extra: &str) -> Model {
    model(
        typology,
        &[("A", 1000.0), ("B", 0.0)],
        &[("p", "0.25")],
        &["jumped"],
        &[transition(
            "jump",
            &["A"],
            &["B"],
            "p",
            &format!(r#","accumulators":["jumped"]{extra}"#),
        )],
        if typology == "DifferentialEquations" {
            TIGHT
        } else {
            ""
        },
    )
}

#[test]
fn an_impulse_moves_its_fraction_at_once_in_both_engines() {
    let impulse = r#","impulse_at":[1.0, 3.0]"#;
    let ode = engine(&impulse_model("DifferentialEquations", impulse))
        .run(4)
        .unwrap();
    let mut difference =
        DifferenceEquations::try_from_model(&impulse_model("DifferenceEquations", impulse))
            .unwrap();
    let diff = difference.run(4).unwrap();
    let expected_a = [1000.0, 750.0, 750.0, 562.5, 562.5];
    for rows in [&ode, &diff] {
        for (t, &a) in expected_a.iter().enumerate() {
            assert_close(rows[t][0], a, 1e-12, &format!("A({t})"));
            assert_close(rows[t][2], 1000.0 - a, 1e-12, &format!("jumped({t})"));
        }
    }
}

#[test]
fn an_ode_impulse_may_act_between_integers() {
    let rows = engine(&impulse_model(
        "DifferentialEquations",
        r#","impulse_at":[0.5]"#,
    ))
    .run_at(&[0.25, 0.5, 0.75])
    .unwrap();
    assert_close(rows[0][0], 1000.0, 1e-12, "before");
    assert_close(rows[1][0], 750.0, 1e-12, "at the impulse: after the jump");
    assert_close(rows[2][0], 750.0, 1e-12, "after");
}

#[test]
fn a_difference_impulse_must_act_at_integers() {
    let error = DifferenceEquations::try_from_model(&impulse_model(
        "DifferenceEquations",
        r#","impulse_every":[1.5, 0.0]"#,
    ))
    .err()
    .expect("1.5 is not an integer step");
    assert!(error.contains("'jump'") && error.contains("1.5"), "{error}");
}

#[test]
fn pulse_vaccination_reaches_the_periodic_orbit() {
    // S' = mu K - mu S between jumps, S -> (1 - p) S at t = k T.
    let (mu, k, p, period) = (0.02_f64, 1000.0_f64, 0.3_f64, 5.0_f64);
    let model = model(
        "DifferentialEquations",
        &[("S", 1000.0), ("V", 0.0)],
        &[("mu", "0.02"), ("K", "1000.0"), ("p", "0.3")],
        &[],
        &[
            transition("birth", &[], &["S"], "mu * K", ""),
            transition("death", &["S"], &[], "mu", ""),
            transition(
                "vaccination",
                &["S"],
                &["V"],
                "p",
                r#","impulse_every":[5.0, 5.0]"#,
            ),
        ],
        TIGHT,
    );
    let positions: Vec<f64> = (1..=200).map(|j| j as f64 * period).collect();
    let rows = engine(&model).run_at(&positions).unwrap();
    let decay = (-mu * period).exp();
    let before_jump = k * (1.0 - decay) / (1.0 - (1.0 - p) * decay);
    let after_jump = (1.0 - p) * before_jump;
    assert_close(rows.last().unwrap()[0], after_jump, 1e-7, "S+*");
}

#[test]
fn the_engine_rejects_a_difference_model_and_bad_tolerances() {
    let error = DifferentialEquations::try_from_model(&impulse_model("DifferenceEquations", ""))
        .err()
        .unwrap();
    assert!(error.contains("DifferenceEquations"), "{error}");
    let error = DifferentialEquations::try_from_model(&decay_model(
        r#","solver":{"method":"tsit45","rtol":0.0,"atol":1e-8}"#,
    ))
    .err()
    .unwrap();
    assert!(error.contains("rtol"), "{error}");
}

#[test]
fn a_missing_parameter_is_an_error_not_a_nan() {
    let model = model(
        "DifferentialEquations",
        &[("A", 1000.0), ("B", 0.0)],
        &[],
        &[],
        &[transition("ab", &["A"], &["B"], "missing * A", "")],
        "",
    );
    let error = engine(&model).run(2).unwrap_err();
    assert!(error.contains("missing"), "{error}");
}

#[test]
fn reset_and_calibration_parameters_rerun_the_model() {
    let mut engine = engine(&decay_model(TIGHT));
    let first = engine.run(5).unwrap();
    engine.reset();
    engine.set_parameter("k", 0.6).unwrap();
    let second = engine.run(5).unwrap();
    assert_close(
        second[5][0],
        1000.0 * (-3.0_f64).exp(),
        1e-7,
        "A(5), k = 0.6",
    );
    assert!(first[5][0] > second[5][0]);
    let mut buffer = Vec::new();
    engine.reset();
    engine
        .run_at_positions_into_buffer(&[0.0, 2.5], &mut buffer)
        .unwrap();
    assert_close(buffer[1][0], 1000.0 * (-1.5_f64).exp(), 1e-7, "A(2.5)");
}

#[test]
fn a_large_stiff_model_uses_the_sparse_path() {
    // A chain of `n` compartments, each emptying into the next.
    let n = super::SPARSE_THRESHOLD + 10;
    let ids: Vec<String> = (0..n).map(|i| format!("X{i}")).collect();
    let mut bins: Vec<(&str, f64)> = ids.iter().map(|id| (id.as_str(), 0.0)).collect();
    bins[0].1 = 1000.0;
    let transitions: Vec<String> = (0..n - 1)
        .map(|i| transition(&format!("t{i}"), &[&ids[i]], &[&ids[i + 1]], "k", ""))
        .collect();
    let model = model(
        "DifferentialEquations",
        &bins,
        &[("k", "100.0")],
        &[],
        &transitions,
        r#","solver":{"method":"bdf","rtol":1e-6,"atol":1e-6}"#,
    );
    let rows = engine(&model).run(5).unwrap();
    let total: f64 = rows[5].iter().sum();
    assert_close(total, 1000.0, 1e-8, "conservation");
    assert_close(rows[5][0], 1000.0 * (-500.0_f64).exp(), 1e-6, "X0(5)");
}

/// A stiff chain of `n` compartments, rates alternating 100 and 0.01: its
/// Jacobian has two diagonals.
fn stiff_chain(n: usize) -> Model {
    let ids: Vec<String> = (0..n).map(|i| format!("X{i}")).collect();
    let mut bins: Vec<(&str, f64)> = ids.iter().map(|id| (id.as_str(), 0.0)).collect();
    bins[0].1 = 1000.0;
    let transitions: Vec<String> = (0..n - 1)
        .map(|i| {
            let rate = if i % 2 == 0 { "fast" } else { "slow" };
            transition(&format!("t{i}"), &[&ids[i]], &[&ids[i + 1]], rate, "")
        })
        .collect();
    model(
        "DifferentialEquations",
        &bins,
        &[("fast", "100.0"), ("slow", "0.01")],
        &[],
        &transitions,
        r#","solver":{"method":"bdf","rtol":1e-6,"atol":1e-6}"#,
    )
}

/// A, B, C stratified into `groups` categories; A → B at `k1 * B / N` reads
/// the bin total `B`, so every A row of the Jacobian reads every B
/// compartment.
fn coupled_groups(groups: usize, method: &str) -> Model {
    let categories: Vec<String> = (0..groups).map(|g| format!(r#""g{g}""#)).collect();
    let fractions: Vec<String> = (0..groups)
        .map(|g| {
            format!(
                r#"{{"category":"g{g}","fraction":{}}}"#,
                1.0 / groups as f64
            )
        })
        .collect();
    let json = format!(
        r#"{{"name":"m","population":{{"bins":[{{"id":"A","name":"A"}},{{"id":"B","name":"B"}},
        {{"id":"C","name":"C"}}],"accumulators":[],
        "stratifications":[{{"id":"group","categories":[{}]}}],"transitions":[],
        "initial_conditions":{{"population_size":10000,"bin_fractions":[
        {{"bin":"A","fraction":0.99}},{{"bin":"B","fraction":0.01}},{{"bin":"C","fraction":0.0}}],
        "stratification_fractions":[{{"stratification":"group","fractions":[{}]}}]}}}},
        "parameters":[{{"id":"k1","value":0.3}},{{"id":"k2","value":0.1}}],
        "dynamics":{{"typology":"DifferentialEquations","transitions":[{},{}],
        "solver":{{"method":"{method}","rtol":1e-6,"atol":1e-6}}}}}}"#,
        categories.join(","),
        fractions.join(","),
        transition("ab", &["A"], &["B"], "k1 * B / N", ""),
        transition("bc", &["B"], &["C"], "k2", ""),
    );
    serde_json::from_str(&json).expect("valid model JSON")
}

/// Fastest of `repeats` runs of `run`, in milliseconds.
fn best_ms(repeats: usize, mut run: impl FnMut()) -> f64 {
    (0..repeats)
        .map(|_| {
            let started = Instant::now();
            run();
            started.elapsed().as_secs_f64() * 1e3
        })
        .fold(f64::INFINITY, f64::min)
}

fn bdf_ms(model: &Model, sparse: bool, steps: u32) -> f64 {
    let engine = engine(model);
    let positions: Vec<f64> = (0..=steps).map(f64::from).collect();
    best_ms(3, || {
        let mut engine = engine.clone();
        let mut state = engine.state.clone();
        let mut rows = vec![Vec::new(); positions.len() - 1];
        super::solve::integrate_with(
            &mut engine.model,
            &engine.settings,
            sparse,
            0.0,
            &mut state,
            &positions[1..],
            &mut rows,
        )
        .unwrap();
    })
}

/// Dense against sparse `bdf`, and the cost against the difference engine.
/// Run with `cargo test --release -p commol-differential -- --ignored
/// --nocapture benchmark`.
#[test]
#[ignore = "benchmark: run on demand in release mode"]
fn benchmark_dense_against_sparse_bdf() {
    println!("| model | states | dense bdf (ms) | sparse bdf (ms) |");
    println!("|---|---|---|---|");
    for n in [16, 32, 64, 128, 256, 512] {
        let chain = stiff_chain(n);
        println!(
            "| stiff chain | {n} | {:.2} | {:.2} |",
            bdf_ms(&chain, false, 50),
            bdf_ms(&chain, true, 50)
        );
    }
    for groups in [8, 16, 32, 64, 128] {
        let coupled = coupled_groups(groups, "bdf");
        println!(
            "| coupled groups | {} | {:.2} | {:.2} |",
            3 * groups,
            bdf_ms(&coupled, false, 50),
            bdf_ms(&coupled, true, 50)
        );
    }

    println!();
    println!("| model | states | difference (ms) | tsit45 (ms) | bdf (ms) |");
    println!("|---|---|---|---|---|");
    for groups in [1, 10, 100] {
        let mut difference_model = coupled_groups(groups, "bdf");
        difference_model.dynamics.typology = commol_core::ModelTypes::DifferenceEquations;
        difference_model.dynamics.solver = None;
        let difference = DifferenceEquations::try_from_model(&difference_model).unwrap();
        let tsit45 = engine(&coupled_groups(groups, "tsit45"));
        let bdf = engine(&coupled_groups(groups, "bdf"));
        println!(
            "| coupled groups, 100 steps | {} | {:.3} | {:.3} | {:.3} |",
            3 * groups,
            best_ms(5, || {
                difference.clone().run(100).unwrap();
            }),
            best_ms(5, || {
                tsit45.clone().run(100).unwrap();
            }),
            best_ms(5, || {
                bdf.clone().run(100).unwrap();
            }),
        );
    }
}
