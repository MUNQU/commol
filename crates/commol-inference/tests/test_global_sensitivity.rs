//! Global sensitivity analysis: the quantities of interest, the batch runner
//! and the methods of independent and dependent inputs.

use commol_calibration::{
    CalibrationParameter, CalibrationProblem, EvaluationError, ForwardModel, LossConfig,
    ObservedDataPoint, Violation, Workspace,
};
use commol_core::{Model, SimulationEngine};
use commol_difference::DifferenceEquations;
use commol_inference::Priors;
use commol_inference::batch::BatchRunner;
use commol_inference::design::InputSpace;
use commol_inference::global_sensitivity::{
    ChaosExpansion, ForwardAnalysis, Truncation, morris, prcc, sobol,
};
use commol_inference::quantity::{Quantities, Quantity, Series};
use commol_inference::{MethodError, ParameterSample, Predictive, PredictiveError};
use commol_stochastic::StochasticSteps;
use nalgebra::{DMatrix, DVector};

/// A → B at the proportional rate `k`, from A = 1000, run by `engine`.
fn decay(engine: &str) -> Model {
    let json = format!(
        r#"{{"name":"m","parameters":[{{"id":"k","value":0.1}}],
        "engine":{{"type":"{engine}","settings":null}},
        "process":{{"bins":[{{"id":"A","name":"A"}},{{"id":"B","name":"B"}}],
        "accumulators":[],"stratifications":[],
        "transitions":[{{"id":"ab","source":["A"],"target":["B"],"rate":"k"}}]}},
        "structure":{{"kind":"aggregate","bin_amounts":{{"A":1000.0,"B":0.0}}}}}}"#
    );
    serde_json::from_str(&json).expect("valid model JSON")
}

fn k() -> Vec<CalibrationParameter> {
    vec![CalibrationParameter::new("k".to_string(), 0.0, 1.0)]
}

fn names(outputs: &[&str]) -> Vec<String> {
    outputs.iter().map(|name| name.to_string()).collect()
}

fn level(outputs: &[&str]) -> Series {
    Series::Level {
        outputs: names(outputs),
    }
}

#[test]
fn quantities_equal_hand_computations() {
    let engine = DifferenceEquations::try_from_model(&decay("DifferenceEquations")).unwrap();
    let forward = ForwardModel::unobserved(engine.clone(), k(), None, None, 0).unwrap();
    let quantities = Quantities::new(
        &[
            Quantity::ValueAt {
                series: level(&["A"]),
                at: 2.0,
            },
            Quantity::ValueAt {
                series: level(&["A", "B"]),
                at: 3.0,
            },
            Quantity::Gain {
                outputs: names(&["B"]),
                start: 1.0,
                end: 3.0,
            },
            Quantity::Peak {
                series: level(&["A"]),
                positions: vec![1.0, 2.0, 3.0],
            },
            Quantity::PeakPosition {
                series: level(&["B"]),
                positions: vec![1.0, 2.0, 3.0],
            },
        ],
        &engine.output_names(),
    )
    .unwrap();
    assert_eq!(quantities.len(), 5);
    assert!(!quantities.is_empty());
    let runner = BatchRunner::new(&forward, &quantities).unwrap();

    let values = runner.means(&[vec![0.2]]).unwrap();

    // A = 1000 · 0.8ᵗ and A + B = 1000.
    let expected = [640.0, 1000.0, 288.0, 800.0, 3.0];
    for (value, expected) in values[0].iter().zip(expected) {
        assert!((value - expected).abs() < 1e-9, "{value} ≠ {expected}");
    }

    // The gain is the observation with the same window.
    let mut windowed = ObservedDataPoint::new(3.0, "B".to_string(), 0.0);
    windowed.window = Some(2.0);
    let problem = CalibrationProblem::new(
        engine,
        vec![windowed],
        k(),
        vec![],
        LossConfig::SumSquaredError,
    )
    .unwrap();
    let observed = problem
        .forward()
        .run_observed(&[0.2], &mut Workspace::default())
        .unwrap()
        .to_vec();
    assert_eq!(values[0][2], observed[0]);
}

#[test]
fn a_gain_reads_its_start_exactly() {
    let outputs = names(&["A", "B"]);
    let gain = Quantities::new(
        &[Quantity::Gain {
            outputs: names(&["B"]),
            start: 0.1,
            end: 0.4,
        }],
        &outputs,
    )
    .unwrap();
    let with_value = Quantities::new(
        &[
            Quantity::ValueAt {
                series: level(&["B"]),
                at: 0.1,
            },
            Quantity::Gain {
                outputs: names(&["B"]),
                start: 0.1,
                end: 0.4,
            },
        ],
        &outputs,
    )
    .unwrap();

    // 0.4 − (0.4 − 0.1) is not 0.1.
    assert_ne!(0.4 - (0.4 - 0.1), 0.1);
    assert_eq!(gain.schedule(), [0.1, 0.4]);
    assert_eq!(with_value.schedule(), [0.1, 0.4]);
    // B is 2 at 0.1 and 7 at 0.4.
    let rows = [vec![0.0, 2.0], vec![0.0, 7.0]];
    assert_eq!(gain.evaluate(&rows), [5.0]);
    assert_eq!(with_value.evaluate(&rows), [2.0, 5.0]);
}

#[test]
fn stochastic_points_average_replicates_with_common_seeds() {
    let engine = StochasticSteps::try_from_model(&decay("StochasticSteps")).unwrap();
    let forward = ForwardModel::unobserved(engine.clone(), k(), None, Some(4), 9).unwrap();
    let quantities = Quantities::new(
        &[Quantity::ValueAt {
            series: level(&["A"]),
            at: 5.0,
        }],
        &engine.output_names(),
    )
    .unwrap();
    let runner = BatchRunner::new(&forward, &quantities).unwrap();

    let replicates = runner.replicates(&[vec![0.2], vec![0.3]]).unwrap();
    let means = runner.means(&[vec![0.3], vec![0.2], vec![0.2]]).unwrap();

    assert_eq!(replicates[0].len(), 4);
    let mean = replicates[0].iter().map(|values| values[0]).sum::<f64>() / 4.0;
    assert!((means[1][0] - mean).abs() < 1e-9);
    // The same point gives the same value wherever it is in a design.
    assert_eq!(means[1], means[2]);
    let distinct = replicates[0]
        .iter()
        .any(|values| values[0] != replicates[0][0][0]);
    assert!(distinct, "the replicates have their own seeds");
}

#[test]
fn a_quantity_reads_only_what_the_model_has() {
    let engine = DifferenceEquations::try_from_model(&decay("DifferenceEquations")).unwrap();
    let forward = ForwardModel::unobserved(engine.clone(), k(), None, None, 0).unwrap();
    let outputs = engine.output_names();
    let quantity = |quantity: Quantity| Quantities::new(&[quantity], &outputs);

    let unknown = quantity(Quantity::ValueAt {
        series: level(&["C"]),
        at: 1.0,
    })
    .unwrap_err();
    let backwards = quantity(Quantity::Gain {
        outputs: names(&["B"]),
        start: 3.0,
        end: 1.0,
    })
    .unwrap_err();
    let no_positions = quantity(Quantity::Peak {
        series: level(&["A"]),
        positions: vec![],
    })
    .unwrap_err();
    let no_outputs = quantity(Quantity::ValueAt {
        series: Series::Level { outputs: vec![] },
        at: 1.0,
    })
    .unwrap_err();
    let not_finite = quantity(Quantity::ValueAt {
        series: level(&["A"]),
        at: f64::NAN,
    })
    .unwrap_err();
    let none = Quantities::new(&[], &outputs).unwrap_err();
    let half_step = quantity(Quantity::ValueAt {
        series: level(&["A"]),
        at: 1.5,
    })
    .unwrap();
    let rejected = BatchRunner::new(&forward, &half_step)
        .err()
        .unwrap()
        .to_string();

    assert!(unknown.contains("'C'"), "{unknown}");
    assert!(backwards.contains("ends after it starts"), "{backwards}");
    assert!(no_positions.contains("needs positions"), "{no_positions}");
    assert!(no_outputs.contains("no outputs"), "{no_outputs}");
    assert!(not_finite.contains("finite"), "{not_finite}");
    assert!(none.contains("No quantities"), "{none}");
    assert!(rejected.contains("1.5"), "{rejected}");
}

/// `count` points of `columns` independent uniform inputs on `[0, 1)`.
fn uniform_points(count: usize, columns: usize, seed: u64) -> Vec<Vec<f64>> {
    commol_random::latin_hypercube(
        &mut commol_random::Generator::from_seed(seed),
        count,
        columns,
    )
}

#[test]
fn prcc_finds_monotone_inputs_and_ignores_a_dummy() {
    let inputs = uniform_points(500, 3, 11);
    let outputs: Vec<Vec<f64>> = inputs
        .iter()
        .map(|x| vec![2.0 * x[0] + x[1].powi(3)])
        .collect();

    let result = prcc(&inputs, &outputs, &vec![1.0; inputs.len()]).unwrap();

    let [x1, x2, dummy] = result.coefficients[0][..] else {
        panic!("three coefficients")
    };
    // The ranks of the sum are not an exact function of the ranks of x1 and
    // x2, so their PRCCs are high, not 1.
    assert!(x1 > 0.85 && x2 > 0.85, "{x1} {x2}");
    assert!(dummy.abs() < 0.1, "{dummy}");
    assert!(result.p_values[0][0] < 1e-10 && result.p_values[0][1] < 1e-10);
    assert_eq!(result.effective_size.round(), 500.0);
}

/// The unweighted PRCC by its definition: the correlation of the residuals
/// of the usual mid-ranks of the input and of the output, each regressed by
/// least squares on the mid-ranks of the other inputs.
fn textbook_prcc(inputs: &[Vec<f64>], outputs: &[f64], input: usize) -> f64 {
    let ranks = |values: Vec<f64>| -> Vec<f64> {
        values
            .iter()
            .map(|v| {
                let below = values.iter().filter(|w| *w < v).count() as f64;
                let equal = values.iter().filter(|w| *w == v).count() as f64;
                below + (equal + 1.0) / 2.0
            })
            .collect()
    };
    let n = inputs.len();
    let others: Vec<Vec<f64>> = (0..inputs[0].len())
        .filter(|&j| j != input)
        .map(|j| ranks(inputs.iter().map(|x| x[j]).collect()))
        .collect();
    let design = DMatrix::from_fn(n, others.len() + 1, |i, j| {
        if j == 0 { 1.0 } else { others[j - 1][i] }
    });
    let residuals = |y: Vec<f64>| -> DVector<f64> {
        let y = DVector::from_vec(y);
        let fit = design
            .clone()
            .svd(true, true)
            .solve(&y, 1e-12)
            .expect("a least-squares fit");
        &y - &design * fit
    };
    let a = residuals(ranks(inputs.iter().map(|x| x[input]).collect()));
    let b = residuals(ranks(outputs.to_vec()));
    a.dot(&b) / (a.norm() * b.norm())
}

#[test]
fn weighted_prcc_with_equal_weights_is_prcc() {
    let inputs = uniform_points(40, 3, 5);
    let outputs: Vec<Vec<f64>> = inputs
        .iter()
        .map(|x| vec![x[0] - 0.5 * x[1] * x[2], (x[2] * 3.0).floor()])
        .collect();

    let result = prcc(&inputs, &outputs, &vec![0.025; 40]).unwrap();

    for q in 0..2 {
        let column: Vec<f64> = outputs.iter().map(|row| row[q]).collect();
        for j in 0..3 {
            let expected = textbook_prcc(&inputs, &column, j);
            let actual = result.coefficients[q][j];
            assert!(
                (actual - expected).abs() < 1e-10,
                "{q} {j}: {actual} ≠ {expected}"
            );
        }
    }
}

#[test]
fn a_weight_of_two_is_a_repeated_draw() {
    let inputs = uniform_points(30, 2, 8);
    let outputs: Vec<Vec<f64>> = inputs.iter().map(|x| vec![x[0] + x[1] * x[1]]).collect();
    let mut weights = vec![1.0; 30];
    weights[3] = 2.0;
    let mut repeated_inputs = inputs.clone();
    let mut repeated_outputs = outputs.clone();
    repeated_inputs.push(inputs[3].clone());
    repeated_outputs.push(outputs[3].clone());

    let weighted = prcc(&inputs, &outputs, &weights).unwrap();
    let repeated = prcc(&repeated_inputs, &repeated_outputs, &vec![1.0; 31]).unwrap();

    for (a, b) in weighted.coefficients[0]
        .iter()
        .zip(&repeated.coefficients[0])
    {
        assert!((a - b).abs() < 1e-12, "{a} ≠ {b}");
    }
    // The repeated draw adds no information, so the effective size of the
    // weights is below the 31 draws of the repetition.
    assert!(weighted.effective_size < 31.0);
}

/// The g-function of Sobol on `[0, 1]^k`: `Π (|4x − 2| + a) / (1 + a)`; an
/// input matters less the larger its `a`.
fn g_function(x: &[f64], a: &[f64]) -> f64 {
    x.iter()
        .zip(a)
        .map(|(x, a)| ((4.0 * x - 2.0).abs() + a) / (1.0 + a))
        .product()
}

fn unit_inputs(count: usize) -> InputSpace {
    let names: Vec<String> = (0..count).map(|i| format!("x{i}")).collect();
    InputSpace::Independent(Priors::uniform_on_bounds(&names, &vec![(0.0, 1.0); count]).unwrap())
}

#[test]
fn morris_ranks_the_g_function() {
    let a = [0.0, 1.0, 4.5, 9.0, 99.0, 99.0];
    let design = unit_inputs(6).morris("Morris", 60, 4, 3).unwrap();
    let outputs: Vec<Vec<f64>> = design
        .points
        .iter()
        .map(|x| vec![g_function(x, &a)])
        .collect();

    let result = morris(&design, &outputs).unwrap();

    let mu_star = &result.mu_star[0];
    for i in 0..3 {
        assert!(mu_star[i] > mu_star[i + 1], "{mu_star:?}");
    }
    assert!(
        mu_star[4] < mu_star[3] && mu_star[5] < mu_star[3],
        "{mu_star:?}"
    );
    assert_eq!(result.sigma[0].len(), 6);
}

/// The Ishigami function, `sin x1 + 7 sin² x2 + 0.1 x3⁴ sin x1`, on
/// `[−π, π]³`.
fn ishigami(x: &[f64]) -> f64 {
    x[0].sin() + 7.0 * x[1].sin().powi(2) + 0.1 * x[2].powi(4) * x[0].sin()
}

fn ishigami_inputs() -> InputSpace {
    let names: Vec<String> = (1..=3).map(|i| format!("x{i}")).collect();
    let pi = std::f64::consts::PI;
    InputSpace::Independent(Priors::uniform_on_bounds(&names, &[(-pi, pi); 3]).unwrap())
}

/// The analytic first-order and total indices of the Ishigami function with
/// `a = 7` and `b = 0.1`.
const ISHIGAMI_FIRST: [f64; 3] = [0.3139, 0.4424, 0.0];
const ISHIGAMI_TOTAL: [f64; 3] = [0.5576, 0.4424, 0.2437];

#[test]
fn a_saltelli_design_needs_two_base_draws() {
    let error = ishigami_inputs().saltelli("Sobol", 1, 21).unwrap_err();

    assert_eq!(error, "Sobol needs at least 2 base draws, got 1");
}

#[test]
fn sobol_indices_of_ishigami() {
    let design = ishigami_inputs().saltelli("Sobol", 8192, 21).unwrap();
    let outputs: Vec<Vec<f64>> = design.points.iter().map(|x| vec![ishigami(x)]).collect();

    let result = sobol(&design, &outputs, 500, 0.99, 21).unwrap();

    for i in 0..3 {
        let (low, high) = (
            result.first_interval[0][0][i],
            result.first_interval[1][0][i],
        );
        assert!(
            low <= ISHIGAMI_FIRST[i] && ISHIGAMI_FIRST[i] <= high,
            "S1 of x{}: [{low}, {high}] misses {}",
            i + 1,
            ISHIGAMI_FIRST[i]
        );
        let (low, high) = (
            result.total_interval[0][0][i],
            result.total_interval[1][0][i],
        );
        assert!(
            low <= ISHIGAMI_TOTAL[i] && ISHIGAMI_TOTAL[i] <= high,
            "ST of x{}: [{low}, {high}] misses {}",
            i + 1,
            ISHIGAMI_TOTAL[i]
        );
        assert!((result.first[0][i] - ISHIGAMI_FIRST[i]).abs() < 0.05);
        assert!((result.total[0][i] - ISHIGAMI_TOTAL[i]).abs() < 0.05);
    }
    assert_eq!(design.points.len(), 8192 * 5);
}

#[test]
fn chaos_indices_of_ishigami_with_fewer_runs() {
    let inputs = ishigami_inputs();
    let points = inputs.latin_hypercube("PolynomialChaos", 1200, 4).unwrap();
    let unit = inputs.to_unit("PolynomialChaos", &points).unwrap();
    let outputs: Vec<Vec<f64>> = points.iter().map(|x| vec![ishigami(x)]).collect();

    let chaos = ChaosExpansion::fit(&unit, &outputs, 10, Truncation::TOTAL_DEGREE).unwrap();

    let (first, total) = (chaos.first(), chaos.total());
    for i in 0..3 {
        assert!(
            (first[0][i] - ISHIGAMI_FIRST[i]).abs() < 0.02,
            "S1 of x{}: {}",
            i + 1,
            first[0][i]
        );
        assert!(
            (total[0][i] - ISHIGAMI_TOTAL[i]).abs() < 0.02,
            "ST of x{}: {}",
            i + 1,
            total[0][i]
        );
    }
    assert!(chaos.loo_error[0] < 0.01, "{}", chaos.loo_error[0]);
    // 1200 runs against the 8192 · 5 of the Sobol design.
    assert!(points.len() < 8192 * 5);
    // The expansion is a surrogate of the function.
    let x = [0.5, -1.0, 2.0];
    let u = inputs.to_unit("PolynomialChaos", &[x.to_vec()]).unwrap();
    assert!((chaos.evaluate(&u[0])[0] - ishigami(&x)).abs() < 0.2);
}

#[test]
fn a_truncated_chaos_basis_fits_many_inputs() {
    // The Ishigami function of x1, x2 and x3, padded with 17 inputs it does
    // not read.
    let names: Vec<String> = (1..=20).map(|i| format!("x{i}")).collect();
    let pi = std::f64::consts::PI;
    let inputs =
        InputSpace::Independent(Priors::uniform_on_bounds(&names, &[(-pi, pi); 20]).unwrap());
    let pairs = Truncation::new(1.0, Some(2)).unwrap();
    // 1 + 20 · 4 + C(20, 2) · 6, against 10,626 for the total degree.
    assert_eq!(ChaosExpansion::terms(20, 4, pairs), 1_221);
    let basis = ChaosExpansion::new(20, 4, pairs, vec![]).unwrap();
    assert_eq!(basis.indices.len(), 1_221);
    let used = |index: &Vec<usize>| index.iter().filter(|&&a| a > 0).count();
    assert!(basis.indices.iter().all(|index| used(index) <= 2));
    assert_eq!(
        basis
            .indices
            .iter()
            .filter(|index| used(index) == 2)
            .count(),
        190 * 6
    );

    // A fit of those 1,221 terms needs more points than terms, and a dense
    // least squares of that size takes minutes in a test build; a q-norm of
    // 0.5 keeps 271 of them, the pairs of degree one each.
    let truncation = Truncation::new(0.5, Some(2)).unwrap();
    assert_eq!(ChaosExpansion::terms(20, 4, truncation), 1 + 20 * 4 + 190);
    let points = inputs.latin_hypercube("PolynomialChaos", 800, 4).unwrap();
    let unit = inputs.to_unit("PolynomialChaos", &points).unwrap();
    let outputs: Vec<Vec<f64>> = points.iter().map(|x| vec![ishigami(x)]).collect();

    let chaos = ChaosExpansion::fit(&unit, &outputs, 4, truncation).unwrap();

    let (first, total) = (chaos.first(), chaos.total());
    for i in 0..2 {
        assert!(
            (first[0][i] - ISHIGAMI_FIRST[i]).abs() < 0.06,
            "S1 of x{}: {}",
            i + 1,
            first[0][i]
        );
    }
    // The inputs the function does not read explain almost nothing.
    for (i, share) in total[0].iter().enumerate().skip(3) {
        assert!(*share < 0.05, "ST of x{}: {share}", i + 1);
    }
    // Degree 4 misses the higher harmonics of x2 and the x3^4 of the
    // interaction, but predicts better than the mean.
    assert!(chaos.loo_error[0] < 0.7, "{}", chaos.loo_error[0]);
}

fn decay_inputs() -> InputSpace {
    InputSpace::Independent(Priors::uniform_on_bounds(&["k".to_string()], &[(0.0, 1.0)]).unwrap())
}

fn analysis<E: SimulationEngine + Send + Sync>(
    engine: E,
    replicates: Option<usize>,
) -> ForwardAnalysis<E> {
    ForwardAnalysis::new(engine, k(), None, replicates, decay_inputs(), 5).unwrap()
}

fn a_at(at: f64) -> Vec<Quantity> {
    vec![Quantity::ValueAt {
        series: level(&["A"]),
        at,
    }]
}

#[test]
fn a_propagation_runs_the_draws_as_their_predictive() {
    let engine = StochasticSteps::try_from_model(&decay("StochasticSteps")).unwrap();
    let analysis = analysis(engine, Some(3));

    let result = analysis.propagate(Some(4), &a_at(5.0)).unwrap();
    let predictive =
        Predictive::new(analysis.forward(), &result.sample, &[5.0], analysis.seed()).unwrap();

    assert_eq!(result.values.len(), 4 * 3);
    let from_predictive: Vec<Vec<f64>> = predictive
        .trajectories()
        .iter()
        .map(|rows| vec![rows[0][0]])
        .collect();
    assert_eq!(result.values, from_predictive);
    assert_eq!(result.weights, predictive.weights());
    // Each replicate has a seed of its own, unlike the common seeds of a
    // design.
    let means = BatchRunner::new(
        analysis.forward(),
        &Quantities::new(&a_at(5.0), &analysis.forward().base_engine().output_names()).unwrap(),
    )
    .unwrap()
    .replicates(result.sample.values())
    .unwrap();
    let common: Vec<f64> = means.iter().flatten().map(|row| row[0]).collect();
    let fresh: Vec<f64> = result.values.iter().map(|row| row[0]).collect();
    assert_ne!(common, fresh);
}

#[test]
fn an_analysis_takes_draws_by_its_inputs() {
    let engine = DifferenceEquations::try_from_model(&decay("DifferenceEquations")).unwrap();
    let independent = analysis(engine.clone(), None);
    let sample = ParameterSample::new(
        vec!["k".to_string()],
        vec![(0.0, 1.0)],
        vec![vec![0.1], vec![0.2], vec![0.3], vec![0.4]],
        &[0.0, -0.5, -1.0, -1.5],
        None,
    )
    .unwrap();
    let of_sample = ForwardAnalysis::new(
        engine.clone(),
        k(),
        None,
        None,
        InputSpace::Sample(sample.clone()),
        5,
    )
    .unwrap();

    let needs = independent.draws("PRCC", None).unwrap_err().to_string();
    let unset = of_sample.draws("PRCC", Some(4)).unwrap_err().to_string();
    let drawn = independent.draws("PRCC", Some(6)).unwrap();

    assert!(
        needs.contains("PRCC of independent inputs needs draws"),
        "{needs}"
    );
    assert!(unset.contains("leave draws unset"), "{unset}");
    assert_eq!(drawn.len(), 6);
    assert_eq!(of_sample.draws("PRCC", None).unwrap(), sample);
    let (design, result) = of_sample.prcc(None, &a_at(3.0)).unwrap();
    assert_eq!(design, sample);
    // A is a falling function of k alone.
    assert!((result.coefficients[0][0] + 1.0).abs() < 1e-12);
    let other = ForwardAnalysis::new(
        engine,
        vec![CalibrationParameter::new("k".to_string(), 0.0, 2.0)],
        None,
        None,
        decay_inputs(),
        5,
    )
    .err()
    .unwrap()
    .to_string();
    assert!(other.contains("the parameters are"), "{other}");
}

#[test]
fn an_analysis_of_no_replicates_is_refused() {
    let engine = StochasticSteps::try_from_model(&decay("StochasticSteps")).unwrap();

    let error = ForwardAnalysis::new(engine, k(), None, Some(0), decay_inputs(), 5)
        .err()
        .unwrap()
        .to_string();

    assert_eq!(error, "A stochastic engine needs at least one replicate");
}

#[test]
fn a_propagation_names_the_draw_that_has_no_run() {
    // With k = 1.5 the first step takes 1500 from A, which holds 1000.
    let engine = DifferenceEquations::try_from_model(&decay("DifferenceEquations")).unwrap();
    let parameters = vec![CalibrationParameter::new("k".to_string(), 0.0, 2.0)];
    let sample = ParameterSample::new(
        vec!["k".to_string()],
        vec![(0.0, 2.0)],
        vec![vec![0.5], vec![1.5]],
        &[0.0, 0.0],
        None,
    )
    .unwrap();
    let analysis = ForwardAnalysis::new(
        engine,
        parameters,
        None,
        None,
        InputSpace::Sample(sample),
        5,
    )
    .unwrap();

    let error = analysis.propagate(None, &a_at(3.0)).unwrap_err();

    let MethodError::Prediction(PredictiveError::Draw {
        draw: 1,
        error: EvaluationError::Violation(Violation::InfeasibleRun { message }),
    }) = error
    else {
        panic!("{error:?}");
    };
    assert!(
        message.contains("leave compartment 'A'") && message.contains("step 0"),
        "{message}"
    );
}

#[test]
fn an_analysis_runs_each_method_in_one_call() {
    let engine = DifferenceEquations::try_from_model(&decay("DifferenceEquations")).unwrap();
    let analysis = analysis(engine, None);

    let (screening, morris_runs) = analysis.morris(3, 4, &a_at(3.0)).unwrap();
    let (indices, sobol_runs) = analysis.sobol(16, 20, 0.9, &a_at(3.0)).unwrap();
    let (chaos, chaos_runs) = analysis
        .chaos(2, Truncation::TOTAL_DEGREE, 10, &a_at(3.0))
        .unwrap();

    assert_eq!(morris_runs, 3 * 2);
    assert!(screening.mu[0][0] < 0.0);
    assert_eq!(sobol_runs, 16 * 3);
    // One input explains all the variance.
    assert!((indices.total[0][0] - 1.0).abs() < 0.2);
    assert_eq!(chaos_runs, 10);
    assert!((chaos.total()[0][0] - 1.0).abs() < 1e-9);
    let few = analysis
        .chaos(2, Truncation::TOTAL_DEGREE, 3, &a_at(3.0))
        .unwrap_err()
        .to_string();
    assert!(
        few.contains("has 3 terms and needs more points than terms, got 3"),
        "{few}"
    );
    assert!(analysis.sobol(16, 1, 0.9, &a_at(3.0)).is_err());
    assert!(analysis.sobol(16, 20, 1.0, &a_at(3.0)).is_err());
    let unknown = analysis
        .morris(
            2,
            4,
            &[Quantity::ValueAt {
                series: level(&["C"]),
                at: 1.0,
            }],
        )
        .unwrap_err()
        .to_string();
    assert!(unknown.contains("'C'"), "{unknown}");
}
