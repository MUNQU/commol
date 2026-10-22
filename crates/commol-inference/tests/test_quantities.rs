//! Quantities of interest: a series and its reduction, the rules for ties
//! and NaN, the form a quantity crosses in, the NaN rule of every reduction
//! across runs, and the ratio of two quantities.

use commol_calibration::{CalibrationParameter, ForwardModel};
use commol_core::{Model, SimulationEngine};
use commol_difference::DifferenceEquations;
use commol_inference::batch::BatchRunner;
use commol_inference::design::InputSpace;
use commol_inference::global_sensitivity::{
    ChaosExpansion, ForwardAnalysis, Truncation, morris, prcc, sobol,
};
use commol_inference::quantity::{
    MissingPositions, Quantities, Quantity, ReplicateRule, Series, Undefined,
};
use commol_inference::statistics::{self, NotANumber};
use commol_inference::{NanRun, PredictiveError, Priors};
use commol_stochastic::StochasticSteps;

fn names(outputs: &[&str]) -> Vec<String> {
    outputs.iter().map(|name| name.to_string()).collect()
}

fn level(outputs: &[&str]) -> Series {
    Series::Level {
        outputs: names(outputs),
    }
}

fn increments(outputs: &[&str], window: f64) -> Series {
    Series::Increments {
        outputs: names(outputs),
        window,
    }
}

/// The rows of one output, `acc`, at the positions 0, 7, 14 and 21.
fn accumulator(values: [f64; 4]) -> (Vec<f64>, Vec<Vec<f64>>) {
    (
        vec![0.0, 7.0, 14.0, 21.0],
        values.iter().map(|&v| vec![v]).collect(),
    )
}

#[test]
fn the_peak_of_increments_is_the_largest_gain_of_a_window() {
    let quantities = Quantities::new(
        &[
            Quantity::Peak {
                series: increments(&["acc"], 7.0),
                positions: vec![7.0, 14.0, 21.0],
            },
            Quantity::PeakPosition {
                series: increments(&["acc"], 7.0),
                positions: vec![7.0, 14.0, 21.0],
            },
            Quantity::Trajectory {
                series: increments(&["acc"], 7.0),
                positions: vec![7.0, 14.0, 21.0],
            },
            Quantity::ValueAt {
                series: increments(&["acc"], 14.0),
                at: 21.0,
            },
            Quantity::Peak {
                series: level(&["acc"]),
                positions: vec![7.0, 14.0, 21.0],
            },
        ],
        &names(&["acc"]),
    )
    .unwrap();
    // The accumulator gains 1, 4 and 2 over three windows of 7.
    let (schedule, rows) = accumulator([0.0, 1.0, 5.0, 7.0]);

    assert_eq!(quantities.schedule(), schedule);
    assert_eq!(quantities.len(), 7);
    assert_eq!(
        quantities.evaluate(&rows),
        [4.0, 14.0, 1.0, 4.0, 2.0, 6.0, 7.0]
    );
}

#[test]
fn a_tie_gives_the_first_position() {
    let quantities = Quantities::new(
        &[Quantity::PeakPosition {
            series: level(&["acc"]),
            positions: vec![0.0, 7.0, 14.0, 21.0],
        }],
        &names(&["acc"]),
    )
    .unwrap();

    assert_eq!(
        quantities.evaluate(&accumulator([1.0, 3.0, 3.0, 2.0]).1),
        [7.0]
    );
}

#[test]
fn a_nan_makes_the_quantity_nan() {
    let positions = vec![0.0, 7.0, 14.0, 21.0];
    let quantities = Quantities::new(
        &[
            Quantity::Peak {
                series: level(&["acc"]),
                positions: positions.clone(),
            },
            Quantity::PeakPosition {
                series: level(&["acc"]),
                positions: positions.clone(),
            },
            Quantity::Trajectory {
                series: level(&["acc"]),
                positions: vec![7.0, 14.0],
            },
            Quantity::ValueAt {
                series: level(&["acc"]),
                at: 21.0,
            },
        ],
        &names(&["acc"]),
    )
    .unwrap();

    // `f64::max` would skip the NaN, and `>` would keep the first position.
    let values = quantities.evaluate(&accumulator([1.0, f64::NAN, 9.0, 2.0]).1);

    assert!(values[0].is_nan() && values[1].is_nan(), "{values:?}");
    assert!(values[2].is_nan());
    assert_eq!(values[3..], [9.0, 2.0]);
}

#[test]
fn a_done_run_is_read_at_the_schedule_within_the_tolerance_of_the_axis() {
    let quantities = [
        Quantity::Gain {
            outputs: names(&["acc"]),
            start: 0.0,
            end: 0.3,
        },
        Quantity::ValueAt {
            series: increments(&["acc"], 0.1),
            at: 0.3,
        },
    ];
    let compiled = Quantities::new(&quantities, &names(&["other", "acc"])).unwrap();
    // The run holds 0.2, where the window of 0.1 that ends at 0.3 starts at
    // 0.19999999999999998, and positions the quantities do not read.
    let positions = [0.0, 0.1, 0.2, 0.3 + 1e-12, 0.4];
    let rows: Vec<Vec<f64>> = [0.0, 1.0, 3.0, 6.0, 10.0]
        .iter()
        .map(|&acc| vec![-1.0, acc])
        .collect();

    assert_eq!(compiled.evaluate_run(&positions, &rows), Ok(vec![6.0, 3.0]));
    assert_eq!(
        Quantities::schedule_of(&quantities).unwrap(),
        compiled.schedule()
    );
}

#[test]
fn a_done_run_lists_every_position_it_lacks_with_its_quantity() {
    let quantities = Quantities::new(
        &[
            Quantity::ValueAt {
                series: level(&["acc"]),
                at: 7.0,
            },
            Quantity::Peak {
                series: increments(&["acc"], 7.0),
                positions: vec![14.0, 21.0],
            },
        ],
        &names(&["acc"]),
    )
    .unwrap();
    let (_, rows) = accumulator([0.0, 1.0, 5.0, 7.0]);

    let missing = quantities
        .evaluate_run(&[0.0, 7.0, 10.0, 21.0], &rows)
        .unwrap_err();

    assert_eq!(
        missing,
        [MissingPositions {
            quantity: 1,
            positions: vec![14.0],
        }]
    );
}

#[test]
fn the_schedule_alone_reads_the_outputs_the_quantities_name() {
    let schedule = Quantities::schedule_of(&[
        Quantity::Trajectory {
            series: level(&["B", "A"]),
            positions: vec![5.0, 1.0],
        },
        Quantity::Gain {
            outputs: names(&["C"]),
            start: 2.0,
            end: 3.0,
        },
    ])
    .unwrap();

    assert_eq!(schedule, [1.0, 2.0, 3.0, 5.0]);
    assert!(Quantities::schedule_of(&[]).is_err());
}

#[test]
fn increments_cannot_start_before_zero() {
    let compile = |series: Series, at: f64| {
        Quantities::new(&[Quantity::ValueAt { series, at }], &names(&["acc"])).unwrap_err()
    };

    let early = compile(increments(&["acc"], 7.0), 5.0);
    let empty = compile(increments(&["acc"], 0.0), 5.0);
    let infinite = compile(increments(&["acc"], f64::INFINITY), 5.0);
    let no_outputs = compile(increments(&[], 1.0), 5.0);
    let no_positions = Quantities::new(
        &[Quantity::Trajectory {
            series: level(&["acc"]),
            positions: vec![],
        }],
        &names(&["acc"]),
    )
    .unwrap_err();

    assert!(
        early.contains("windows of 7") && early.contains("the end 5"),
        "{early}"
    );
    assert!(empty.contains("above 0, got 0"), "{empty}");
    assert!(infinite.contains("finite"), "{infinite}");
    assert!(no_outputs.contains("no outputs"), "{no_outputs}");
    assert!(no_positions.contains("needs positions"), "{no_positions}");
}

#[test]
fn each_quantity_crosses_as_json_tagged_by_kind() {
    let quantities = vec![
        Quantity::ValueAt {
            series: level(&["A", "B"]),
            at: 1.5,
        },
        Quantity::Gain {
            outputs: names(&["A"]),
            start: 0.1,
            end: 0.4,
        },
        Quantity::Peak {
            series: increments(&["A"], 7.0),
            positions: vec![7.0, 14.0],
        },
        Quantity::PeakPosition {
            series: level(&["A"]),
            positions: vec![1.0],
        },
        Quantity::Trajectory {
            series: increments(&["A"], 1.0),
            positions: vec![1.0, 2.0],
        },
        ratio_of(
            value_at(&["A"], 1.0),
            Quantity::Gain {
                outputs: names(&["B"]),
                start: 0.0,
                end: 1.0,
            },
            ReplicateRule::WithinPositive,
        ),
    ];

    let json = serde_json::to_string(&quantities).unwrap();
    let back: Vec<Quantity> = serde_json::from_str(&json).unwrap();

    assert_eq!(back, quantities);
    assert!(json.starts_with(
        r#"[{"kind":"value_at","series":{"kind":"level","outputs":["A","B"]},"at":1.5}"#
    ));
    assert!(json.contains(r#"{"kind":"increments","outputs":["A"],"window":7.0}"#));
    assert!(
        json.contains(r#""replicate_rule":"within_positive""#),
        "{json}"
    );
    assert!(serde_json::from_str::<Quantity>(r#"{"kind":"mean","at":1.0}"#).is_err());
    assert!(quantities.iter().all(|quantity| !quantity.is_empty()));
}

/// The error of a quantity NaN at the points 1 and 3 of five.
fn nan_at_one_and_three() -> PredictiveError {
    PredictiveError::NotANumber {
        quantity: 0,
        points: vec![1, 3],
        run: None,
    }
}

/// One quantity at each of `points`, NaN at the points 1 and 3.
fn with_nan(points: usize) -> Vec<Vec<f64>> {
    (0..points)
        .map(|i| vec![if i == 1 || i == 3 { f64::NAN } else { i as f64 }])
        .collect()
}

fn unit_inputs(count: usize) -> InputSpace {
    let names: Vec<String> = (0..count).map(|i| format!("x{i}")).collect();
    InputSpace::Independent(Priors::uniform_on_bounds(&names, &vec![(0.0, 1.0); count]).unwrap())
}

#[test]
fn every_estimator_rejects_a_nan() {
    let inputs = unit_inputs(1);
    let screening = inputs.morris("Morris", 3, 4, 1).unwrap();
    let saltelli = inputs.saltelli("Sobol", 2, 1).unwrap();
    let points = inputs.latin_hypercube("PolynomialChaos", 5, 1).unwrap();
    let unit = inputs.to_unit("PolynomialChaos", &points).unwrap();

    let ranked = prcc(&points, &with_nan(5), &[1.0; 5]).unwrap_err();
    let effects = morris(&screening, &with_nan(screening.points.len())).unwrap_err();
    let indices = sobol(&saltelli, &with_nan(saltelli.points.len()), 10, 0.9, 1).unwrap_err();
    let fitted = ChaosExpansion::fit(&unit, &with_nan(5), 1, Truncation::TOTAL_DEGREE).unwrap_err();
    let quantile = statistics::quantiles(&[0.0, f64::NAN, 1.0], &[1.0; 3], &[0.5]).unwrap_err();

    assert_eq!(ranked, nan_at_one_and_three());
    assert_eq!(effects, nan_at_one_and_three());
    assert_eq!(indices, nan_at_one_and_three());
    assert_eq!(fitted, nan_at_one_and_three());
    assert_eq!(quantile, NotANumber { points: vec![1] });
    assert!(String::from(quantile).contains("[1] are NaN"));
    let message = morris(&screening, &[vec![0.0]]).unwrap_err().to_string();
    assert!(
        message.contains("one row of quantities per point"),
        "{message}"
    );
}

/// Bins A, B and C of 1e308, 0.79e308 and 1e308: each is finite, and so is
/// A + B, but A + B + C is infinite, so its increments are NaN.
fn overflow() -> Model {
    let json = r#"{"name":"m","parameters":[{"id":"k","value":0.1}],
        "engine":{"type":"DifferenceEquations","settings":null},
        "process":{"bins":[{"id":"A","name":"A"},{"id":"B","name":"B"},{"id":"C","name":"C"}],
        "accumulators":[],"stratifications":[],
        "transitions":[{"id":"ab","source":["A"],"target":["B"],"rate":"k * 0.001"}]},
        "structure":{"kind":"aggregate","bin_amounts":{"A":1e308,"B":0.79e308,"C":1e308}}}"#;
    serde_json::from_str(json).expect("valid model JSON")
}

#[test]
fn a_nan_run_is_named_by_its_point_and_inputs() {
    let engine = DifferenceEquations::try_from_model(&overflow()).unwrap();
    let inputs =
        InputSpace::Independent(Priors::uniform_on_bounds(&names(&["k"]), &[(0.0, 1.0)]).unwrap());
    let analysis = ForwardAnalysis::new(
        engine,
        vec![CalibrationParameter::new("k".to_string(), 0.0, 1.0)],
        None,
        None,
        inputs,
        5,
    )
    .unwrap();
    let quantities = [
        Quantity::ValueAt {
            series: level(&["A", "B"]),
            at: 2.0,
        },
        Quantity::Peak {
            series: increments(&["A", "B", "C"], 1.0),
            positions: vec![1.0, 2.0],
        },
    ];

    let propagation = analysis.propagate(Some(4), &quantities).unwrap();
    let error = analysis.prcc(Some(4), &quantities).unwrap_err();

    // A propagation keeps the NaN of each run.
    assert!(propagation.values.iter().all(|row| row[1].is_nan()));
    let PredictiveError::NotANumber {
        quantity: 1,
        points,
        run: Some(run),
    } = &error
    else {
        panic!("{error:?}");
    };
    assert_eq!(points, &[0, 1, 2, 3]);
    assert_eq!(
        **run,
        NanRun {
            draw: 0,
            replicate: Some(0),
            seed: None,
            inputs: propagation.sample.values()[0].clone(),
        }
    );
    let message = error.to_string();
    assert!(
        message.contains("first at the point 0") && message.contains("Propagation"),
        "{message}"
    );
}

#[test]
fn the_seed_of_a_nan_replicate_is_named() {
    let error = PredictiveError::NotANumber {
        quantity: 2,
        points: vec![4],
        run: Some(Box::new(NanRun {
            draw: 4,
            replicate: Some(1),
            seed: Some(17),
            inputs: vec![0.5],
        })),
    };

    let message = error.to_string();

    assert!(
        message.contains("point 4 of the inputs [0.5], replicate 1 of seed 17"),
        "{message}"
    );
}

fn value_at(outputs: &[&str], at: f64) -> Quantity {
    Quantity::ValueAt {
        series: level(outputs),
        at,
    }
}

fn ratio_of(numerator: Quantity, denominator: Quantity, replicate_rule: ReplicateRule) -> Quantity {
    Quantity::Ratio {
        numerator: Box::new(numerator),
        denominator: Box::new(denominator),
        replicate_rule,
    }
}

/// The ratio of the level of `N` to the level of `D` at 0, combined by
/// `rule`.
fn n_over_d(rule: ReplicateRule) -> Quantity {
    ratio_of(value_at(&["N"], 0.0), value_at(&["D"], 0.0), rule)
}

const RULES: [ReplicateRule; 3] = [
    ReplicateRule::Within,
    ReplicateRule::WithinPositive,
    ReplicateRule::Pooled,
];

/// The value of a ratio of `N` to `D` under each rule, for replicates of the
/// values `(N, D)` given.
fn combined(replicates: &[(f64, f64)]) -> Vec<Result<f64, Undefined>> {
    let quantities = Quantities::new(&RULES.map(n_over_d), &names(&["N", "D"])).unwrap();
    let terms: Vec<_> = replicates
        .iter()
        .map(|&(n, d)| quantities.terms(&[vec![n, d]]))
        .collect();
    quantities.combine(&terms)
}

#[test]
fn a_ratio_combines_replicates_as_the_user_chooses() {
    assert_eq!(
        combined(&[(1.0, 2.0), (3.0, 6.0)]),
        [Ok(0.5), Ok(0.5), Ok(0.5)]
    );
    assert_eq!(
        combined(&[(1.0, 1.0), (0.0, 4.0)]),
        [Ok(0.5), Ok(0.5), Ok(0.2)]
    );
    // A NaN term leaves every rule with no value, and names its replicate.
    assert_eq!(
        combined(&[(1.0, 1.0), (f64::NAN, 4.0)]),
        [Err(Undefined::Replicate(1)); 3]
    );
}

#[test]
fn a_ratio_divides_two_values_of_one_run() {
    let quantities = [
        ratio_of(
            Quantity::Peak {
                series: level(&["acc"]),
                positions: vec![7.0, 14.0],
            },
            Quantity::Gain {
                outputs: names(&["acc"]),
                start: 7.0,
                end: 21.0,
            },
            ReplicateRule::Pooled,
        ),
        value_at(&["acc"], 21.0),
    ];
    let compiled = Quantities::new(&quantities, &names(&["acc"])).unwrap();
    let (positions, rows) = accumulator([0.0, 1.0, 5.0, 7.0]);
    let (_, empty) = accumulator([0.0, 1.0, 1.0, 1.0]);

    assert_eq!(compiled.len(), 2);
    assert_eq!(compiled.schedule(), [7.0, 14.0, 21.0]);
    assert_eq!(
        Quantities::schedule_of(&quantities).unwrap(),
        compiled.schedule()
    );
    // The peak 5 over the gain 6; a zero denominator is undefined, even
    // over a numerator of 1.
    assert_eq!(compiled.evaluate(&rows[1..]), [5.0 / 6.0, 7.0]);
    assert!(compiled.evaluate(&empty[1..])[0].is_nan());
    assert!(compiled.evaluate_run(&positions, &empty).unwrap()[0].is_nan());
}

#[test]
fn a_ratio_takes_values_only() {
    let outputs = names(&["acc"]);
    let of = |operand: Quantity| {
        Quantities::new(
            &[ratio_of(
                operand,
                value_at(&["acc"], 1.0),
                ReplicateRule::Within,
            )],
            &outputs,
        )
        .unwrap_err()
    };

    let position = of(Quantity::PeakPosition {
        series: level(&["acc"]),
        positions: vec![1.0],
    });
    let trajectory = of(Quantity::Trajectory {
        series: level(&["acc"]),
        positions: vec![1.0],
    });
    let nested = of(n_over_d(ReplicateRule::Within));

    assert!(position.contains("got peak_position"), "{position}");
    assert!(trajectory.contains("got trajectory"), "{trajectory}");
    assert!(nested.contains("got ratio"), "{nested}");
}

/// A → B at the proportional rate `k`, from one unit of A, and two units
/// of C that never move, run by the engine of chain binomial steps.
fn one_unit() -> StochasticSteps {
    let json = r#"{"name":"m","parameters":[{"id":"k","value":0.5}],
        "engine":{"type":"StochasticSteps","settings":null},
        "process":{"bins":[{"id":"A","name":"A"},{"id":"B","name":"B"},{"id":"C","name":"C"}],
        "accumulators":[],"stratifications":[],
        "transitions":[{"id":"ab","source":["A"],"target":["B"],"rate":"k"}]},
        "structure":{"kind":"aggregate","bin_amounts":{"A":1.0,"B":0.0,"C":2.0}}}"#;
    let model: Model = serde_json::from_str(json).expect("valid model JSON");
    StochasticSteps::try_from_model(&model).unwrap()
}

/// The three units over B at `at`, combined by `rule`.
fn all_over_b(at: f64, rule: ReplicateRule) -> Quantity {
    ratio_of(value_at(&["A", "B", "C"], 1.0), value_at(&["B"], at), rule)
}

/// The forward model of [`one_unit`], 8 replicates of `k`.
fn one_unit_forward() -> ForwardModel<StochasticSteps> {
    let k = vec![CalibrationParameter::new("k".to_string(), 0.0, 1.0)];
    ForwardModel::unobserved(one_unit(), k, None, Some(8), 3).unwrap()
}

/// The replicates of `quantity` at two points of `k = 0.5`, combined.
fn means_of(quantity: Quantity) -> Result<Vec<Vec<f64>>, PredictiveError> {
    let forward = one_unit_forward();
    let quantities = Quantities::new(&[quantity], &forward.base_engine().output_names()).unwrap();
    BatchRunner::new(&forward, &quantities)
        .unwrap()
        .means(&[vec![0.5], vec![0.5]])
}

#[test]
fn a_zero_denominator_is_undefined_unless_the_user_conditions() {
    let forward = one_unit_forward();
    let within = Quantities::new(
        &[all_over_b(1.0, ReplicateRule::Within)],
        &forward.base_engine().output_names(),
    )
    .unwrap();
    let runner = BatchRunner::new(&forward, &within).unwrap();
    let ratios: Vec<f64> = runner.replicates(&[vec![0.5]]).unwrap()[0]
        .iter()
        .map(|values| values[0])
        .collect();
    // B is 1 in some replicates, where the ratio is 3, and 0 in the others.
    let first_zero = ratios.iter().position(|ratio| ratio.is_nan()).unwrap();
    assert!(ratios.iter().all(|&ratio| ratio.is_nan() || ratio == 3.0));
    assert!(ratios.contains(&3.0), "{ratios:?}");

    let error = means_of(all_over_b(1.0, ReplicateRule::Within)).unwrap_err();
    let positive = means_of(all_over_b(1.0, ReplicateRule::WithinPositive)).unwrap();

    assert_eq!(
        error,
        PredictiveError::NotANumber {
            quantity: 0,
            points: vec![0, 1],
            run: Some(Box::new(NanRun {
                draw: 0,
                replicate: Some(first_zero),
                seed: Some(forward.replicates().unwrap().seeds[first_zero]),
                inputs: vec![0.5],
            })),
        }
    );
    assert_eq!(positive, [[3.0], [3.0]]);
}

#[test]
fn a_draw_with_no_positive_denominator_raises() {
    // B is 0 at 0 in every replicate.
    for rule in [ReplicateRule::WithinPositive, ReplicateRule::Pooled] {
        let error = means_of(all_over_b(0.0, rule)).unwrap_err();

        assert_eq!(
            error,
            PredictiveError::NotANumber {
                quantity: 0,
                points: vec![0, 1],
                run: Some(Box::new(NanRun {
                    draw: 0,
                    replicate: None,
                    seed: None,
                    inputs: vec![0.5],
                })),
            }
        );
        let message = error.to_string();
        assert!(
            message.contains("point 0 of the inputs [0.5] (no replicate is NaN"),
            "{message}"
        );
    }
}

/// A → B at the rate `max(0, k − 0.5)` from A = 1000: B gains nothing over
/// the first step when `k` is at most 0.5.
fn gated() -> ForwardAnalysis<DifferenceEquations> {
    let json = r#"{"name":"m","parameters":[{"id":"k","value":0.7}],
        "engine":{"type":"DifferenceEquations","settings":null},
        "process":{"bins":[{"id":"A","name":"A"},{"id":"B","name":"B"}],
        "accumulators":[],"stratifications":[],
        "transitions":[{"id":"ab","source":["A"],"target":["B"],"rate":"max(0, k - 0.5)"}]},
        "structure":{"kind":"aggregate","bin_amounts":{"A":1000.0,"B":0.0}}}"#;
    let model: Model = serde_json::from_str(json).expect("valid model JSON");
    ForwardAnalysis::new(
        DifferenceEquations::try_from_model(&model).unwrap(),
        vec![CalibrationParameter::new("k".to_string(), 0.0, 1.0)],
        None,
        None,
        InputSpace::Independent(Priors::uniform_on_bounds(&names(&["k"]), &[(0.0, 1.0)]).unwrap()),
        5,
    )
    .unwrap()
}

/// A at 1 over the gain of B over (0, 1], combined by `rule`.
fn a_over_gain(rule: ReplicateRule) -> Vec<Quantity> {
    vec![ratio_of(
        value_at(&["A"], 1.0),
        Quantity::Gain {
            outputs: names(&["B"]),
            start: 0.0,
            end: 1.0,
        },
        rule,
    )]
}

/// The point and the replicate of the first run a NaN error names.
fn named_run(error: PredictiveError) -> (Vec<usize>, NanRun) {
    let PredictiveError::NotANumber {
        quantity: 0,
        points,
        run: Some(run),
    } = error
    else {
        panic!("{error:?}");
    };
    (points, *run)
}

#[test]
fn a_zero_denominator_follows_the_nan_rule_in_every_method() {
    let analysis = gated();

    for (rule, replicate) in [
        (ReplicateRule::Within, Some(0)),
        (ReplicateRule::Pooled, None),
    ] {
        let quantities = a_over_gain(rule);
        let errors = [
            analysis.prcc(Some(8), &quantities).map(|_| ()).unwrap_err(),
            analysis.morris(3, 4, &quantities).map(|_| ()).unwrap_err(),
            analysis
                .sobol(4, 2, 0.9, &quantities)
                .map(|_| ())
                .unwrap_err(),
            analysis
                .chaos(1, Truncation::TOTAL_DEGREE, 8, &quantities)
                .map(|_| ())
                .unwrap_err(),
        ];

        for error in errors {
            let (points, run) = named_run(error);
            assert_eq!(points[0], run.draw);
            assert!(run.inputs[0] <= 0.5, "{run:?}");
            assert_eq!(run.replicate, replicate);
        }
    }
    // A propagation keeps the NaN of each run, which its quantiles reject.
    let propagation = analysis
        .propagate(Some(8), &a_over_gain(ReplicateRule::Within))
        .unwrap();
    let ratios: Vec<f64> = propagation.values.iter().map(|row| row[0]).collect();
    let undefined: Vec<usize> = (0..ratios.len())
        .filter(|&run| propagation.sample.values()[run][0] <= 0.5)
        .collect();
    assert!(!undefined.is_empty());
    let error = statistics::quantiles(&ratios, &propagation.weights, &[0.5]).unwrap_err();
    assert_eq!(error, NotANumber { points: undefined });
}
