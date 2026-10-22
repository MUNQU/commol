//! Quantities of interest: a series and its reduction, the rules for ties
//! and NaN, the form a quantity crosses in, and the NaN rule of every
//! reduction across runs.

use commol_calibration::CalibrationParameter;
use commol_core::Model;
use commol_difference::DifferenceEquations;
use commol_inference::design::InputSpace;
use commol_inference::global_sensitivity::{ChaosExpansion, ForwardAnalysis, morris, prcc, sobol};
use commol_inference::quantity::{MissingPositions, Quantities, Quantity, Series};
use commol_inference::statistics::{self, NotANumber};
use commol_inference::{NanRun, PredictiveError, Priors};

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
    ];

    let json = serde_json::to_string(&quantities).unwrap();
    let back: Vec<Quantity> = serde_json::from_str(&json).unwrap();

    assert_eq!(back, quantities);
    assert!(json.starts_with(
        r#"[{"kind":"value_at","series":{"kind":"level","outputs":["A","B"]},"at":1.5}"#
    ));
    assert!(json.contains(r#"{"kind":"increments","outputs":["A"],"window":7.0}"#));
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
    let fitted = ChaosExpansion::fit(&unit, &with_nan(5), 1).unwrap_err();
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
            replicate: 0,
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
            replicate: 1,
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
