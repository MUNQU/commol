//! The statistics of the dependence of a series: autocorrelations with gaps,
//! the robust portmanteau, the integrated time of Geyer and the factor of
//! the information of several series.

use commol_calibration::ObservedDataPoint;
use commol_inference::dependence::{SeriesDependence, series_dependence};
use commol_random::Generator;
use commol_random::sample::normal;

/// Observations of `label` at `positions`; their values are read apart.
fn observations(label: &str, positions: &[f64]) -> Vec<ObservedDataPoint> {
    positions
        .iter()
        .map(|&at| ObservedDataPoint::new(at, label.to_string(), 0.0))
        .collect()
}

/// The positions 0, 1, …, `count − 1`.
fn steps(count: usize) -> Vec<f64> {
    (0..count).map(|t| t as f64).collect()
}

fn analysis(observations: &[ObservedDataPoint], values: &[f64], lags: usize) -> SeriesDependence {
    series_dependence(observations, values, lags, 19, 0.9, 5).unwrap()
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} is not within {tolerance} of {expected}"
    );
}

/// Standard normal draws.
fn noise(count: usize, seed: u64) -> Vec<f64> {
    let mut generator = Generator::from_seed(seed);
    (0..count).map(|_| normal(&mut generator)).collect()
}

/// The autoregression `x_t = Σ_j phi_j x_{t−j} + e_t` of the innovations
/// `innovations`, from zeros.
fn autoregression(phi: &[f64], innovations: &[f64]) -> Vec<f64> {
    let mut series: Vec<f64> = Vec::with_capacity(innovations.len());
    for (t, &e) in innovations.iter().enumerate() {
        let past: f64 = phi
            .iter()
            .enumerate()
            .filter(|(j, _)| t > *j)
            .map(|(j, &p)| p * series[t - 1 - j])
            .sum();
        series.push(past + e);
    }
    series
}

/// `values` minus their least-squares fit on the polynomials of degree
/// below `parameters` in the position, by Gram–Schmidt.
fn residuals_of_fit(values: &[f64], parameters: usize) -> Vec<f64> {
    let n = values.len();
    let x = |t: usize| 2.0 * t as f64 / (n - 1) as f64 - 1.0;
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for degree in 0..parameters {
        let mut column: Vec<f64> = (0..n).map(|t| x(t).powi(degree as i32)).collect();
        for b in &basis {
            let dot: f64 = column.iter().zip(b).map(|(c, b)| c * b).sum();
            column.iter_mut().zip(b).for_each(|(c, b)| *c -= dot * b);
        }
        let norm = column.iter().map(|c| c * c).sum::<f64>().sqrt();
        basis.push(column.into_iter().map(|c| c / norm).collect());
    }
    let mut residuals = values.to_vec();
    for b in &basis {
        let dot: f64 = residuals.iter().zip(b).map(|(r, b)| r * b).sum();
        residuals.iter_mut().zip(b).for_each(|(r, b)| *r -= dot * b);
    }
    residuals
}

#[test]
fn the_statistics_of_a_known_series() {
    // x: 1, 2, 4, 3, 5, 6 at 0 … 5, centered: −2.5, −1.5, 0.5, −0.5, 1.5, 2.5.
    // y: 1, −1, 1, −1 at 0, 1, 3, 4, with a gap at 2.
    let mut points = observations("x", &steps(6));
    points.extend(observations("y", &[0.0, 1.0, 3.0, 4.0]));
    let values = [1.0, 2.0, 4.0, 3.0, 5.0, 6.0, 1.0, -1.0, 1.0, -1.0];

    let result = analysis(&points, &values, 2);

    let x = &result.series[0];
    // γ(0) = 17.5 / 6; the products at lag 1 sum to 5.75 over 5 pairs, at
    // lag 2 to −1 over 4.
    let variance = 17.5 / 6.0;
    let r1 = 5.75 / 5.0 / variance;
    let r2 = -1.0 / 4.0 / variance;
    close(x.correlations[0].unwrap(), r1, 1e-12);
    close(x.correlations[1].unwrap(), r2, 1e-12);
    assert_eq!(x.pairs, vec![5, 4]);
    // Q̃: 5.75² / Σ(products²) at lag 1, (−1)² / Σ(products²) at lag 2.
    let q = 5.75 * 5.75 / 29.3125 + 1.0 / 4.25;
    close(x.portmanteau.unwrap(), q, 1e-12);
    assert_eq!(x.degrees_of_freedom, 2);
    // χ²(2) has the survival function exp(−q / 2).
    close(x.p_value.unwrap(), (-q / 2.0).exp(), 1e-12);
    // Γ_0 = γ(0) + γ(1) > 0, Γ_1 = γ(2) + γ(3) = −0.25 + 0.25 / 3 ≤ 0: the
    // sum stops after Γ_0, so τ = 1 + 2 r(1).
    close(x.tau.unwrap(), 1.0 + 2.0 * r1, 1e-12);
    assert_eq!(x.cut_lag, Some(1));
    let (lower, upper) = x.tau_band.unwrap();
    assert!(lower <= upper);

    let y = &result.series[1];
    // Lag 1 has the pairs (0, 1) and (3, 4); lag 2 only (1, 3).
    assert_eq!(y.pairs, vec![2, 1]);
    close(y.correlations[0].unwrap(), -1.0, 1e-12);
    close(y.correlations[1].unwrap(), -1.0, 1e-12);
    assert_eq!(y.values, 4);

    // At their shared positions 0, 1, 3, 4: x is 1, 2, 3, 5 and y 1, −1, 1, −1.
    let correlation = result.correlations[0].correlation.unwrap();
    close(correlation, -3.0 / 35f64.sqrt(), 1e-12);
    assert_eq!(
        (
            result.correlations[0].first.as_str(),
            result.correlations[0].second.as_str()
        ),
        ("x", "y")
    );
}

#[test]
fn a_lag_with_few_pairs_leaves_the_portmanteau() {
    // Eight values with max_lag 7: lags 5, 6 and 7 read 3, 2 and 1 pairs,
    // fewer than 4, and leave Q̃ and its degrees of freedom; their
    // correlations stay.
    let values = noise(8, 9);
    let points = observations("x", &steps(8));

    let all = analysis(&points, &values, 7);
    let four = analysis(&points, &values, 4);

    let (x, x4) = (&all.series[0], &four.series[0]);
    assert_eq!(x.pairs, vec![7, 6, 5, 4, 3, 2, 1]);
    assert_eq!(x.few_pairs, vec![(5, 3), (6, 2), (7, 1)]);
    assert_eq!(x.degrees_of_freedom, 4);
    close(x.portmanteau.unwrap(), x4.portmanteau.unwrap(), 1e-12);
    close(x.p_value.unwrap(), x4.p_value.unwrap(), 1e-12);
    assert!(x.correlations.iter().all(Option::is_some));
    assert_eq!(x.portmanteau_reason, None);
    assert!(x4.few_pairs.is_empty());

    // A gap costs pairs: on 0 … 3 and 8 … 11, lags 1, 2 and 8 read 6, 4
    // and 4 pairs, lags 3 to 7 read 2, 0, 1, 2 and 3.
    let mut positions = steps(4);
    positions.extend((8..12).map(f64::from));
    let gapped = analysis(&observations("x", &positions), &noise(8, 10), 8);
    let x = &gapped.series[0];
    assert_eq!(x.pairs, vec![6, 4, 2, 0, 1, 2, 3, 4]);
    assert_eq!(x.few_pairs, vec![(3, 2), (4, 0), (5, 1), (6, 2), (7, 3)]);
    assert_eq!(x.degrees_of_freedom, 3);

    // With no lag of 4 pairs, Q̃ has no test, and says why; τ still reads
    // the series.
    let short = analysis(&observations("x", &steps(4)), &[1.0, 3.0, 2.0, 5.0], 3);
    let x = &short.series[0];
    assert_eq!(x.few_pairs, vec![(1, 3), (2, 2), (3, 1)]);
    assert_eq!(x.portmanteau, None);
    assert_eq!(x.degrees_of_freedom, 0);
    assert_eq!(x.p_value, None);
    let reason = x.portmanteau_reason.as_deref().unwrap();
    assert!(reason.contains("fewer than 4 pairs"), "{reason}");
    assert!(x.correlations.iter().all(Option::is_some));
    assert!(x.tau.is_some());
    assert_eq!(x.cut_lag, Some(1));

    // A constant series has lags of many pairs but no product other than
    // 0.
    let constant = analysis(&observations("x", &steps(8)), &[2.0; 8], 2);
    let reason = constant.series[0].portmanteau_reason.as_deref().unwrap();
    assert!(reason.contains("product other than 0"), "{reason}");
}

#[test]
fn the_time_of_an_ar1_series() {
    let values = autoregression(&[0.5], &noise(2000, 3));

    let result = analysis(&observations("x", &steps(2000)), &values, 10);

    // τ = (1 + ρ) / (1 − ρ) = 3.
    close(result.series[0].tau.unwrap(), 3.0, 0.45);
}

#[test]
fn the_edges_of_the_time() {
    // Alternating values: r̂(1) near −1.
    let alternating: Vec<f64> = noise(200, 1)
        .iter()
        .enumerate()
        .map(|(t, e)| if t % 2 == 0 { 1.0 } else { -1.0 } + 0.2 * e)
        .collect();
    let negative = analysis(&observations("x", &steps(200)), &alternating, 5);
    let series = &negative.series[0];
    assert!(series.correlations[0].unwrap() < -0.5);
    assert_eq!(series.tau, None);
    let reason = series.tau_reason.as_deref().unwrap();
    assert!(reason.contains("not above 0"), "{reason}");
    // The wild sets, of independent signs, have a band above 0: the
    // undefined τ reads below it.
    assert!(series.tau_band.unwrap().0 > f64::NEG_INFINITY);

    // The residuals of a fit of 10 parameters to 35 positions of an AR(1)
    // of 0.8: the median τ̂ over data sets is above 1.
    let mut taus: Vec<f64> = (0..101)
        .map(|seed| {
            let values = residuals_of_fit(&autoregression(&[0.8], &noise(35, seed)), 10);
            analysis(&observations("x", &steps(35)), &values, 7).series[0]
                .tau
                .unwrap_or(0.0)
        })
        .collect();
    taus.sort_by(f64::total_cmp);
    assert!(taus[50] > 1.0, "median {}", taus[50]);

    // An oscillating correlation: the AR(2) of φ = (1, −0.5) has r(1) = 2/3,
    // r(2) = 1/6 and r(3) = −1/6, so Γ_1 = 0 cuts the sum after Γ_0: τ̂
    // tends to 1 + 2 r(1) = 7/3, above the τ of the process, 5/3, as the
    // table of the guide shows.
    let values = autoregression(&[1.0, -0.5], &noise(2000, 4));
    let oscillating = analysis(&observations("x", &steps(2000)), &values, 10);
    let tau = oscillating.series[0].tau.unwrap();
    close(tau, 7.0 / 3.0, 0.35);
    assert!(tau > 5.0 / 3.0 + 0.2, "{tau}");
}

#[test]
fn a_common_shock_counts_the_information_twice() {
    let n = 2000;
    let both = |shock: &[f64]| -> SeriesDependence {
        let mut points = observations("x", &steps(n));
        points.extend(observations("y", &steps(n)));
        let mut values: Vec<f64> = shock.iter().zip(noise(n, 7)).map(|(c, e)| c + e).collect();
        values.extend(shock.iter().zip(noise(n, 8)).map(|(c, e)| c + e));
        analysis(&points, &values, 10)
    };

    let white = both(&noise(n, 6));
    let autocorrelated = both(&autoregression(&[0.8], &noise(n, 6)));

    // A white shock of the variance of the own noise: Var(Σz) / ΣVar(z) =
    // 1 + ρ = 1.5, and the sum is as independent as its parts.
    close(white.problem.variance_factor.unwrap(), 1.5, 0.15);
    close(white.problem.tau.unwrap(), 1.0, 0.25);
    assert!(white.problem.information_factor.unwrap() > 1.3);
    assert!(white.series.iter().all(|s| s.tau.unwrap() < 1.3));
    // An autocorrelated shock: the sum is dependent too.
    assert!(autocorrelated.problem.information_factor.unwrap() > 3.0);
    assert!(autocorrelated.problem.tau.unwrap() > 2.0);
    assert!(autocorrelated.problem.p_value.unwrap() <= 0.05);
}

#[test]
fn what_a_grid_reads_and_refuses() {
    // An observation of weight 0 is left out: a replicate of a resampler
    // can hold one.
    let mut points = observations("x", &steps(6));
    points.push(ObservedDataPoint {
        weight: 0.0,
        ..ObservedDataPoint::new(6.0, "x".to_string(), 0.0)
    });
    let values = [1.0, 2.0, 4.0, 3.0, 5.0, 6.0, 100.0];
    let weighted = analysis(&points, &values, 2);
    let alone = analysis(&points[..6], &values[..6], 2);
    assert_eq!(weighted.series[0].values, 6);
    assert_eq!(
        weighted.series[0].correlations,
        alone.series[0].correlations
    );
    assert_eq!(weighted.series[0].tau, alone.series[0].tau);

    // One position close to another sets a step that would read few pairs.
    let mut positions = steps(50);
    positions.push(0.01);
    positions.sort_by(f64::total_cmp);
    let sparse = analysis(&observations("x", &positions), &noise(51, 2), 3);
    let reason = sparse.series[0].reason.as_deref().unwrap();
    assert!(reason.contains("labels of their own"), "{reason}");

    // A constant series has no τ in any replicate either: no band, and no
    // test of the information factor.
    let constant = analysis(&observations("x", &steps(8)), &[2.0; 8], 2);
    assert_eq!(constant.series[0].tau, None);
    assert_eq!(constant.series[0].tau_band, None);
    assert_eq!(constant.problem.information_factor, None);
    assert_eq!(constant.problem.p_value, None);

    // Two series that share one position have no correlation, and positions
    // off one grid have no factor of the problem.
    let mut points = observations("x", &steps(5));
    points.extend(observations("y", &[4.0, 5.3, 6.6, 7.9, 9.2]));
    let values = [1.0, 3.0, 2.0, 5.0, 4.0, 2.0, 1.0, 3.0, 1.0, 2.0];
    let apart = analysis(&points, &values, 1);
    assert!(apart.series.iter().all(|s| s.reason.is_none()));
    assert_eq!(apart.correlations[0].correlation, None);
    assert_eq!(apart.problem.information_factor, None);
    assert!(
        apart
            .problem
            .tau_reason
            .as_deref()
            .unwrap()
            .contains("together")
    );

    // With no series that has statistics, the problem has none.
    let twice = observations("x", &[1.0, 1.0, 2.0]);
    let none = analysis(&twice, &[1.0, 2.0, 3.0], 1);
    assert!(none.series[0].reason.is_some());
    assert!(none.correlations.is_empty());
    assert_eq!(none.problem.variance_factor, None);
    assert_eq!(none.problem.p_value, None);
}
