//! Latin hypercubes on the unit cube.

use crate::Generator;
use crate::sample::index;

/// A Latin hypercube of `rows` points in `[0, 1)^columns`, point after
/// point.
///
/// Each column cuts `[0, 1)` into `rows` strata of equal width and puts one
/// point in each, at a uniform position inside it; the strata of each column
/// come in their own random order. Column after column, the generator draws
/// the permutation of the strata (Fisher–Yates), then one uniform per
/// stratum, so a seed gives the same points on every platform.
pub fn latin_hypercube(generator: &mut Generator, rows: usize, columns: usize) -> Vec<Vec<f64>> {
    let mut points = vec![Vec::with_capacity(columns); rows];
    for _ in 0..columns {
        let mut strata: Vec<usize> = (0..rows).collect();
        for i in (1..rows).rev() {
            strata.swap(i, index(generator, i + 1));
        }
        for (point, stratum) in points.iter_mut().zip(strata) {
            point.push((stratum as f64 + generator.uniform()) / rows as f64);
        }
    }
    points
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strata(points: &[Vec<f64>], column: usize) -> Vec<usize> {
        let mut strata: Vec<usize> = points
            .iter()
            .map(|point| (point[column] * points.len() as f64) as usize)
            .collect();
        strata.sort_unstable();
        strata
    }

    #[test]
    fn each_column_puts_one_point_in_each_stratum() {
        let points = latin_hypercube(&mut Generator::from_seed(3), 8, 3);
        assert_eq!(points.len(), 8);
        for column in 0..3 {
            assert_eq!(strata(&points, column), (0..8).collect::<Vec<_>>());
        }
        assert!(points.iter().flatten().all(|&u| (0.0..1.0).contains(&u)));
    }

    #[test]
    fn each_column_orders_its_strata_on_its_own() {
        let points = latin_hypercube(&mut Generator::from_seed(3), 8, 2);
        let order = |column: usize| {
            points
                .iter()
                .map(|point| (point[column] * 8.0) as usize)
                .collect::<Vec<_>>()
        };
        assert_ne!(order(0), order(1));
    }

    #[test]
    fn one_seed_gives_one_hypercube() {
        let draw = |seed| latin_hypercube(&mut Generator::from_seed(seed), 5, 2);
        assert_eq!(draw(1), draw(1));
        assert_ne!(draw(1), draw(2));
    }

    #[test]
    fn no_rows_or_no_columns_give_empty_points() {
        let mut generator = Generator::from_seed(1);
        assert!(latin_hypercube(&mut generator, 0, 3).is_empty());
        assert_eq!(
            latin_hypercube(&mut generator, 2, 0),
            vec![Vec::new(), Vec::new()]
        );
    }
}
