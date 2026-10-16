//! The generator every stochastic engine draws from.

use rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

/// The golden-ratio increment of SplitMix64.
const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// One step of SplitMix64 (Steele, Lea and Flood 2014): advance `state` and
/// return the next number of its sequence.
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(GOLDEN_GAMMA);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The seed of replicate `index` of a run seeded with `seed`: the SplitMix64
/// of `seed + index`.
///
/// Neighbouring replicates get unrelated seeds, and replicate `index` gets the
/// same seed however many replicates run.
pub fn replicate_seed(seed: u64, index: u64) -> u64 {
    let mut state = seed.wrapping_add(index);
    splitmix64(&mut state)
}

/// A `Xoshiro256PlusPlus` generator (Blackman and Vigna 2021).
///
/// Its four words of state are the first four SplitMix64 numbers of the
/// seed, so one seed gives one sequence on every platform and version.
#[derive(Clone, Debug)]
pub struct Generator {
    rng: Xoshiro256PlusPlus,
}

impl Generator {
    /// A generator seeded with `seed`.
    pub fn from_seed(seed: u64) -> Self {
        let mut state = seed;
        let mut bytes = [0u8; 32];
        for chunk in bytes.as_chunks_mut::<8>().0 {
            *chunk = splitmix64(&mut state).to_le_bytes();
        }
        Self {
            rng: Xoshiro256PlusPlus::from_seed(bytes),
        }
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.rng.next_u64()
    }

    /// A number in `[0, 1)`: the 53 high bits of [`Self::next_u64`], scaled.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// A number in `(0, 1]`, so that its logarithm is finite.
    pub fn uniform_positive(&mut self) -> f64 {
        ((self.next_u64() >> 11) + 1) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix64_matches_its_reference_sequence() {
        // The first numbers of SplitMix64 from state 0 (Vigna, splitmix64.c).
        let mut state = 0;
        assert_eq!(splitmix64(&mut state), 0xE220_A839_7B1D_CDAF);
        assert_eq!(splitmix64(&mut state), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(splitmix64(&mut state), 0x06C4_5D18_8009_454F);
    }

    #[test]
    fn one_seed_gives_one_sequence() {
        let mut first = Generator::from_seed(42);
        let mut second = Generator::from_seed(42);
        let mut other = Generator::from_seed(43);
        let a: Vec<u64> = (0..8).map(|_| first.next_u64()).collect();
        let b: Vec<u64> = (0..8).map(|_| second.next_u64()).collect();
        let c: Vec<u64> = (0..8).map(|_| other.next_u64()).collect();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn uniforms_stay_in_their_intervals() {
        let mut generator = Generator::from_seed(7);
        for _ in 0..10_000 {
            let u = generator.uniform();
            assert!((0.0..1.0).contains(&u));
            let v = generator.uniform_positive();
            assert!(v > 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn replicate_seeds_differ_and_do_not_depend_on_their_number() {
        let seeds: Vec<u64> = (0..100).map(|index| replicate_seed(5, index)).collect();
        let mut unique = seeds.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), seeds.len());
        assert_eq!(replicate_seed(5, 3), seeds[3]);
        let mut state = 8;
        assert_eq!(replicate_seed(5, 3), splitmix64(&mut state));
    }
}
