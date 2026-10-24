//! The seed streams of every method that draws from a seed: the starts of a
//! calibration and the inference methods. One number for each method and
//! purpose, so that two methods that take one seed never draw from one
//! sequence.
//!
//! Every random number of a method follows one scheme: the seed of its
//! source, then a stream of this registry ([`Stream::seed`]), then, for a
//! method of many draws or replicates, the index of the draw
//! ([`Stream::replicate`], `replicate_seed(stream, index)`), with a mix at
//! each level. No method draws from the seed of its source or from its
//! replicates. [`Stream::ALL`] lists every stream with its method and
//! purpose, and the compiler rejects two streams of one number. The
//! registry lives in `commol-random`, below every crate that draws, so no
//! crate takes a stream number outside it.

use crate::generator::{mixed, replicate_seed, splitmix64};

/// The seed of stream `stream` of a run seeded with `seed`: it mixes
/// `stream` into the mixed seed with a multiplier, then takes the SplitMix64
/// of the result.
fn stream_seed(seed: u64, stream: u64) -> u64 {
    let mut state = mixed(seed) ^ stream.wrapping_mul(0xD1B5_4A32_D192_ED03);
    splitmix64(&mut state)
}

/// The streams, each as `Name = number, "method", "purpose";`: the enum,
/// its list and its descriptions come from this one table.
macro_rules! streams {
    ($($name:ident = $number:literal, $method:literal, $purpose:literal;)+) => {
        /// A seed stream of a method.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[repr(u64)]
        pub enum Stream {
            $(
                #[doc = concat!("Stream ", $number, " (", $method, "): ", $purpose, ".")]
                $name = $number,
            )+
        }

        impl Stream {
            /// Every stream, in the order of their numbers.
            pub const ALL: &'static [Stream] = &[$(Stream::$name),+];

            /// The method that draws from the stream.
            pub fn method(self) -> &'static str {
                match self {
                    $(Stream::$name => $method,)+
                }
            }

            /// What the method draws from the stream.
            pub fn purpose(self) -> &'static str {
                match self {
                    $(Stream::$name => $purpose,)+
                }
            }
        }
    };
}

streams! {
    SensitivityDesign = 1, "a sensitivity analysis",
        "the designs: the Latin hypercube, the Morris trajectories and the Saltelli matrices";
    SensitivityReplicates = 2, "a sensitivity analysis",
        "the replicates of a stochastic engine at each point of a design, common to every point";
    SobolBootstrap = 3, "Sobol", "the rows of each bootstrap replicate";
    PredictiveDraws = 4, "the predictive of a sample and a propagation",
        "the seed of each draw, whose replicates a stochastic engine runs";
    PitTies = 5, "the PIT of observation draws",
        "the place of an observation among the draws that tie with it";
    LocalDraws = 6, "the Laplace approximation and the sandwich",
        "the draws of the normal or the Student t on the unbounded scale";
    ImportanceDraws = 7, "importance sampling and the Laplace mixture",
        "the draws of the proposal";
    SmcParticles = 8, "the SMC sampler",
        "the generator of each particle: its draw of the prior, its moves and its runs";
    SmcResampling = 9, "the SMC sampler", "the uniform of each systematic resampling";
    EtaWeights = 10, "bootstrap coverage of eta",
        "the Bayesian-bootstrap weights of each replicate";
    EtaSamplers = 11, "bootstrap coverage of eta",
        "the seed of the tempered sampler of each replicate";
    ReplicateData = 12, "the bootstraps and the bootstrap thresholds of a profile",
        "the data of each replicate";
    ReplicateFits = 13, "every method that fits replicates (bootstraps, jackknife, conformal)",
        "the seed of the fit of each replicate: its Latin hypercube and its starts";
    EnbpiBlocks = 14, "EnbPI", "the block counts of each replicate";
    ThresholdRefits = 15, "the bootstrap thresholds of a profile",
        "the seed of the refits of each replicate with a quantity held";
    ProfileRefits = 16, "a profile",
        "the seed of each refit along a side of the profile of each quantity";
    ObservationNoise = 17, "observation draws", "the noise around the values of each draw";
    SampleResampling = 18, "a parameter sample", "the draws taken with replacement by weight";
    DistributionDraws = 19, "a distribution", "the draws of its sampler";
    PosteriorReplicate = 20, "a tempered posterior", "the one replicate of a stochastic model at every point";
    StartPoints = 21, "the starts of a calibration", "the Latin hypercube of the start points";
    StartSeeds = 22, "the starts of a calibration", "the seed of the particle swarm of each start";
}

impl Stream {
    /// The number of the stream.
    pub fn number(self) -> u64 {
        self as u64
    }

    /// The seed of the stream of a method seeded with `seed`.
    pub fn seed(self, seed: u64) -> u64 {
        stream_seed(seed, self.number())
    }

    /// The seed of draw or replicate `index` of the stream of a method seeded
    /// with `seed`: `replicate_seed(self.seed(seed), index)`.
    pub fn replicate(self, seed: u64, index: u64) -> u64 {
        replicate_seed(self.seed(seed), index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_stream_has_its_own_seed() {
        let seeds: Vec<u64> = Stream::ALL[..4]
            .iter()
            .map(|stream| stream.seed(7))
            .collect();
        for (i, a) in seeds.iter().enumerate() {
            for b in &seeds[i + 1..] {
                assert_ne!(a, b);
            }
            for index in 0..100 {
                assert_ne!(*a, replicate_seed(7, index));
            }
        }
        let stream = Stream::SensitivityReplicates;
        assert_eq!(stream.seed(7), stream.seed(7));
        assert_ne!(stream.seed(7), stream.seed(8));
    }

    #[test]
    fn the_seed_of_a_stream_mixes_its_number_into_the_mixed_seed() {
        let stream = Stream::SensitivityReplicates;
        let mut state = mixed(7) ^ stream.number().wrapping_mul(0xD1B5_4A32_D192_ED03);
        assert_eq!(stream.seed(7), splitmix64(&mut state));
        assert_eq!(
            Stream::StartSeeds.replicate(7, 3),
            replicate_seed(Stream::StartSeeds.seed(7), 3)
        );
    }
}
