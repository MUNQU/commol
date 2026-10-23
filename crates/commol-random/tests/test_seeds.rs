//! The seeds of replicates and streams: two seeds give unrelated seeds for
//! every pair of indices.

use commol_random::{replicate_seed, stream_seed};

fn all_differ(mut seeds: Vec<u64>) -> bool {
    let count = seeds.len();
    seeds.sort_unstable();
    seeds.dedup();
    seeds.len() == count
}

#[test]
fn neighbouring_seeds_share_no_replicate_seed() {
    let replicates = |seed: u64| (0..20).map(move |index| replicate_seed(seed, index));
    let streams = |seed: u64| (1..20).map(move |stream| stream_seed(seed, stream));
    assert!(all_differ((40..60).flat_map(replicates).collect()));
    assert!(all_differ(
        (40..60)
            .flat_map(replicates)
            .chain((40..60).flat_map(streams))
            .collect()
    ));
}

/// The first seeds of seed 1, from a SplitMix64 written apart: a change of
/// the rule changes every run of a seed.
#[test]
fn the_first_seeds_are_fixed() {
    let replicates: Vec<u64> = (0..3).map(|index| replicate_seed(1, index)).collect();
    let streams: Vec<u64> = (1..4).map(|stream| stream_seed(1, stream)).collect();
    assert_eq!(
        replicates,
        [
            6_791_897_765_849_424_158,
            9_716_232_063_330_790_915,
            13_608_149_317_741_381_227
        ]
    );
    assert_eq!(
        streams,
        [
            2_569_293_373_224_866_520,
            3_279_294_537_695_293_187,
            3_791_248_478_382_738_589
        ]
    );
}
