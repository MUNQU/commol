//! The registry of seed streams: one number per stream, and one seed per
//! stream of a seed.

use commol_random::Stream;

const SEED: u64 = 11;

#[test]
fn every_stream_has_its_own_number() {
    let numbers: Vec<u64> = Stream::ALL.iter().map(|stream| stream.number()).collect();
    for (i, number) in numbers.iter().enumerate() {
        assert!(!numbers[..i].contains(number), "stream {number} twice");
    }
    let seeds: Vec<u64> = Stream::ALL.iter().map(|stream| stream.seed(SEED)).collect();
    for (i, seed) in seeds.iter().enumerate() {
        assert!(!seeds[..i].contains(seed), "{:?}", Stream::ALL[i]);
    }
    for stream in Stream::ALL {
        assert!(!stream.method().is_empty() && !stream.purpose().is_empty());
    }
}
