//! The one place commol draws a seed that its caller does not set.

/// `seed`, or a seed drawn from the entropy of the operating system when it
/// is None.
///
/// A method that accepts a drawn seed reads its seed here and reports the
/// seed it ran with, so its result can be repeated. A method that needs the
/// seed of its caller, such as an analysis of a stochastic engine, rejects
/// None before it calls this.
///
/// # Panics
///
/// When the operating system gives no random bytes.
pub fn seed_or_drawn(seed: Option<u64>) -> u64 {
    seed.unwrap_or_else(|| getrandom::u64().expect("the operating system gives random bytes"))
}
