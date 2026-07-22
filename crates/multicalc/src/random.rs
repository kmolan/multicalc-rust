//! Credits: [`Pcg32`] follows the permuted congruential generator designed and published by
//! Melissa O'Neill, with the constants from her reference implementation (see
//! <https://www.pcg-random.org/>). The normal draw uses the polar form of the Box–Muller transform.
//! Thanks to both for putting the method and code in the open.
//!
//! Small, fast, seedable pseudo-random numbers. Deterministic and not for cryptography; the same seed
//! reproduces the same sequence, so a run repeats exactly. [`RandomSource`] is the trait every generator
//! implements, with uniform and normal draws built on top of a raw 32-bit word; [`Pcg32`] is the built-in
//! generator.

/// A source of random 32-bit words, and the uniform and normal draws built from them.
pub trait RandomSource {
    /// The next raw 32-bit word.
    fn next_u32(&mut self) -> u32;

    /// The next 64-bit word, from two 32-bit words (high word first).
    fn next_u64(&mut self) -> u64 {
        ((self.next_u32() as u64) << 32) | (self.next_u32() as u64)
    }

    /// A uniform draw in the half-open range 0.0 up to 1.0, with 53 bits of precision.
    fn next_unit_f64(&mut self) -> f64 {
        // Top 53 bits scaled into [0, 1); the low 11 bits are dropped so the result never reaches 1.
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// One draw from the standard normal distribution (mean 0, standard deviation 1).
    ///
    /// Uses the polar pair method and returns one of the pair, consuming two uniform draws. The
    /// first draw is nudged off zero so the logarithm stays finite.
    fn standard_normal(&mut self) -> f64 {
        let mut first = self.next_unit_f64();
        if first <= f64::MIN_POSITIVE {
            first = f64::MIN_POSITIVE;
        }
        let second = self.next_unit_f64();
        libm::sqrt(-2.0 * libm::log(first)) * libm::cos(core::f64::consts::TAU * second)
    }
}

/// A small, fast, seedable pseudo-random generator (PCG-XSH-RR, 32-bit output).
///
/// Deterministic: the same seed and stream reproduce the same sequence, so a run repeats exactly.
/// Not for cryptography.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pcg32 {
    state: u64,
    increment: u64,
}

impl Pcg32 {
    /// A generator seeded on the default stream.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self::with_stream(seed, DEFAULT_STREAM)
    }

    /// A generator on a chosen stream, so independent filters draw independent sequences from the
    /// same seed.
    #[must_use]
    pub fn with_stream(seed: u64, stream: u64) -> Self {
        let mut generator = Pcg32 {
            state: 0,
            increment: (stream << 1) | 1,
        };
        let _ = generator.next_u32();
        generator.state = generator.state.wrapping_add(seed);
        let _ = generator.next_u32();
        generator
    }
}

impl RandomSource for Pcg32 {
    fn next_u32(&mut self) -> u32 {
        // Advance the 64-bit state one step, then scramble the value it held into the output word.
        // The state moves on with a fixed multiply-and-add; the output is built from the old state
        // so it is returned before the step, not after.
        let previous = self.state;
        self.state = previous
            .wrapping_mul(PCG_MULTIPLIER)
            .wrapping_add(self.increment);
        // Fold the high bits down into the low half and keep 32 of them, then rotate that half by an
        // amount taken from the very top bits. The data-driven rotate is what hides the state's
        // regular stepping, so nearby seeds do not give visibly related sequences.
        let xorshifted = (((previous >> 18) ^ previous) >> 27) as u32;
        let rotation = (previous >> 59) as u32;
        xorshifted.rotate_right(rotation)
    }
}

/// The fixed multiplier that steps the 64-bit state each draw. This exact value is the one from the
/// reference implementation; it is chosen so the state cycles through every 64-bit value before
/// repeating.
const PCG_MULTIPLIER: u64 = 6364136223846793005;

/// The stream used when a caller does not pick one. Two generators on different streams draw
/// unrelated sequences from the same seed, so this is just the default choice of stream; any odd
/// value works equally well.
const DEFAULT_STREAM: u64 = 0xda3e_39cb_94b9_5bdb;
