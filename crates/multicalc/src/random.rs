//! Credits: [`Pcg32`] follows the permuted congruential generator designed and published by
//! Melissa O'Neill, with the constants from her reference implementation (see
//! <https://www.pcg-random.org/>). The normal draw uses the polar form of the Box–Muller transform.
//! Thanks to both for putting the method and code in the open.
//!
//! Small, fast, seedable pseudo-random numbers. Deterministic and not for cryptography; the same seed
//! reproduces the same sequence, so a run repeats exactly. [`RandomSource`] is the trait every generator
//! implements, with uniform and normal draws built on top of a raw 32-bit word; [`Pcg32`] is the built-in
//! generator.

use crate::Numeric;

/// A source of random 32-bit words, and the uniform and normal draws built from them.
pub trait RandomSource<T: RandomScalar> {
    /// The next raw 32-bit word.
    #[must_use]
    fn next_u32(&mut self) -> u32;

    /// The next 64-bit word, from two 32-bit words (high word first).
    #[must_use]
    fn next_u64(&mut self) -> u64 {
        ((self.next_u32() as u64) << 32) | (self.next_u32() as u64)
    }

    /// A uniform draw in the half-open range 0.0 up to 1.0
    #[must_use]
    fn next_unit(&mut self) -> T {
        T::next_unit(self)
    }

    /// One draw from the standard normal distribution (mean 0, standard deviation 1).
    ///
    /// Uses the Marsaglia polar method and returns one of the pair, consuming two uniform draws.
    #[must_use]
    fn standard_normal(&mut self) -> T {
        if let Some(cached) = self.get_cache() {
            return cached;
        }

        loop {
            let x = T::TWO * self.next_unit() - T::ONE;
            let y = T::TWO * self.next_unit() - T::ONE;
            let s = x.powi(2) + y.powi(2);

            if s > T::ZERO && s < T::ONE {
                let scale = (-T::TWO * s.ln() / s).sqrt();
                self.set_cache(y * scale);
                return x * scale;
            }
        }
    }

    fn get_cache(&mut self) -> Option<T>;

    fn set_cache(&mut self, value: T);
}

pub trait RandomScalar: Numeric {
    /// A uniform draw in the half-open range 0.0 up to 1.0.
    #[must_use]
    fn next_unit<R: RandomSource<Self> + ?Sized>(source: &mut R) -> Self;
}

impl RandomScalar for f64 {
    /// A uniform draw in the half-open range 0.0 up to 1.0, with 53 bits of precision.
    #[inline]
    fn next_unit<R: RandomSource<f64> + ?Sized>(source: &mut R) -> Self {
        // Top 53 bits scaled into [0, 1); the low 11 bits are dropped so the result never reaches 1.
        (source.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

impl RandomScalar for f32 {
    /// A uniform draw in the half-open range 0.0 up to 1.0, with 24 bits of precision.
    #[inline]
    fn next_unit<R: RandomSource<f32> + ?Sized>(source: &mut R) -> Self {
        (source.next_u32() >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
    }
}

/// A small, fast, seedable pseudo-random generator (PCG-XSH-RR, 32-bit output).
///
/// Deterministic: the same seed and stream reproduce the same sequence, so a run repeats exactly.
/// Not for cryptography.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pcg32<T: RandomScalar> {
    state: u64,
    increment: u64,
    cache: Option<T>,
}

impl<T: RandomScalar> Pcg32<T> {
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
            cache: None,
        };
        let _ = generator.next_u32();
        generator.state = generator.state.wrapping_add(seed);
        let _ = generator.next_u32();
        generator
    }
}

impl<T: RandomScalar> RandomSource<T> for Pcg32<T> {
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

    fn set_cache(&mut self, value: T) {
        self.cache = Some(value);
    }

    fn get_cache(&mut self) -> Option<T> {
        self.cache.take()
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
