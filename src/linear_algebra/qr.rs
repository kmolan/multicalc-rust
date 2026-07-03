//! Overflow- and underflow-safe Euclidean norm and small comparison helpers.

use crate::scalar::Numeric;

/// Euclidean norm of `v`, computed so it neither overflows on large components nor
/// underflows on small ones.
///
/// Components are split into three magnitude bands. Small and large components are summed
/// against a running maximum in that band, so every squared term stays within range; only
/// the mid band is squared directly. This is the MINPACK `enorm` scheme.
#[allow(dead_code)]
pub(crate) fn enorm<const K: usize, T: Numeric>(v: &[T; K]) -> T {
    // Below `rdwarf`, squaring underflows; above `agiant`, summing `K` squares overflows.
    let rdwarf = T::MIN_POSITIVE.sqrt();
    let rgiant = T::MAX.sqrt();
    let agiant = rgiant / T::from_usize(K);

    let mut small_sum = T::ZERO;
    let mut mid_sum = T::ZERO;
    let mut large_sum = T::ZERO;
    let mut small_max = T::ZERO;
    let mut large_max = T::ZERO;

    for &value in v {
        let a = value.abs();

        if a > rdwarf && a < agiant {
            mid_sum += a * a;
        } else if a > rdwarf {
            // Large band: rescale against the running large maximum.
            if a > large_max {
                let ratio = large_max / a;
                large_sum = T::ONE + large_sum * ratio * ratio;
                large_max = a;
            } else {
                let ratio = a / large_max;
                large_sum += ratio * ratio;
            }
        } else if a != T::ZERO {
            // Small band: rescale against the running small maximum.
            if a > small_max {
                let ratio = small_max / a;
                small_sum = T::ONE + small_sum * ratio * ratio;
                small_max = a;
            } else {
                let ratio = a / small_max;
                small_sum += ratio * ratio;
            }
        }
    }

    if large_sum != T::ZERO {
        large_max * (large_sum + (mid_sum / large_max) / large_max).sqrt()
    } else if mid_sum != T::ZERO {
        if mid_sum >= small_max {
            (mid_sum * (T::ONE + (small_max / mid_sum) * (small_max * small_sum))).sqrt()
        } else {
            (small_max * ((mid_sum / small_max) + (small_max * small_sum))).sqrt()
        }
    } else {
        small_max * small_sum.sqrt()
    }
}

/// Returns the larger of `a` and `b`. If the two do not compare (a NaN is involved),
/// returns `a`.
#[allow(dead_code)]
pub(crate) fn max<T: PartialOrd>(a: T, b: T) -> T {
    if b > a { b } else { a }
}

/// Returns the smaller of `a` and `b`. If the two do not compare (a NaN is involved),
/// returns `a`.
#[allow(dead_code)]
pub(crate) fn min<T: PartialOrd>(a: T, b: T) -> T {
    if b < a { b } else { a }
}
