//! Column-pivoted Householder QR factorization, with an overflow-safe norm and helpers.

use crate::linear_algebra::Matrix;
use crate::scalar::Numeric;
use crate::utils::error_codes::CalcError;

/// Euclidean norm of `v`, computed so it neither overflows on large components nor
/// underflows on small ones.
///
/// Components are split into three magnitude bands. Small and large components are summed
/// against a running maximum in that band, so every squared term stays within range; only
/// the mid band is squared directly. This is the MINPACK `enorm` scheme.
pub(crate) fn enorm<T: Numeric>(v: &[T]) -> T {
    // Below `rdwarf`, squaring underflows; above `agiant`, summing the squares overflows.
    let rdwarf = T::MIN_POSITIVE.sqrt();
    let rgiant = T::MAX.sqrt();
    let agiant = rgiant / T::from_usize(v.len());

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
pub(crate) fn max<T: PartialOrd>(a: T, b: T) -> T {
    if b > a { b } else { a }
}

/// Returns the smaller of `a` and `b`. If the two do not compare (a NaN is involved),
/// returns `a`.
#[allow(dead_code)]
pub(crate) fn min<T: PartialOrd>(a: T, b: T) -> T {
    if b < a { b } else { a }
}

/// Column-pivoted Householder QR of an `M`-by-`N` matrix with `M >= N`.
///
/// Holds the factorization in packed form: the strict lower triangle of `qr` stores the
/// Householder vectors, the strict upper triangle stores the off-diagonal of `R`, and
/// `r_diag` holds the diagonal of `R`. `permutation` gives the pivot order, so
/// `A * P == Q * R`, where column `j` of `P` is column `permutation[j]` of the identity.
#[derive(Debug, Clone, Copy)]
#[must_use]
#[allow(dead_code)]
pub struct PivotedQr<const M: usize, const N: usize, T = f64> {
    /// Packed reflectors (below the diagonal) and off-diagonal `R` (above it).
    pub(crate) qr: Matrix<M, N, T>,
    /// Diagonal of `R`.
    pub(crate) r_diag: [T; N],
    /// Euclidean norms of the original columns of `A`, in original order.
    pub(crate) column_norms: [T; N],
    /// Pivot order: column `j` of `A * P` is column `permutation[j]` of `A`.
    pub(crate) permutation: [usize; N],
}

impl<const M: usize, const N: usize, T: Numeric> PivotedQr<M, N, T> {
    /// Factorizes `a` by column-pivoted Householder QR.
    ///
    /// Returns [`CalcError::Underdetermined`] if `M < N`. Never panics: a zero pivot column
    /// leaves the corresponding `r_diag` entry at zero rather than dividing by it, so a
    /// rank-deficient matrix factorizes without error (the deficiency surfaces in a solve).
    pub fn decompose(a: Matrix<M, N, T>) -> Result<Self, CalcError> {
        if M < N {
            return Err(CalcError::Underdetermined);
        }

        let mut qr = a;
        let mut r_diag = [T::ZERO; N];
        let mut column_norms = [T::ZERO; N];
        let mut reference_norm = [T::ZERO; N];
        let mut permutation = [0usize; N];

        // Initial column norms; `r_diag` doubles as the running partial norm until each
        // column is reduced, after which it holds the final `R` diagonal.
        for j in 0..N {
            let mut column = [T::ZERO; M];
            for i in 0..M {
                column[i] = qr[(i, j)];
            }
            let norm = enorm(&column);
            column_norms[j] = norm;
            r_diag[j] = norm;
            reference_norm[j] = norm;
            permutation[j] = j;
        }

        let epsmch = T::EPSILON;
        let p05 = T::from_f64(0.05);

        for j in 0..N {
            // Bring the column of largest remaining partial norm into position `j`.
            let mut kmax = j;
            for k in j..N {
                if r_diag[k] > r_diag[kmax] {
                    kmax = k;
                }
            }
            if kmax != j {
                for i in 0..M {
                    let tmp = qr[(i, j)];
                    qr[(i, j)] = qr[(i, kmax)];
                    qr[(i, kmax)] = tmp;
                }
                r_diag[kmax] = r_diag[j];
                reference_norm[kmax] = reference_norm[j];
                permutation.swap(j, kmax);
            }

            // Householder transformation zeroing column `j` below the diagonal.
            let mut column = [T::ZERO; M];
            for i in j..M {
                column[i] = qr[(i, j)];
            }
            let mut ajnorm = enorm(&column[j..]);
            if ajnorm == T::ZERO {
                r_diag[j] = -ajnorm;
                continue;
            }
            // Sign chosen so the pivot element is at least one, keeping the divisor below safe.
            if qr[(j, j)] < T::ZERO {
                ajnorm = -ajnorm;
            }
            for i in j..M {
                qr[(i, j)] /= ajnorm;
            }
            qr[(j, j)] += T::ONE;

            // Apply the transformation to the remaining columns and downdate their norms.
            for k in (j + 1)..N {
                let mut sum = T::ZERO;
                for i in j..M {
                    sum += qr[(i, j)] * qr[(i, k)];
                }
                let factor = sum / qr[(j, j)];
                for i in j..M {
                    let reflected = qr[(i, k)] - factor * qr[(i, j)];
                    qr[(i, k)] = reflected;
                }

                if r_diag[k] != T::ZERO {
                    let ratio = qr[(j, k)] / r_diag[k];
                    r_diag[k] *= max(T::ZERO, T::ONE - ratio * ratio).sqrt();
                    // `reference_norm[k]` is the column's original norm, nonzero here.
                    let relative = r_diag[k] / reference_norm[k];
                    if p05 * relative * relative <= epsmch {
                        // Recompute from the remaining rows to shed accumulated round-off.
                        let mut tail = [T::ZERO; M];
                        for i in (j + 1)..M {
                            tail[i] = qr[(i, k)];
                        }
                        r_diag[k] = enorm(&tail[(j + 1)..]);
                        reference_norm[k] = r_diag[k];
                    }
                }
            }

            r_diag[j] = -ajnorm;
        }

        Ok(PivotedQr {
            qr,
            r_diag,
            column_norms,
            permutation,
        })
    }
}
