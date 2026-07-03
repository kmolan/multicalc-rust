//! Column-pivoted Householder QR factorization, with an overflow-safe norm and helpers.

use crate::linear_algebra::{Matrix, Vector};
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
    ///
    /// # Examples
    /// ```
    /// use multicalc::linear_algebra::{Matrix, PivotedQr, Vector};
    ///
    /// // Least-squares fit of y = a + b*t through (0, 1), (1, 3), (2, 5): a = 1, b = 2.
    /// let a = Matrix::<3, 2>::new([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]);
    /// let b = Vector::new([1.0, 3.0, 5.0]);
    /// let x = PivotedQr::decompose(a).unwrap().solve_least_squares(b).unwrap();
    /// assert!((x[0] - 1.0).abs() < 1e-12);
    /// assert!((x[1] - 2.0).abs() < 1e-12);
    /// ```
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

    /// The `N`-by-`N` upper-triangular factor `R`.
    pub fn r(&self) -> Matrix<N, N, T> {
        Matrix::from_fn(|row, col| {
            if row == col {
                self.r_diag[row]
            } else if col > row {
                self.qr[(row, col)]
            } else {
                T::ZERO
            }
        })
    }

    /// The `M`-by-`N` factor `Q`, formed by applying the stored reflectors to the identity.
    /// Its columns are orthonormal.
    pub fn q(&self) -> Matrix<M, N, T> {
        let mut q = Matrix::from_fn(|row, col| if row == col { T::ONE } else { T::ZERO });
        for col in 0..N {
            for j in (0..N).rev() {
                let pivot = self.qr[(j, j)];
                if pivot == T::ZERO {
                    continue;
                }
                let mut sum = T::ZERO;
                for i in j..M {
                    sum += self.qr[(i, j)] * q[(i, col)];
                }
                let factor = sum / pivot;
                for i in j..M {
                    q[(i, col)] -= factor * self.qr[(i, j)];
                }
            }
        }
        q
    }

    /// The pivot order: column `j` of `A * P` is column `permutation()[j]` of `A`.
    #[inline]
    #[must_use]
    pub fn permutation(&self) -> [usize; N] {
        self.permutation
    }

    /// Solves the least-squares problem `min ‖A x − b‖`, reusing this factorization. When `A`
    /// is square and full rank this is the exact solve of `A x = b`.
    ///
    /// Returns [`CalcError::SingularMatrix`] if `A` is rank-deficient — a diagonal entry of `R`
    /// at or below `EPSILON * max(M, N)` times the largest — rather than dividing by a tiny pivot.
    ///
    /// ```
    /// use multicalc::linear_algebra::{Matrix, PivotedQr, Vector};
    /// // A x = b has the exact solution x = [1, 1, 1].
    /// let a = Matrix::<3, 3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]);
    /// let b = Vector::new([6.0, 15.0, 25.0]);
    /// let x = PivotedQr::decompose(a).unwrap().solve_least_squares(b).unwrap();
    /// assert!((x[0] - 1.0).abs() < 1e-12);
    /// assert!((x[1] - 1.0).abs() < 1e-12);
    /// assert!((x[2] - 1.0).abs() < 1e-12);
    /// ```
    pub fn solve_least_squares(&self, b: Vector<M, T>) -> Result<Vector<N, T>, CalcError> {
        // Apply the reflectors to b, leaving Qᵀb in the first N entries.
        let mut qtb = b;
        for j in 0..N {
            let pivot = self.qr[(j, j)];
            if pivot == T::ZERO {
                continue;
            }
            let mut sum = T::ZERO;
            for i in j..M {
                sum += self.qr[(i, j)] * qtb[i];
            }
            let factor = sum / pivot;
            for i in j..M {
                qtb[i] -= factor * self.qr[(i, j)];
            }
        }

        // A diagonal entry at or below this fraction of the largest signals rank deficiency.
        let threshold = if N == 0 {
            T::ZERO
        } else {
            T::EPSILON * T::from_usize(M.max(N)) * self.r_diag[0].abs()
        };

        // Back-substitute R y = Qᵀb over the first N rows.
        let mut y = [T::ZERO; N];
        for row in (0..N).rev() {
            if self.r_diag[row].abs() <= threshold {
                return Err(CalcError::SingularMatrix);
            }
            let mut acc = qtb[row];
            for (col, &y_value) in y.iter().enumerate().skip(row + 1) {
                acc -= self.qr[(row, col)] * y_value;
            }
            y[row] = acc / self.r_diag[row];
        }

        // Undo the column permutation: x = P y.
        let mut x = [T::ZERO; N];
        for (j, &target) in self.permutation.iter().enumerate() {
            x[target] = y[j];
        }
        Ok(Vector::new(x))
    }

    /// Turns this factorization into a reusable damped least-squares problem for `b`,
    /// precomputing `Qᵀb` so a whole family of damped systems shares one factorization.
    pub fn into_damped(self, b: Vector<M, T>) -> DampedLeastSquares<N, T> {
        // Apply the reflectors to b, leaving Qᵀb in the first N entries.
        let mut transformed = b;
        for j in 0..N {
            let pivot = self.qr[(j, j)];
            if pivot == T::ZERO {
                continue;
            }
            let mut sum = T::ZERO;
            for i in j..M {
                sum += self.qr[(i, j)] * transformed[i];
            }
            let factor = sum / pivot;
            for i in j..M {
                transformed[i] -= factor * self.qr[(i, j)];
            }
        }
        let mut qt_b = [T::ZERO; N];
        for (dst, &src) in qt_b.iter_mut().zip(transformed.as_array().iter().take(N)) {
            *dst = src;
        }

        DampedLeastSquares {
            r: self.r(),
            qt_b,
            permutation: self.permutation,
            column_norms: self.column_norms,
        }
    }
}

/// A reusable damped least-squares problem built from one QR factorization of `A`.
///
/// Given `A = Q R P` and `b`, it solves `(AᵀA + D²) x = Aᵀb` for any diagonal `D` without
/// refactorizing, using the MINPACK `qrsolv` scheme (Givens rotations that eliminate the `D`
/// rows into `R`). This is the linear subproblem the Levenberg-Marquardt trust region solves.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct DampedLeastSquares<const N: usize, T = f64> {
    /// Upper-triangular factor `R`.
    pub(crate) r: Matrix<N, N, T>,
    /// `Qᵀb`, the right-hand side in factored coordinates.
    pub(crate) qt_b: [T; N],
    /// Pivot order carried over from the factorization.
    pub(crate) permutation: [usize; N],
    /// Euclidean norms of the original columns of `A`, in original order.
    pub(crate) column_norms: [T; N],
}

impl<const N: usize, T: Numeric> DampedLeastSquares<N, T> {
    /// Solves `(AᵀA + D²) x = Aᵀb` for the diagonal `D` given by `diag`, returning the solution
    /// and the Cholesky-like factor `S` of `AᵀA + D²`.
    pub fn solve_with_diagonal(&self, diag: &[T; N]) -> (Vector<N, T>, CholeskyFactor<N, T>) {
        // Working matrix: upper triangle is R, lower triangle mirrors it as scratch.
        let mut s = self.r;
        for j in 0..N {
            for i in (j + 1)..N {
                s[(i, j)] = s[(j, i)];
            }
        }

        let mut saved_diag = [T::ZERO; N];
        let mut wa = [T::ZERO; N];
        for j in 0..N {
            saved_diag[j] = s[(j, j)];
            wa[j] = self.qt_b[j];
        }

        let mut s_diag = [T::ZERO; N];
        let quarter = T::from_f64(0.25);

        for j in 0..N {
            let l = self.permutation[j];
            if diag[l] != T::ZERO {
                for entry in s_diag.iter_mut().skip(j) {
                    *entry = T::ZERO;
                }
                s_diag[j] = diag[l];

                // Eliminate the diagonal row of D with Givens rotations, carrying the extra
                // right-hand-side element (initially zero) alongside.
                let mut qtbpj = T::ZERO;
                for k in j..N {
                    if s_diag[k] == T::ZERO {
                        continue;
                    }
                    let (sin, cos) = if s[(k, k)].abs() >= s_diag[k].abs() {
                        let tan = s_diag[k] / s[(k, k)];
                        let cos = T::HALF / (quarter + quarter * tan * tan).sqrt();
                        (cos * tan, cos)
                    } else {
                        let cotan = s[(k, k)] / s_diag[k];
                        let sin = T::HALF / (quarter + quarter * cotan * cotan).sqrt();
                        (sin, sin * cotan)
                    };

                    s[(k, k)] = cos * s[(k, k)] + sin * s_diag[k];
                    let temp = cos * wa[k] + sin * qtbpj;
                    qtbpj = -sin * wa[k] + cos * qtbpj;
                    wa[k] = temp;

                    for i in (k + 1)..N {
                        let rotated = cos * s[(i, k)] + sin * s_diag[i];
                        s_diag[i] = -sin * s[(i, k)] + cos * s_diag[i];
                        s[(i, k)] = rotated;
                    }
                }
            }
            // Store the S diagonal and restore R's.
            s_diag[j] = s[(j, j)];
            s[(j, j)] = saved_diag[j];
        }

        // Solve the triangular system for the permuted solution, zeroing any singular tail.
        let mut nsing = N;
        for j in 0..N {
            if s_diag[j] == T::ZERO && nsing == N {
                nsing = j;
            }
            if nsing < N {
                wa[j] = T::ZERO;
            }
        }
        for k in 0..nsing {
            let j = nsing - 1 - k;
            let mut sum = T::ZERO;
            for i in (j + 1)..nsing {
                sum += s[(i, j)] * wa[i];
            }
            wa[j] = (wa[j] - sum) / s_diag[j];
        }

        // Permute the solution back to original coordinates.
        let mut x = [T::ZERO; N];
        for (j, &target) in self.permutation.iter().enumerate() {
            x[target] = wa[j];
        }

        (Vector::new(x), CholeskyFactor { s, s_diag })
    }

    /// Solves the undamped problem `AᵀA x = Aᵀb` (the Gauss-Newton direction).
    pub fn solve_with_zero_diagonal(&self) -> (Vector<N, T>, CholeskyFactor<N, T>) {
        self.solve_with_diagonal(&[T::ZERO; N])
    }

    /// The largest scaled gradient component `|Aᵀb|ⱼ / (b_norm · ‖columnⱼ‖)`, used by the
    /// gradient convergence test. Returns zero when `b_norm` is zero.
    #[must_use]
    pub fn max_a_t_b_scaled(&self, b_norm: T) -> T {
        if b_norm == T::ZERO {
            return T::ZERO;
        }
        let mut result = T::ZERO;
        for j in 0..N {
            let l = self.permutation[j];
            if self.column_norms[l] != T::ZERO {
                let mut sum = T::ZERO;
                for (i, &qi) in self.qt_b.iter().enumerate().take(j + 1) {
                    sum += self.r[(i, j)] * (qi / b_norm);
                }
                result = max(result, (sum / self.column_norms[l]).abs());
            }
        }
        result
    }

    /// The norm `‖A x‖`, computed as `‖R P x‖` since `Q` has orthonormal columns.
    #[must_use]
    pub fn a_x_norm(&self, x: &Vector<N, T>) -> T {
        let mut w = [T::ZERO; N];
        for j in 0..N {
            let xl = x[self.permutation[j]];
            for (i, slot) in w.iter_mut().enumerate().take(j + 1) {
                *slot += self.r[(i, j)] * xl;
            }
        }
        enorm(&w)
    }

    /// Whether `R` has full rank (no diagonal entry negligible against the largest).
    #[must_use]
    pub fn is_non_singular(&self) -> bool {
        if N == 0 {
            return true;
        }
        let threshold = T::EPSILON * T::from_usize(N) * self.r[(0, 0)].abs();
        (0..N).all(|j| self.r[(j, j)].abs() > threshold)
    }
}

/// The Cholesky-like factor `S` (upper triangular) of `AᵀA + D²` from a damped solve.
///
/// Stored in the working matrix's strict lower triangle (as `Sᵀ`) plus a separate diagonal.
/// Its [`solve`](CholeskyFactor::solve) forward-substitutes `Sᵀ`, which the trust-region
/// parameter search uses for its Newton correction.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct CholeskyFactor<const N: usize, T = f64> {
    /// Working matrix whose strict lower triangle holds `Sᵀ`.
    pub(crate) s: Matrix<N, N, T>,
    /// Diagonal of `S`.
    pub(crate) s_diag: [T; N],
}

impl<const N: usize, T: Numeric> CholeskyFactor<N, T> {
    /// Forward-solves `Sᵀ w = rhs`, with `rhs` and the result in the factor's internal order.
    #[must_use]
    pub fn solve(&self, mut rhs: [T; N]) -> [T; N] {
        for j in 0..N {
            if self.s_diag[j] != T::ZERO {
                rhs[j] /= self.s_diag[j];
            } else {
                rhs[j] = T::ZERO;
            }
            let temp = rhs[j];
            for (i, slot) in rhs.iter_mut().enumerate().skip(j + 1) {
                *slot -= self.s[(i, j)] * temp;
            }
        }
        rhs
    }
}
