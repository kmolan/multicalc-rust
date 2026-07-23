//! Column-pivoted Householder QR factorization, with an overflow-safe norm and helpers.
//!
//! The factorization, damped solve, and norm port MINPACK's `qrfac`, `qrsolv`, and `enorm` (Moré,
//! Garbow, Hillstrom; public domain, netlib) — a clean-room, fixed-size `no_std` reimplementation
//! on this crate's own [`Vector`] and [`Matrix`] types.

use crate::error::LinalgError;
use crate::linear_algebra::{Matrix, Vector};
use crate::scalar::Numeric;

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
    /// Returns [`LinalgError::Underdetermined`] if `M < N`. Never panics: a zero pivot column
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
    /// assert!((x.as_array()[0] - 1.0).abs() < 1e-12);
    /// assert!((x.as_array()[1] - 2.0).abs() < 1e-12);
    /// ```
    pub fn decompose(a: Matrix<M, N, T>) -> Result<Self, LinalgError> {
        if M < N {
            return Err(LinalgError::Underdetermined);
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
            for (i, slot) in column.iter_mut().enumerate() {
                *slot = qr.get(i, j).copied().unwrap_or(T::ZERO);
            }
            let norm = enorm(&column);
            if let Some(slot) = column_norms.get_mut(j) {
                *slot = norm;
            }
            if let Some(slot) = r_diag.get_mut(j) {
                *slot = norm;
            }
            if let Some(slot) = reference_norm.get_mut(j) {
                *slot = norm;
            }
            if let Some(slot) = permutation.get_mut(j) {
                *slot = j;
            }
        }

        let epsmch = T::EPSILON;
        let p05 = T::from_f64(0.05);

        for j in 0..N {
            // Bring the column of largest remaining partial norm into position `j`.
            let mut kmax = j;
            for k in j..N {
                let Some(&rk) = r_diag.get(k) else { continue };
                let Some(&rkmax) = r_diag.get(kmax) else {
                    continue;
                };
                if rk > rkmax {
                    kmax = k;
                }
            }
            if kmax != j {
                for row in qr.as_mut_slice_rows() {
                    row.swap(j, kmax)
                }
                let rj = r_diag.get(j).copied();
                if let (Some(value), Some(slot)) = (rj, r_diag.get_mut(kmax)) {
                    *slot = value;
                }
                let nj = reference_norm.get(j).copied();
                if let (Some(value), Some(slot)) = (nj, reference_norm.get_mut(kmax)) {
                    *slot = value;
                }
                permutation.swap(j, kmax);
            }

            // Householder transformation zeroing column `j` below the diagonal.
            let mut column = [T::ZERO; M];
            for i in j..M {
                if let Some(slot) = column.get_mut(i) {
                    *slot = qr.get(i, j).copied().unwrap_or(T::ZERO);
                }
            }
            let mut ajnorm = column.get(j..).map(enorm).unwrap_or(T::ZERO);
            if ajnorm == T::ZERO {
                if let Some(slot) = r_diag.get_mut(j) {
                    *slot = -ajnorm;
                }
                continue;
            }
            // Sign chosen so the pivot element is at least one, keeping the divisor below safe.
            if qr.get(j, j).copied().unwrap_or(T::ZERO) < T::ZERO {
                ajnorm = -ajnorm;
            }
            for row in qr.as_mut_slice_rows().iter_mut().skip(j) {
                if let Some(slot) = row.get_mut(j) {
                    *slot /= ajnorm;
                }
            }
            if let Some(slot) = qr.get_mut(j, j) {
                *slot += T::ONE;
            }

            // Apply the transformation to the remaining columns and downdate their norms.
            for k in (j + 1)..N {
                let mut sum = T::ZERO;
                for i in j..M {
                    let a = qr.get(i, j).copied().unwrap_or(T::ZERO);
                    let b = qr.get(i, k).copied().unwrap_or(T::ZERO);
                    sum += a * b;
                }
                let pivot = qr.get(j, j).copied().unwrap_or(T::ZERO);
                let factor = sum / pivot;
                for i in j..M {
                    let current = qr.get(i, k).copied().unwrap_or(T::ZERO);
                    let reflector = qr.get(i, j).copied().unwrap_or(T::ZERO);
                    if let Some(slot) = qr.get_mut(i, k) {
                        *slot = current - factor * reflector;
                    }
                }

                if r_diag.get(k).copied() != Some(T::ZERO) {
                    let ratio = qr.get(j, k).copied().unwrap_or(T::ZERO)
                        / r_diag.get(k).copied().unwrap_or(T::ZERO);
                    if let Some(slot) = r_diag.get_mut(k) {
                        *slot *= max(T::ZERO, T::ONE - ratio * ratio).sqrt();
                    }
                    let rk = r_diag.get(k).copied().unwrap_or(T::ZERO);
                    let refk = reference_norm.get(k).copied().unwrap_or(T::ZERO);
                    let relative = rk / refk;
                    if p05 * relative * relative <= epsmch {
                        let mut tail = [T::ZERO; M];
                        for i in (j + 1)..M {
                            if let Some(slot) = tail.get_mut(i) {
                                *slot = qr.get(i, k).copied().unwrap_or(T::ZERO);
                            }
                        }
                        let recomputed = tail.get((j + 1)..).map(enorm).unwrap_or(T::ZERO);
                        if let Some(slot) = r_diag.get_mut(k) {
                            *slot = recomputed;
                        }
                        if let Some(slot) = reference_norm.get_mut(k) {
                            *slot = recomputed;
                        }
                    }
                }
            }

            if let Some(slot) = r_diag.get_mut(j) {
                *slot = -ajnorm;
            }
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
                self.r_diag.get(row).copied().unwrap_or(T::ZERO)
            } else if col > row {
                self.qr.get(row, col).copied().unwrap_or(T::ZERO)
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
                let pivot = self.qr.get(j, j).copied().unwrap_or(T::ZERO);
                if pivot == T::ZERO {
                    continue;
                }
                let mut sum = T::ZERO;
                for i in j..M {
                    let reflector = self.qr.get(i, j).copied().unwrap_or(T::ZERO);
                    let qi = q.get(i, col).copied().unwrap_or(T::ZERO);
                    sum += reflector * qi;
                }
                let factor = sum / pivot;
                for i in j..M {
                    let reflector = self.qr.get(i, j).copied().unwrap_or(T::ZERO);
                    if let Some(slot) = q.get_mut(i, col) {
                        *slot -= factor * reflector;
                    }
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
    /// Returns [`LinalgError::Singular`] if `A` is rank-deficient — a diagonal entry of `R`
    /// at or below `EPSILON * max(M, N)` times the largest — rather than dividing by a tiny pivot.
    ///
    /// ```
    /// use multicalc::linear_algebra::{Matrix, PivotedQr, Vector};
    /// // A x = b has the exact solution x = [1, 1, 1].
    /// let a = Matrix::<3, 3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]);
    /// let b = Vector::new([6.0, 15.0, 25.0]);
    /// let x = PivotedQr::decompose(a).unwrap().solve_least_squares(b).unwrap();
    /// assert!((x.as_array()[0] - 1.0).abs() < 1e-12);
    /// assert!((x.as_array()[1] - 1.0).abs() < 1e-12);
    /// assert!((x.as_array()[2] - 1.0).abs() < 1e-12);
    /// ```
    pub fn solve_least_squares(&self, b: Vector<M, T>) -> Result<Vector<N, T>, LinalgError> {
        // Apply the reflectors to b, leaving Qᵀb in the first N entries.
        let mut qtb = b;
        for j in 0..N {
            let pivot = self.qr.get(j, j).copied().unwrap_or(T::ZERO);
            if pivot == T::ZERO {
                continue;
            }
            let mut sum = T::ZERO;
            for i in j..M {
                let reflector = self.qr.get(i, j).copied().unwrap_or(T::ZERO);
                sum += reflector * qtb.get(i).copied().unwrap_or(T::ZERO);
            }
            let factor = sum / pivot;
            for i in j..M {
                let reflector = self.qr.get(i, j).copied().unwrap_or(T::ZERO);
                if let Some(slot) = qtb.get_mut(i) {
                    *slot -= factor * reflector;
                }
            }
        }

        // A diagonal entry at or below this fraction of the largest signals rank deficiency.
        let threshold = if N == 0 {
            T::ZERO
        } else {
            T::EPSILON
                * T::from_usize(M.max(N))
                * self.r_diag.first().copied().unwrap_or(T::ZERO).abs()
        };

        // Back-substitute R y = Qᵀb over the first N rows.
        let mut y = [T::ZERO; N];
        for row in (0..N).rev() {
            let diag = self.r_diag.get(row).copied().unwrap_or(T::ZERO);
            if diag.abs() <= threshold {
                return Err(LinalgError::Singular);
            }
            let mut acc = qtb.get(row).copied().unwrap_or(T::ZERO);
            for (col, &y_value) in y.iter().enumerate().skip(row + 1) {
                acc -= self.qr.get(row, col).copied().unwrap_or(T::ZERO) * y_value;
            }
            if let Some(slot) = y.get_mut(row) {
                *slot = acc / diag;
            }
        }

        // Undo the column permutation: x = P y.
        let mut x = [T::ZERO; N];
        for (j, &target) in self.permutation.iter().enumerate() {
            if let (Some(slot), Some(&yj)) = (x.get_mut(target), y.get(j)) {
                *slot = yj;
            }
        }
        Ok(Vector::new(x))
    }

    /// Turns this factorization into a reusable damped least-squares problem for `b`,
    /// precomputing `Qᵀb` so a whole family of damped systems shares one factorization.
    pub fn into_damped(self, b: Vector<M, T>) -> DampedLeastSquares<N, T> {
        // Apply the reflectors to b, leaving Qᵀb in the first N entries.
        let mut transformed = b;
        for j in 0..N {
            let pivot = self.qr.get(j, j).copied().unwrap_or(T::ZERO);
            if pivot == T::ZERO {
                continue;
            }
            let mut sum = T::ZERO;
            for i in j..M {
                let reflector = self.qr.get(i, j).copied().unwrap_or(T::ZERO);
                sum += reflector * transformed.get(i).copied().unwrap_or(T::ZERO);
            }
            let factor = sum / pivot;
            for i in j..M {
                let reflector = self.qr.get(i, j).copied().unwrap_or(T::ZERO);
                if let Some(slot) = transformed.get_mut(i) {
                    *slot -= factor * reflector;
                }
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
                let value = s.get(j, i).copied().unwrap_or(T::ZERO);
                if let Some(slot) = s.get_mut(i, j) {
                    *slot = value;
                }
            }
        }

        let mut saved_diag = [T::ZERO; N];
        let mut wa = [T::ZERO; N];
        for j in 0..N {
            if let Some(slot) = saved_diag.get_mut(j) {
                *slot = s.get(j, j).copied().unwrap_or(T::ZERO);
            }
            if let (Some(slot), Some(&qj)) = (wa.get_mut(j), self.qt_b.get(j)) {
                *slot = qj;
            }
        }

        let mut s_diag = [T::ZERO; N];
        let quarter = T::from_f64(0.25);

        for j in 0..N {
            let Some(&l) = self.permutation.get(j) else {
                continue;
            };
            if diag.get(l).copied() != Some(T::ZERO) {
                for entry in s_diag.iter_mut().skip(j) {
                    *entry = T::ZERO;
                }
                if let (Some(slot), Some(&dl)) = (s_diag.get_mut(j), diag.get(l)) {
                    *slot = dl;
                }

                // Eliminate the diagonal row of D with Givens rotations, carrying the extra
                // right-hand-side element (initially zero) alongside.
                let mut qtbpj = T::ZERO;
                for k in j..N {
                    if s_diag.get(k).copied() == Some(T::ZERO) {
                        continue;
                    }
                    let skk = s.get(k, k).copied().unwrap_or(T::ZERO);
                    let sk = s_diag.get(k).copied().unwrap_or(T::ZERO);
                    let (sin, cos) = if skk.abs() >= sk.abs() {
                        let tan = sk / skk;
                        let cos = T::HALF / (quarter + quarter * tan * tan).sqrt();
                        (cos * tan, cos)
                    } else {
                        let cotan = skk / sk;
                        let sin = T::HALF / (quarter + quarter * cotan * cotan).sqrt();
                        (sin, sin * cotan)
                    };

                    if let Some(slot) = s.get_mut(k, k) {
                        *slot = cos * skk + sin * sk;
                    }
                    let wak = wa.get(k).copied().unwrap_or(T::ZERO);
                    let temp = cos * wak + sin * qtbpj;
                    qtbpj = -sin * wak + cos * qtbpj;
                    if let Some(slot) = wa.get_mut(k) {
                        *slot = temp;
                    }

                    for i in (k + 1)..N {
                        let sik = s.get(i, k).copied().unwrap_or(T::ZERO);
                        let si = s_diag.get(i).copied().unwrap_or(T::ZERO);
                        let rotated = cos * sik + sin * si;
                        if let Some(slot) = s_diag.get_mut(i) {
                            *slot = -sin * sik + cos * si;
                        }
                        if let Some(slot) = s.get_mut(i, k) {
                            *slot = rotated;
                        }
                    }
                }
            }
            let sjj = s.get(j, j).copied().unwrap_or(T::ZERO);
            let saved = saved_diag.get(j).copied().unwrap_or(T::ZERO);
            if let Some(slot) = s_diag.get_mut(j) {
                *slot = sjj;
            }
            if let Some(slot) = s.get_mut(j, j) {
                *slot = saved;
            }
        }

        // Solve the triangular system for the permuted solution, zeroing any singular tail.
        let mut nsing = N;
        for j in 0..N {
            if s_diag.get(j).copied() == Some(T::ZERO) && nsing == N {
                nsing = j;
            }
            if nsing < N {
                if let Some(slot) = wa.get_mut(j) {
                    *slot = T::ZERO;
                }
            }
        }
        for k in 0..nsing {
            let j = nsing - 1 - k;
            let mut sum = T::ZERO;
            for i in (j + 1)..nsing {
                sum +=
                    s.get(i, j).copied().unwrap_or(T::ZERO) * wa.get(i).copied().unwrap_or(T::ZERO);
            }
            let waj = wa.get(j).copied().unwrap_or(T::ZERO);
            let dj = s_diag.get(j).copied().unwrap_or(T::ZERO);
            if let Some(slot) = wa.get_mut(j) {
                *slot = (waj - sum) / dj;
            }
        }

        // Permute the solution back to original coordinates.
        let mut x = [T::ZERO; N];
        for (j, &target) in self.permutation.iter().enumerate() {
            if let (Some(slot), Some(&waj)) = (x.get_mut(target), wa.get(j)) {
                *slot = waj;
            }
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
            let Some(&l) = self.permutation.get(j) else {
                continue;
            };
            if self.column_norms.get(l).copied() != Some(T::ZERO) {
                let mut sum = T::ZERO;
                for (i, &qi) in self.qt_b.iter().enumerate().take(j + 1) {
                    sum += self.r.get(i, j).copied().unwrap_or(T::ZERO) * (qi / b_norm);
                }
                let norm = self.column_norms.get(l).copied().unwrap_or(T::ZERO);
                result = max(result, (sum / norm).abs());
            }
        }
        result
    }

    /// The norm `‖A x‖`, computed as `‖R P x‖` since `Q` has orthonormal columns.
    #[must_use]
    pub fn a_x_norm(&self, x: &Vector<N, T>) -> T {
        let mut w = [T::ZERO; N];
        for j in 0..N {
            let Some(&p) = self.permutation.get(j) else {
                continue;
            };
            let Some(&xl) = x.as_array().get(p) else {
                continue;
            };
            for (i, slot) in w.iter_mut().enumerate().take(j + 1) {
                *slot += self.r.get(i, j).copied().unwrap_or(T::ZERO) * xl;
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
        let threshold =
            T::EPSILON * T::from_usize(N) * self.r.get(0, 0).copied().unwrap_or(T::ZERO).abs();
        (0..N).all(|j| self.r.get(j, j).copied().unwrap_or(T::ZERO).abs() > threshold)
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
            match self.s_diag.get(j) {
                Some(&d) if d != T::ZERO => {
                    if let Some(s) = rhs.get_mut(j) {
                        *s /= d;
                    }
                }
                _ => {
                    if let Some(s) = rhs.get_mut(j) {
                        *s = T::ZERO;
                    }
                }
            }
            let temp = rhs.get(j).copied().unwrap_or(T::ZERO);
            for (i, slot) in rhs.iter_mut().enumerate().skip(j + 1) {
                *slot -= self.s.get(i, j).copied().unwrap_or(T::ZERO) * temp;
            }
        }
        rhs
    }
}
