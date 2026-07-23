//! LU factorization with partial pivoting (Doolittle), for square systems.
//!
//! A fixed-size `no_std` implementation of the standard Doolittle algorithm with
//! partial pivoting on this crate's own [`Vector`] and [`Matrix`] types; results are checked
//! against numpy/LAPACK reference values.

use crate::error::LinalgError;
use crate::linear_algebra::{Matrix, Vector};
use crate::scalar::Numeric;

/// An LU factorization with partial pivoting, as produced by [`Matrix::lu`].
///
/// The two triangular factors share one matrix: the strict lower triangle holds `L` (its unit
/// diagonal is implicit), and the diagonal and upper triangle hold `U`. `perm` records the row
/// order after pivoting, so `P·A = L·U`, where row `i` of `P·A` is row `perm[i]` of `A`. `sign`
/// is the determinant of `P` (`+1` or `-1`), the parity of the row swaps.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct Lu<const N: usize, T = f64> {
    /// Packed factors: `L` below the diagonal, `U` on and above it.
    pub(crate) lu: Matrix<N, N, T>,
    /// Row order after pivoting: row `i` of `P·A` is row `perm[i]` of `A`.
    pub(crate) perm: [usize; N],
    /// Sign of the row-swap permutation, used by the determinant.
    pub(crate) sign: T,
}

impl<const N: usize, T: Numeric> Matrix<N, N, T> {
    /// Factorizes `self` by Doolittle LU with partial pivoting.
    ///
    /// Returns [`LinalgError::Singular`] if a pivot column is entirely zero — the largest
    /// available pivot is zero — rather than dividing by it.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let a = Matrix::<3, 3>::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]);
    /// let f = a.lu().unwrap();
    /// // P·A == L·U.
    /// let (l, u, perm) = (f.l(), f.u(), f.permutation());
    /// let aa = a.into_array();
    /// let pa = Matrix::<3, 3>::from_fn(|i, c| aa[perm[i]][c]);
    /// let (pa, prod) = (pa.into_array(), (l * u).into_array());
    /// for i in 0..3 {
    ///     for c in 0..3 {
    ///         assert!((pa[i][c] - prod[i][c]).abs() < 1e-12);
    ///     }
    /// }
    /// ```
    pub fn lu(self) -> Result<Lu<N, T>, LinalgError> {
        let mut a = self;
        let mut perm: [usize; N] = core::array::from_fn(|i| i);
        let mut sign = T::ONE;

        for k in 0..N {
            // Partial pivot: largest magnitude in column k on or below the diagonal.
            let mut p = k;
            let mut best = a.get(k, k).copied().unwrap_or(T::ZERO).abs();
            for i in (k + 1)..N {
                let magnitude = a.get(i, k).copied().unwrap_or(T::ZERO).abs();
                if magnitude > best {
                    best = magnitude;
                    p = i;
                }
            }
            if a.get(p, k).copied().unwrap_or(T::ZERO) == T::ZERO {
                return Err(LinalgError::Singular);
            }
            if p != k {
                a.as_mut_slice_rows().swap(k, p);
                perm.swap(k, p);
                sign = -sign;
            }
            // Eliminate below the pivot, storing each multiplier in L's place.
            for i in (k + 1)..N {
                let pivot = a.get(k, k).copied().unwrap_or(T::ZERO);
                let factor = a.get(i, k).copied().unwrap_or(T::ZERO) / pivot;
                if let Some(slot) = a.get_mut(i, k) {
                    *slot = factor;
                }
                for j in (k + 1)..N {
                    let akj = a.get(k, j).copied().unwrap_or(T::ZERO);
                    if let Some(slot) = a.get_mut(i, j) {
                        *slot -= factor * akj;
                    }
                }
            }
        }

        Ok(Lu { lu: a, perm, sign })
    }

    /// Solves `A·x = b` for `x`, factorizing `self` by LU.
    ///
    /// A one-call convenience over [`Matrix::lu`] followed by [`Lu::solve`]. Returns
    /// [`LinalgError::Singular`] if `self` is singular. To solve several right-hand sides,
    /// factor once with [`Matrix::lu`] and reuse the result. For a symmetric positive-definite
    /// matrix, [`Matrix::cholesky`] is faster.
    ///
    /// ```
    /// use multicalc::linear_algebra::{Matrix, Vector};
    /// let a = Matrix::<3, 3>::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]);
    /// let x = a.solve(Vector::new([7.0, 19.0, 49.0])).unwrap();
    /// let [x0, x1, x2] = *x.as_array();
    /// assert!((x0 - 1.0).abs() < 1e-12);
    /// assert!((x1 - 2.0).abs() < 1e-12);
    /// assert!((x2 - 3.0).abs() < 1e-12);
    /// ```
    pub fn solve(self, b: Vector<N, T>) -> Result<Vector<N, T>, LinalgError> {
        Ok(self.lu()?.solve(b))
    }
}

impl<const N: usize, T: Numeric> Lu<N, T> {
    /// The unit lower-triangular factor `L` (ones on the diagonal).
    pub fn l(&self) -> Matrix<N, N, T> {
        Matrix::from_fn(|r, c| {
            if r == c {
                T::ONE
            } else if c < r {
                self.lu.get(r, c).copied().unwrap_or(T::ZERO)
            } else {
                T::ZERO
            }
        })
    }

    /// The upper-triangular factor `U`.
    pub fn u(&self) -> Matrix<N, N, T> {
        Matrix::from_fn(|r, c| {
            if c >= r {
                self.lu.get(r, c).copied().unwrap_or(T::ZERO)
            } else {
                T::ZERO
            }
        })
    }

    /// The row order after pivoting: row `i` of `P·A` is row `permutation()[i]` of `A`.
    #[inline]
    #[must_use]
    pub fn permutation(&self) -> [usize; N] {
        self.perm
    }

    /// The determinant, `sign · Π U[i][i]`.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let a = Matrix::<3, 3>::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]);
    /// assert!((a.lu().unwrap().determinant() - a.determinant()).abs() < 1e-12);
    /// ```
    #[inline]
    #[must_use]
    pub fn determinant(&self) -> T {
        let mut det = self.sign;
        for i in 0..N {
            det *= self.lu.get(i, i).copied().unwrap_or(T::ONE);
        }
        det
    }

    /// Solves `A·x = b` for `x`, reusing this factorization.
    ///
    /// Infallible: the factorization already guaranteed every `U` diagonal entry is nonzero.
    ///
    /// ```
    /// use multicalc::linear_algebra::{Matrix, Vector};
    /// let a = Matrix::<3, 3>::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]);
    /// // A·x = b has the exact solution x = [1, 2, 3].
    /// let x = a.lu().unwrap().solve(Vector::new([7.0, 19.0, 49.0]));
    /// let [x0, x1, x2] = *x.as_array();
    /// assert!((x0 - 1.0).abs() < 1e-12);
    /// assert!((x1 - 2.0).abs() < 1e-12);
    /// assert!((x2 - 3.0).abs() < 1e-12);
    /// ```
    pub fn solve(&self, b: Vector<N, T>) -> Vector<N, T> {
        // Apply the row permutation: start from P·b.
        let b = b.as_slice();
        let mut x: [T; N] = core::array::from_fn(|i| {
            self.perm
                .get(i)
                .and_then(|&p| b.get(p))
                .copied()
                .unwrap_or(T::ZERO)
        });

        // Forward substitution for L·y = P·b (L has a unit diagonal).
        for i in 0..N {
            let mut sum = x[i];
            for (j, &xj) in x.iter().enumerate().take(i) {
                sum -= self.lu.get(i, j).copied().unwrap_or(T::ZERO) * xj;
            }
            x[i] = sum;
        }

        // Back substitution for U·x = y.
        for i in (0..N).rev() {
            let mut sum = x[i];
            for (j, &xj) in x.iter().enumerate().skip(i + 1) {
                sum -= self.lu.get(i, j).copied().unwrap_or(T::ZERO) * xj;
            }
            let diag = self.lu.get(i, i).copied().unwrap_or(T::ONE);
            x[i] = sum / diag;
        }

        Vector::new(x)
    }

    /// Solves `A·X = B` for `X`, one column at a time, reusing this factorization.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let a = Matrix::<2, 2>::new([[4.0, 3.0], [6.0, 3.0]]);
    /// // Solving A·X = I gives X = A⁻¹.
    /// let x = a.lu().unwrap().solve_matrix(Matrix::<2, 2>::identity());
    /// let p = (a * x).into_array();
    /// assert!((p[0][0] - 1.0).abs() < 1e-12);
    /// assert!((p[1][1] - 1.0).abs() < 1e-12);
    /// ```
    pub fn solve_matrix<const K: usize>(&self, b: Matrix<N, K, T>) -> Matrix<N, K, T> {
        let mut result = Matrix::zeros();
        for c in 0..K {
            let Some(col) = b.try_column(c) else { continue };
            let x = self.solve(col);
            for r in 0..N {
                let Some(&xr) = x.get(r) else { continue };
                if let Some(slot) = result.get_mut(r, c) {
                    *slot = xr;
                }
            }
        }
        result
    }

    /// The inverse of the factorized matrix, from solving `A·X = I`.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let a = Matrix::<3, 3>::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]);
    /// let p = (a * a.lu().unwrap().inverse()).into_array();
    /// for r in 0..3 {
    ///     for c in 0..3 {
    ///         let expected = if r == c { 1.0 } else { 0.0 };
    ///         assert!((p[r][c] - expected).abs() < 1e-12);
    ///     }
    /// }
    /// ```
    pub fn inverse(&self) -> Matrix<N, N, T> {
        self.solve_matrix(Matrix::identity())
    }
}
