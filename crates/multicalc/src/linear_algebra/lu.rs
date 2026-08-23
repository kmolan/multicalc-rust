//! LU factorization with partial pivoting (Doolittle), for square systems.
//!
//! A fixed-size `no_std` implementation of the standard Doolittle algorithm with
//! partial pivoting on this crate's own [`Vector`] and [`Matrix`] types; results are checked
//! against numpy/LAPACK reference values.

use crate::error::LinalgError;
use crate::linear_algebra::{Matrix, Vector};
use crate::scalar::Numeric;

/// An LU factorization with partial pivoting, as produced by [`Matrix::lu_decompose`].
///
/// The two triangular factors share one matrix: the strict lower triangle holds `L` (its unit
/// diagonal is implicit), and the diagonal and upper triangle hold `U`. `perm` records the row
/// order after pivoting, so `P·A = L·U`, where row `i` of `P·A` is row `perm[i]` of `A`. `sign`
/// is the determinant of `P` (`+1` or `-1`), the parity of the row swaps.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct LuDecomposition<const N: usize, T = f64> {
    /// Packed factors: `L` below the diagonal, `U` on and above it.
    pub(crate) lu_decompose: Matrix<N, N, T>,
    /// Row order after pivoting: row `i` of `P·A` is row `perm[i]` of `A`.
    pub(crate) perm: [usize; N],
    /// Sign of the row-swap permutation, used by the determinant.
    pub(crate) sign: T,
}

impl<const N: usize, T: Numeric> Matrix<N, N, T> {
    /// Factorizes `self` by Doolittle LU with partial pivoting.
    ///
    /// Returns [`LinalgError::Singular`] if a pivot column is entirely zero, or
    /// [`LinalgError::IllConditioned`] if the smallest pivot is at most `EPSILON` times the
    /// largest absolute matrix entry, rather than returning a factorization whose solves divide
    /// by a negligible value.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let a = Matrix::<3, 3>::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]);
    /// let f = a.lu_decompose().unwrap();
    /// // P·A == L·U.
    /// let (lower, upper, perm) = (f.lower(), f.upper(), f.permutation());
    /// let pa = Matrix::<3, 3>::from_fn(|i, column| a[(perm[i], column)]);
    /// let prod = lower * upper;
    /// for i in 0..3 {
    ///     for column in 0..3 {
    ///         assert!((pa[(i, column)] - prod[(i, column)]).abs() < 1e-12);
    ///     }
    /// }
    /// ```
    pub fn lu_decompose(self) -> Result<LuDecomposition<N, T>, LinalgError> {
        let scale = self.max_abs();
        let (factorization, smallest_pivot) = self.factor_lu()?;
        if smallest_pivot <= T::EPSILON * scale {
            return Err(LinalgError::IllConditioned);
        }
        Ok(factorization)
    }

    /// Factorizes for determinant evaluation, where a small nonzero pivot is still meaningful.
    pub(super) fn lu_for_determinant(self) -> Result<LuDecomposition<N, T>, LinalgError> {
        self.factor_lu().map(|(factorization, _)| factorization)
    }

    fn factor_lu(self) -> Result<(LuDecomposition<N, T>, T), LinalgError> {
        let mut a = self;
        let mut perm: [usize; N] = core::array::from_fn(|i| i);
        let mut sign = T::ONE;
        let mut smallest_pivot = T::MAX;

        for k in 0..N {
            // Partial pivot: largest magnitude in column k on or below the diagonal.
            let mut pivot = k;
            let mut best = a[(k, k)].abs();
            for i in (k + 1)..N {
                let magnitude = a[(i, k)].abs();
                if magnitude > best {
                    best = magnitude;
                    pivot = i;
                }
            }
            if best == T::ZERO {
                return Err(LinalgError::Singular);
            }
            smallest_pivot = smallest_pivot.min(best);
            if pivot != k {
                a.as_mut_slice_rows().swap(k, pivot);
                perm.swap(k, pivot);
                sign = -sign;
            }
            // Eliminate below the pivot, storing each multiplier in L's place.
            for i in (k + 1)..N {
                let factor = a[(i, k)] / a[(k, k)];
                a[(i, k)] = factor;
                for j in (k + 1)..N {
                    let pivot_row = a[(k, j)];
                    a[(i, j)] -= factor * pivot_row;
                }
            }
        }

        Ok((
            LuDecomposition {
                lu_decompose: a,
                perm,
                sign,
            },
            smallest_pivot,
        ))
    }

    /// Solves `A·x = b` for `x`, factorizing `self` by LU.
    ///
    /// A one-call convenience over [`Matrix::lu_decompose`] followed by [`LuDecomposition::solve`]. Returns
    /// [`LinalgError::Singular`] if `self` is singular or [`LinalgError::IllConditioned`] if a
    /// pivot is too small relative to the matrix. To solve several right-hand sides, factor once
    /// with [`Matrix::lu_decompose`] and reuse the result. For a symmetric positive-definite matrix,
    /// [`Matrix::cholesky`] is faster.
    ///
    /// ```
    /// use multicalc::linear_algebra::{Matrix, Vector};
    /// let a = Matrix::<3, 3>::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]);
    /// let b = Vector::new([7.0, 19.0, 49.0]);
    /// let x = a.solve(b).unwrap();
    /// assert!((x[0] - 1.0).abs() < 1e-12);
    /// assert!((x[1] - 2.0).abs() < 1e-12);
    /// assert!((x[2] - 3.0).abs() < 1e-12);
    /// ```
    pub fn solve(self, b: Vector<N, T>) -> Result<Vector<N, T>, LinalgError> {
        Ok(self.lu_decompose()?.solve(b))
    }
}

impl<const N: usize, T: Numeric> LuDecomposition<N, T> {
    /// The unit lower-triangular factor `L` (ones on the diagonal).
    pub fn lower(&self) -> Matrix<N, N, T> {
        Matrix::from_fn(|row, column| {
            if row == column {
                T::ONE
            } else if column < row {
                self.lu_decompose[(row, column)]
            } else {
                T::ZERO
            }
        })
    }

    /// The upper-triangular factor `U`.
    pub fn upper(&self) -> Matrix<N, N, T> {
        Matrix::from_fn(|row, column| {
            if column >= row {
                self.lu_decompose[(row, column)]
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
    /// assert!((a.lu_decompose().unwrap().determinant() - a.determinant()).abs() < 1e-12);
    /// ```
    #[inline]
    #[must_use]
    pub fn determinant(&self) -> T {
        let mut det = self.sign;
        for i in 0..N {
            det *= self.lu_decompose[(i, i)];
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
    /// let b = Vector::new([7.0, 19.0, 49.0]);
    /// // A·x = b has the exact solution x = [1, 2, 3].
    /// let x = a.lu_decompose().unwrap().solve(b);
    /// assert!((x[0] - 1.0).abs() < 1e-12);
    /// assert!((x[1] - 2.0).abs() < 1e-12);
    /// assert!((x[2] - 3.0).abs() < 1e-12);
    /// ```
    pub fn solve(&self, b: Vector<N, T>) -> Vector<N, T> {
        // Apply the row permutation: start from P·b.
        let mut x: [T; N] = core::array::from_fn(|i| b[self.perm[i]]);

        // Forward substitution for L·y = P·b (L has a unit diagonal).
        for i in 0..N {
            let mut sum = x[i];
            for (j, &x_j) in x.iter().enumerate().take(i) {
                sum -= self.lu_decompose[(i, j)] * x_j;
            }
            x[i] = sum;
        }

        // Back substitution for U·x = y.
        for i in (0..N).rev() {
            let mut sum = x[i];
            for (j, &x_j) in x.iter().enumerate().skip(i + 1) {
                sum -= self.lu_decompose[(i, j)] * x_j;
            }
            x[i] = sum / self.lu_decompose[(i, i)];
        }

        Vector::new(x)
    }

    /// Solves `A·X = B` for `X`, one column at a time, reusing this factorization.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let a = Matrix::<2, 2>::new([[4.0, 3.0], [6.0, 3.0]]);
    /// let identity = Matrix::<2, 2>::identity();
    /// // Solving A·X = I gives X = A⁻¹.
    /// let x = a.lu_decompose().unwrap().solve_matrix(identity);
    /// let product = a * x;
    /// assert!((product[(0, 0)] - 1.0).abs() < 1e-12);
    /// assert!((product[(1, 1)] - 1.0).abs() < 1e-12);
    /// ```
    pub fn solve_matrix<const K: usize>(&self, b: Matrix<N, K, T>) -> Matrix<N, K, T> {
        let mut result = Matrix::zeros();
        for col in 0..K {
            let rhs_column = Vector::from_fn(|row| b[(row, col)]);
            let x = self.solve(rhs_column);
            for row in 0..N {
                result[(row, col)] = x[row];
            }
        }
        result
    }

    /// The inverse of the factorized matrix, from solving `A·X = I`.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let a = Matrix::<3, 3>::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]);
    /// let product = a * a.lu_decompose().unwrap().inverse();
    /// for row in 0..3 {
    ///     for column in 0..3 {
    ///         let expected = if row == column { 1.0 } else { 0.0 };
    ///         assert!((product[(row, column)] - expected).abs() < 1e-12);
    ///     }
    /// }
    /// ```
    pub fn inverse(&self) -> Matrix<N, N, T> {
        self.solve_matrix(Matrix::identity())
    }
}
