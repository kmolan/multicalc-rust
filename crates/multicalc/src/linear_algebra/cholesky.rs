//! Cholesky factorization for symmetric positive-definite matrices.
//!
//! A fixed-size `no_std` implementation of the standard Cholesky–Banachiewicz
//! algorithm on this crate's own [`Vector`] and [`Matrix`] types; results are checked against
//! numpy/LAPACK reference values.

use crate::error::LinalgError;
use crate::linear_algebra::{Matrix, Vector};
use crate::scalar::Numeric;

/// A Cholesky factorization `A = L·Lᵀ`, as produced by [`Matrix::cholesky`].
///
/// `L` is lower-triangular with a strictly positive diagonal; the entries above the diagonal are
/// zero. It exists only for a symmetric positive-definite `A`.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct Cholesky<const N: usize, T = f64> {
    /// Lower-triangular factor `L`, where `A = L·Lᵀ`.
    pub(crate) lower: Matrix<N, N, T>,
}

impl<const N: usize, T: Numeric> Matrix<N, N, T> {
    /// Factorizes `self` as `L·Lᵀ` by the Cholesky–Banachiewicz algorithm.
    ///
    /// Only the lower triangle is read; `self` is assumed symmetric. Returns
    /// [`LinalgError::NotPositiveDefinite`] if a diagonal radicand is not strictly positive — the
    /// matrix is not positive definite — rather than taking a root of it.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let a = Matrix::<3, 3>::new([[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]]);
    /// let lower = a.cholesky().unwrap().lower();
    /// // L·Lᵀ == A.
    /// let prod = lower * lower.transpose();
    /// for row in 0..3 {
    ///     for column in 0..3 {
    ///         assert!((prod[(row, column)] - a[(row, column)]).abs() < 1e-12);
    ///     }
    /// }
    /// ```
    pub fn cholesky(self) -> Result<Cholesky<N, T>, LinalgError> {
        let mut lower = Matrix::zeros();

        for j in 0..N {
            // Diagonal entry: subtract the squares already placed in row j.
            let mut diag = self[(j, j)];
            for k in 0..j {
                diag -= lower[(j, k)] * lower[(j, k)];
            }
            if diag <= T::ZERO {
                return Err(LinalgError::NotPositiveDefinite);
            }
            let ljj = diag.sqrt();
            lower[(j, j)] = ljj;

            // Below-diagonal entries of column j.
            for i in (j + 1)..N {
                let mut sum = self[(i, j)];
                for k in 0..j {
                    sum -= lower[(i, k)] * lower[(j, k)];
                }
                lower[(i, j)] = sum / ljj;
            }
        }

        Ok(Cholesky { lower })
    }
}

impl<const N: usize, T: Numeric> Cholesky<N, T> {
    /// The lower-triangular factor `L`, where `A = L·Lᵀ`.
    pub fn lower(&self) -> Matrix<N, N, T> {
        self.lower
    }

    /// The determinant, `Π L[i][i]²`.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let a = Matrix::<2, 2>::new([[4.0, 2.0], [2.0, 3.0]]);
    /// assert!((a.cholesky().unwrap().determinant() - a.determinant()).abs() < 1e-12);
    /// ```
    #[inline]
    #[must_use]
    pub fn determinant(&self) -> T {
        let mut det = T::ONE;
        for i in 0..N {
            det *= self.lower[(i, i)] * self.lower[(i, i)];
        }
        det
    }

    /// Solves `A·x = b` for `x`, reusing this factorization.
    ///
    /// Infallible: the factorization already guaranteed every `L` diagonal entry is positive.
    ///
    /// ```
    /// use multicalc::linear_algebra::{Matrix, Vector};
    /// let a = Matrix::<2, 2>::new([[4.0, 2.0], [2.0, 3.0]]);
    /// let b = Vector::new([8.0, 8.0]);
    /// // A·x = b has the exact solution x = [1, 2].
    /// let x = a.cholesky().unwrap().solve(b);
    /// assert!((x[0] - 1.0).abs() < 1e-12);
    /// assert!((x[1] - 2.0).abs() < 1e-12);
    /// ```
    pub fn solve(&self, b: Vector<N, T>) -> Vector<N, T> {
        let mut x: [T; N] = core::array::from_fn(|i| b[i]);

        // Forward substitution for L·y = b.
        for i in 0..N {
            let mut sum = x[i];
            for (j, &x_j) in x.iter().enumerate().take(i) {
                sum -= self.lower[(i, j)] * x_j;
            }
            x[i] = sum / self.lower[(i, i)];
        }

        // Back substitution for Lᵀ·x = y, where Lᵀ[i][j] = L[j][i].
        for i in (0..N).rev() {
            let mut sum = x[i];
            for (j, &x_j) in x.iter().enumerate().skip(i + 1) {
                sum -= self.lower[(j, i)] * x_j;
            }
            x[i] = sum / self.lower[(i, i)];
        }

        Vector::new(x)
    }

    /// Solves `A·X = B` for `X`, one column at a time, reusing this factorization.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let a = Matrix::<2, 2>::new([[4.0, 2.0], [2.0, 3.0]]);
    /// let identity = Matrix::<2, 2>::identity();
    /// // Solving A·X = I gives X = A⁻¹.
    /// let x = a.cholesky().unwrap().solve_matrix(identity);
    /// let product = a * x;
    /// assert!((product[(0, 0)] - 1.0).abs() < 1e-12);
    /// assert!((product[(1, 1)] - 1.0).abs() < 1e-12);
    /// ```
    pub fn solve_matrix<const K: usize>(&self, b: Matrix<N, K, T>) -> Matrix<N, K, T> {
        let mut result = Matrix::zeros();
        for column in 0..K {
            let rhs_column = Vector::from_fn(|row| b[(row, column)]);
            let x = self.solve(rhs_column);
            for row in 0..N {
                result[(row, column)] = x[row];
            }
        }
        result
    }

    /// The inverse of the factorized matrix, from solving `A·X = I`.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let a = Matrix::<3, 3>::new([[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]]);
    /// let product = a * a.cholesky().unwrap().inverse();
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
