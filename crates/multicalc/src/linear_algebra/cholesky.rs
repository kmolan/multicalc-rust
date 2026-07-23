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
    pub(crate) l: Matrix<N, N, T>,
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
    /// let l = a.cholesky().unwrap().l();
    /// // L·Lᵀ == A.
    /// let (prod, aa) = ((l * l.transpose()).into_array(), a.into_array());
    /// for r in 0..3 {
    ///     for c in 0..3 {
    ///         assert!((prod[r][c] - aa[r][c]).abs() < 1e-12);
    ///     }
    /// }
    /// ```
    pub fn cholesky(self) -> Result<Cholesky<N, T>, LinalgError> {
        let mut l = Matrix::zeros();

        for j in 0..N {
            // Diagonal entry: subtract the squares already placed in row j.
            let mut d = self.get(j, j).copied().unwrap_or(T::ZERO);
            for k in 0..j {
                let ljk = l.get(j, k).copied().unwrap_or(T::ZERO);
                d -= ljk * ljk;
            }
            if d <= T::ZERO {
                return Err(LinalgError::NotPositiveDefinite);
            }
            let ljj = d.sqrt();
            if let Some(slot) = l.get_mut(j, j) {
                *slot = ljj;
            }

            // Below-diagonal entries of column j.
            for i in (j + 1)..N {
                let mut s = self.get(i, j).copied().unwrap_or(T::ZERO);
                for k in 0..j {
                    let lik = l.get(i, k).copied().unwrap_or(T::ZERO);
                    let ljk = l.get(j, k).copied().unwrap_or(T::ZERO);
                    s -= lik * ljk;
                }
                if let Some(slot) = l.get_mut(i, j) {
                    *slot = s / ljj;
                }
            }
        }

        Ok(Cholesky { l })
    }
}

impl<const N: usize, T: Numeric> Cholesky<N, T> {
    /// The lower-triangular factor `L`, where `A = L·Lᵀ`.
    pub fn l(&self) -> Matrix<N, N, T> {
        self.l
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
            let di = self.l.get(i, i).copied().unwrap_or(T::ONE);
            det *= di * di;
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
    /// // A·x = b has the exact solution x = [1, 2].
    /// let x = a.cholesky().unwrap().solve(Vector::new([8.0, 8.0]));
    /// let [x0, x1] = *x.as_array();
    /// assert!((x0 - 1.0).abs() < 1e-12);
    /// assert!((x1 - 2.0).abs() < 1e-12);
    /// ```
    pub fn solve(&self, b: Vector<N, T>) -> Vector<N, T> {
        let mut x: [T; N] = core::array::from_fn(|i| b.get(i).copied().unwrap_or(T::ZERO));

        // Forward substitution for L·y = b.
        for i in 0..N {
            let mut sum = x[i];
            for (j, &xj) in x.iter().enumerate().take(i) {
                sum -= self.l.get(i, j).copied().unwrap_or(T::ZERO) * xj;
            }
            let diag = self.l.get(i, i).copied().unwrap_or(T::ONE);
            x[i] = sum / diag;
        }

        // Back substitution for Lᵀ·x = y, where Lᵀ[i][j] = L[j][i].
        for i in (0..N).rev() {
            let mut sum = x[i];
            for (j, &xj) in x.iter().enumerate().skip(i + 1) {
                sum -= self.l.get(j, i).copied().unwrap_or(T::ZERO) * xj;
            }
            let diag = self.l.get(i, i).copied().unwrap_or(T::ONE);
            x[i] = sum / diag;
        }

        Vector::new(x)
    }

    /// Solves `A·X = B` for `X`, one column at a time, reusing this factorization.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let a = Matrix::<2, 2>::new([[4.0, 2.0], [2.0, 3.0]]);
    /// // Solving A·X = I gives X = A⁻¹.
    /// let x = a.cholesky().unwrap().solve_matrix(Matrix::<2, 2>::identity());
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
    /// let a = Matrix::<3, 3>::new([[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]]);
    /// let p = (a * a.cholesky().unwrap().inverse()).into_array();
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
