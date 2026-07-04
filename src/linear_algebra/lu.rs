//! LU factorization with partial pivoting (Doolittle), for square systems.

use crate::linear_algebra::{Matrix, Vector};
use crate::scalar::Numeric;
use crate::utils::error_codes::CalcError;

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
    /// Returns [`CalcError::SingularMatrix`] if a pivot column is entirely zero — the largest
    /// available pivot is zero — rather than dividing by it.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let a = Matrix::<3, 3>::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]);
    /// let f = a.lu().unwrap();
    /// // P·A == L·U.
    /// let (l, u, perm) = (f.l(), f.u(), f.permutation());
    /// let pa = Matrix::<3, 3>::from_fn(|i, c| a[(perm[i], c)]);
    /// let prod = l * u;
    /// for i in 0..3 {
    ///     for c in 0..3 {
    ///         assert!((pa[(i, c)] - prod[(i, c)]).abs() < 1e-12);
    ///     }
    /// }
    /// ```
    pub fn lu(self) -> Result<Lu<N, T>, CalcError> {
        let mut a = self;
        let mut perm: [usize; N] = core::array::from_fn(|i| i);
        let mut sign = T::ONE;

        for k in 0..N {
            // Partial pivot: largest magnitude in column k on or below the diagonal.
            let mut p = k;
            let mut best = a[(k, k)].abs();
            for i in (k + 1)..N {
                let magnitude = a[(i, k)].abs();
                if magnitude > best {
                    best = magnitude;
                    p = i;
                }
            }
            if a[(p, k)] == T::ZERO {
                return Err(CalcError::SingularMatrix);
            }
            if p != k {
                for c in 0..N {
                    let tmp = a[(k, c)];
                    a[(k, c)] = a[(p, c)];
                    a[(p, c)] = tmp;
                }
                perm.swap(k, p);
                sign = -sign;
            }
            // Eliminate below the pivot, storing each multiplier in L's place.
            for i in (k + 1)..N {
                let factor = a[(i, k)] / a[(k, k)];
                a[(i, k)] = factor;
                for j in (k + 1)..N {
                    let term = factor * a[(k, j)];
                    a[(i, j)] -= term;
                }
            }
        }

        Ok(Lu { lu: a, perm, sign })
    }

    /// Solves `A·x = b` for `x`, factorizing `self` by LU.
    ///
    /// A one-call convenience over [`Matrix::lu`] followed by [`Lu::solve`]. Returns
    /// [`CalcError::SingularMatrix`] if `self` is singular. To solve several right-hand sides,
    /// factor once with [`Matrix::lu`] and reuse the result. For a symmetric positive-definite
    /// matrix, [`Matrix::cholesky`] is faster.
    ///
    /// ```
    /// use multicalc::linear_algebra::{Matrix, Vector};
    /// let a = Matrix::<3, 3>::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]);
    /// let x = a.solve(Vector::new([7.0, 19.0, 49.0])).unwrap();
    /// assert!((x[0] - 1.0).abs() < 1e-12);
    /// assert!((x[1] - 2.0).abs() < 1e-12);
    /// assert!((x[2] - 3.0).abs() < 1e-12);
    /// ```
    pub fn solve(self, b: Vector<N, T>) -> Result<Vector<N, T>, CalcError> {
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
                self.lu[(r, c)]
            } else {
                T::ZERO
            }
        })
    }

    /// The upper-triangular factor `U`.
    pub fn u(&self) -> Matrix<N, N, T> {
        Matrix::from_fn(|r, c| if c >= r { self.lu[(r, c)] } else { T::ZERO })
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
            det *= self.lu[(i, i)];
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
            for (j, &xj) in x.iter().enumerate().take(i) {
                sum -= self.lu[(i, j)] * xj;
            }
            x[i] = sum;
        }

        // Back substitution for U·x = y.
        for i in (0..N).rev() {
            let mut sum = x[i];
            for (j, &xj) in x.iter().enumerate().skip(i + 1) {
                sum -= self.lu[(i, j)] * xj;
            }
            x[i] = sum / self.lu[(i, i)];
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
    /// let p = a * x;
    /// assert!((p[(0, 0)] - 1.0).abs() < 1e-12);
    /// assert!((p[(1, 1)] - 1.0).abs() < 1e-12);
    /// ```
    pub fn solve_matrix<const K: usize>(&self, b: Matrix<N, K, T>) -> Matrix<N, K, T> {
        let mut result = Matrix::zeros();
        for c in 0..K {
            let x = self.solve(b.column(c));
            for r in 0..N {
                result[(r, c)] = x[r];
            }
        }
        result
    }

    /// The inverse of the factorized matrix, from solving `A·X = I`.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let a = Matrix::<3, 3>::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]);
    /// let p = a * a.lu().unwrap().inverse();
    /// for r in 0..3 {
    ///     for c in 0..3 {
    ///         let expected = if r == c { 1.0 } else { 0.0 };
    ///         assert!((p[(r, c)] - expected).abs() < 1e-12);
    ///     }
    /// }
    /// ```
    pub fn inverse(&self) -> Matrix<N, N, T> {
        self.solve_matrix(Matrix::identity())
    }
}
