//! Eigenvalues and directions of a symmetric matrix, by Jacobi rotations.
//!
//! Pairs of off-diagonal entries are rotated away one at a time until what is left is diagonal.
//! The method follows Golub & Van Loan, *Matrix Computations* — a fixed-size `no_std`
//! implementation on this crate's own [`Vector`] and [`Matrix`] types. Reference values for the
//! tests come from numpy/LAPACK.

use crate::error::LinalgError;
use crate::linear_algebra::{Matrix, Vector};
use crate::scalar::Numeric;

/// An eigendecomposition `A = V·diag(λ)·Vᵀ` of a symmetric matrix, as produced by
/// [`Matrix::symmetric_eigendecomposition`].
///
/// `eigenvalues` holds the λ largest first, and each column of `eigenvectors` is the direction
/// belonging to the eigenvalue in the same position. The columns are orthonormal.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct SymmetricEigendecomposition<const N: usize, T = f64> {
    /// The eigenvalues, largest first.
    pub(crate) eigenvalues: Vector<N, T>,
    /// Each column is the direction belonging to the eigenvalue in the same position.
    pub(crate) eigenvectors: Matrix<N, N, T>,
}

impl<const N: usize, T: Numeric> Matrix<N, N, T> {
    /// Decomposes `self` as `V·diag(λ)·Vᵀ` by Jacobi rotations.
    ///
    /// The eigenvalues come back largest first, each column of `V` is the direction belonging to
    /// the eigenvalue in the same position, and the columns are orthonormal. Returns
    /// [`LinalgError::NotSymmetric`] if the matrix does not read the same across the diagonal
    /// (allowing for rounding), or [`LinalgError::NonFinite`] if any entry is not finite.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// // This matrix has eigenvalues 3 and -1.
    /// let a = Matrix::<2, 2>::new([[1.0, 2.0], [2.0, 1.0]]);
    /// let decomposition = a.symmetric_eigendecomposition().unwrap();
    /// let values = decomposition.eigenvalues();
    /// assert!((values[0] - 3.0).abs() < 1e-12);
    /// assert!((values[1] + 1.0).abs() < 1e-12);
    ///
    /// // V·diag(λ)·Vᵀ == A.
    /// let vectors = decomposition.eigenvectors();
    /// for row in 0..2 {
    ///     for column in 0..2 {
    ///         let mut sum = 0.0;
    ///         for k in 0..2 {
    ///             sum += vectors[(row, k)] * values[k] * vectors[(column, k)];
    ///         }
    ///         assert!((sum - a[(row, column)]).abs() < 1e-12);
    ///     }
    /// }
    /// ```
    pub fn symmetric_eigendecomposition(
        self,
    ) -> Result<SymmetricEigendecomposition<N, T>, LinalgError> {
        if !self.is_finite() {
            return Err(LinalgError::NonFinite);
        }

        // Reject a matrix that does not read the same across the diagonal, and in the same walk
        // record the largest entry, which sets the scale everything below is measured against.
        let mut largest_entry = T::ZERO;
        for row in 0..N {
            largest_entry = largest_entry.max(self[(row, row)].abs());
            for column in (row + 1)..N {
                let upper = self[(row, column)];
                let lower = self[(column, row)];
                // The two only have to agree to within rounding, not exactly, so a matrix that
                // has drifted a little is still accepted while a genuinely lopsided one is not.
                // How much they are allowed to differ grows with how big they are, because
                // bigger numbers are stored more coarsely; the floor at one stops the allowance
                // from shrinking to exact equality for entries near zero.
                let scale = upper.abs().max(lower.abs()).max(T::ONE);
                if (upper - lower).abs() > T::EPSILON_X30 * scale {
                    return Err(LinalgError::NotSymmetric);
                }
                largest_entry = largest_entry.max(upper.abs()).max(lower.abs());
            }
        }

        // Anything this small counts as already zero, both for skipping a pair and for stopping.
        // A zero matrix gives a zero threshold, so every pair is skipped and the first sweep ends
        // the loop; and a pair that is rotated is strictly non-zero, so the rotation below never
        // divides by zero.
        let threshold = T::EPSILON * largest_entry;

        let mut working = self;
        let mut eigenvectors = Matrix::<N, N, T>::identity();

        // Rotate each off-diagonal pair away in turn, until a whole sweep leaves nothing to do.
        let max_sweeps = 60;
        for _ in 0..max_sweeps {
            let mut off_max = T::ZERO;
            for p in 0..N {
                for q in (p + 1)..N {
                    let off_diagonal = working[(p, q)];
                    off_max = off_max.max(off_diagonal.abs());
                    if off_diagonal.abs() <= threshold {
                        continue;
                    }

                    // Rotation that zeroes the entry shared by rows and columns p and q.
                    let alpha = working[(p, p)];
                    let beta = working[(q, q)];
                    let gamma = off_diagonal;
                    let zeta = (beta - alpha) / (T::TWO * gamma);
                    let sign = if zeta < T::ZERO { -T::ONE } else { T::ONE };
                    let t = sign / (zeta.abs() + (T::ONE + zeta * zeta).sqrt());
                    let c = T::ONE / (T::ONE + t * t).sqrt();
                    let s = c * t;

                    // Writing the two off-diagonal entries as exactly zero rather than computing
                    // them keeps the matrix reading the same across the diagonal as sweeps run.
                    working[(p, p)] = alpha - t * gamma;
                    working[(q, q)] = beta + t * gamma;
                    working[(p, q)] = T::ZERO;
                    working[(q, p)] = T::ZERO;

                    for i in 0..N {
                        if i == p || i == q {
                            continue;
                        }
                        let old = working[(i, p)];
                        let other = working[(i, q)];
                        working[(i, p)] = c * old - s * other;
                        working[(p, i)] = working[(i, p)];
                        working[(i, q)] = s * old + c * other;
                        working[(q, i)] = working[(i, q)];
                    }

                    for i in 0..N {
                        let old = eigenvectors[(i, p)];
                        let other = eigenvectors[(i, q)];
                        eigenvectors[(i, p)] = c * old - s * other;
                        eigenvectors[(i, q)] = s * old + c * other;
                    }
                }
            }
            if off_max <= threshold {
                break;
            }
        }

        // What is left on the diagonal are the eigenvalues.
        let mut eigenvalues = Vector::<N, T>::zeros();
        for index in 0..N {
            eigenvalues[index] = working[(index, index)];
        }

        // Sort the eigenvalues descending, carrying the matching columns.
        for k in 0..N {
            let mut top = k;
            for j in (k + 1)..N {
                if eigenvalues[j] > eigenvalues[top] {
                    top = j;
                }
            }
            if top != k {
                let tmp = eigenvalues[k];
                eigenvalues[k] = eigenvalues[top];
                eigenvalues[top] = tmp;
                for i in 0..N {
                    let tmp = eigenvectors[(i, k)];
                    eigenvectors[(i, k)] = eigenvectors[(i, top)];
                    eigenvectors[(i, top)] = tmp;
                }
            }
        }

        // Sign convention: the largest-magnitude entry of each column is positive, so the same
        // input gives the same output on every run and platform.
        for k in 0..N {
            let mut row = 0;
            let mut best = T::ZERO;
            for i in 0..N {
                let magnitude = eigenvectors[(i, k)].abs();
                if magnitude > best {
                    best = magnitude;
                    row = i;
                }
            }
            if eigenvectors[(row, k)] < T::ZERO {
                for i in 0..N {
                    eigenvectors[(i, k)] = -eigenvectors[(i, k)];
                }
            }
        }

        Ok(SymmetricEigendecomposition {
            eigenvalues,
            eigenvectors,
        })
    }
}

impl<const N: usize, T: Numeric> SymmetricEigendecomposition<N, T> {
    /// The eigenvalues, largest first.
    pub fn eigenvalues(&self) -> Vector<N, T> {
        self.eigenvalues
    }

    /// The directions, one per column, in the same order as the eigenvalues. The columns are
    /// orthonormal.
    pub fn eigenvectors(&self) -> Matrix<N, N, T> {
        self.eigenvectors
    }

    /// The determinant, the product of the eigenvalues.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let a = Matrix::<2, 2>::new([[1.0, 2.0], [2.0, 1.0]]);
    /// // The eigenvalues are 3 and -1, so the determinant is -3.
    /// let determinant = a.symmetric_eigendecomposition().unwrap().determinant();
    /// assert!((determinant + 3.0).abs() < 1e-12);
    /// ```
    #[inline]
    #[must_use]
    pub fn determinant(&self) -> T {
        let mut product = T::ONE;
        for index in 0..N {
            product *= self.eigenvalues[index];
        }
        product
    }

    /// The largest eigenvalue magnitude divided by the smallest, or infinity when an eigenvalue is
    /// zero.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let a = Matrix::<2, 2>::new([[1.0, 2.0], [2.0, 1.0]]);
    /// // The eigenvalues are 3 and -1, so the ratio of magnitudes is 3.
    /// let condition = a.symmetric_eigendecomposition().unwrap().condition_number();
    /// assert!((condition - 3.0).abs() < 1e-12);
    /// ```
    #[inline]
    #[must_use]
    pub fn condition_number(&self) -> T {
        if N == 0 {
            return T::INFINITY;
        }
        // The eigenvalues are sorted by value and may be negative, so both ends have to be found
        // by scanning magnitudes rather than read off the first and last entries.
        let mut largest = T::ZERO;
        let mut smallest = T::INFINITY;
        for index in 0..N {
            let magnitude = self.eigenvalues[index].abs();
            largest = largest.max(magnitude);
            smallest = smallest.min(magnitude);
        }
        if smallest <= T::ZERO {
            T::INFINITY
        } else {
            largest / smallest
        }
    }

    /// Whether every eigenvalue is strictly above zero.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let definite = Matrix::<2, 2>::new([[4.0, 2.0], [2.0, 3.0]]);
    /// let indefinite = Matrix::<2, 2>::new([[1.0, 2.0], [2.0, 1.0]]);
    /// assert!(definite.symmetric_eigendecomposition().unwrap().is_positive_definite());
    /// assert!(!indefinite.symmetric_eigendecomposition().unwrap().is_positive_definite());
    /// ```
    #[inline]
    #[must_use]
    pub fn is_positive_definite(&self) -> bool {
        if N == 0 {
            return true;
        }
        // Largest first, so the smallest eigenvalue is the last one.
        self.eigenvalues[N - 1] > T::ZERO
    }

    /// The matrix rebuilt with every eigenvalue raised to at least `minimum_eigenvalue`, keeping
    /// the same directions.
    ///
    /// This is what turns a covariance that has drifted below zero back into one a filter can keep
    /// using. The result reads exactly the same across the diagonal.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let a = Matrix::<2, 2>::new([[1.0, 2.0], [2.0, 1.0]]);
    /// // The eigenvalues are 3 and -1; lifting the floor to 0.5 leaves 3 and 0.5.
    /// let repaired = a.symmetric_eigendecomposition().unwrap().clamped(0.5);
    /// assert!((repaired[(0, 0)] - 1.75).abs() < 1e-12);
    /// assert!((repaired[(0, 1)] - 1.25).abs() < 1e-12);
    /// assert_eq!(repaired[(0, 1)], repaired[(1, 0)]);
    /// ```
    pub fn clamped(&self, minimum_eigenvalue: T) -> Matrix<N, N, T> {
        let mut raised = Vector::<N, T>::zeros();
        for index in 0..N {
            raised[index] = self.eigenvalues[index].max(minimum_eigenvalue);
        }

        // Only half is computed and then mirrored. Working out the two halves separately would
        // leave them a rounding step apart, which defeats the point for a caller repairing a
        // covariance.
        let mut result = Matrix::<N, N, T>::zeros();
        for row in 0..N {
            for column in row..N {
                let mut sum = T::ZERO;
                for k in 0..N {
                    sum += self.eigenvectors[(row, k)] * raised[k] * self.eigenvectors[(column, k)];
                }
                result[(row, column)] = sum;
                result[(column, row)] = sum;
            }
        }
        result
    }
}
