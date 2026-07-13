//! Singular value decomposition by one-sided Jacobi, for tall or square matrices.
//!
//! The method follows Golub & Van Loan, *Matrix Computations*, and Demmel & Veselić for high
//! relative accuracy — a fixed-size `no_std` implementation on this crate's own
//! [`Vector`] and [`Matrix`] types. Reference values for the tests come from numpy/LAPACK.

use crate::linear_algebra::{Matrix, Vector};
use crate::scalar::Numeric;
use crate::error::LinalgError;

/// A thin singular value decomposition `A = U · diag(σ) · Vᵀ` for a matrix with `M ≥ N`.
///
/// `u` has orthonormal columns, `singular_values` holds the σ in descending order (all ≥ 0), and
/// `v` has orthonormal columns.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct Svd<const M: usize, const N: usize, T = f64> {
    /// Left factor `U` with orthonormal columns.
    pub(crate) u: Matrix<M, N, T>,
    /// Singular values in descending order.
    pub(crate) singular_values: Vector<N, T>,
    /// Right factor `V` with orthonormal columns.
    pub(crate) v: Matrix<N, N, T>,
}

impl<const M: usize, const N: usize, T: Numeric> Matrix<M, N, T> {
    /// Decomposes `self` as `U · diag(σ) · Vᵀ` by one-sided Jacobi (thin form, `M ≥ N`).
    ///
    /// `U` has orthonormal columns, the σ are non-negative and descending, and `V` has orthonormal
    /// columns. Returns [`LinalgError::Underdetermined`] for a wide matrix (`M < N`) — transpose it
    /// first — or [`LinalgError::NonFinite`] if any entry is not finite.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let a = Matrix::<3, 2>::new([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    /// let svd = a.svd().unwrap();
    /// let (u, s, v) = (svd.u(), svd.singular_values(), svd.v());
    /// // U · diag(σ) · Vᵀ == A.
    /// for r in 0..3 {
    ///     for c in 0..2 {
    ///         let mut acc = 0.0;
    ///         for k in 0..2 {
    ///             acc += u[(r, k)] * s[k] * v[(c, k)];
    ///         }
    ///         assert!((acc - a[(r, c)]).abs() < 1e-12);
    ///     }
    /// }
    /// ```
    ///
    /// A wide matrix (`M < N`) has no thin form here; take the SVD of its transpose, whose singular
    /// values are the same. Its pseudo-inverse then follows from `A⁺ = ((Aᵀ)⁺)ᵀ`, which
    /// [`Matrix::pseudo_inverse`] applies for any shape.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// // For a wide matrix, decompose its transpose.
    /// let a = Matrix::<2, 3>::new([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]]);
    /// let at = a.transpose();
    /// let svd = at.svd().unwrap();
    /// let (u, s, v) = (svd.u(), svd.singular_values(), svd.v());
    /// for r in 0..3 {
    ///     for c in 0..2 {
    ///         let mut acc = 0.0;
    ///         for k in 0..2 {
    ///             acc += u[(r, k)] * s[k] * v[(c, k)];
    ///         }
    ///         assert!((acc - at[(r, c)]).abs() < 1e-12);
    ///     }
    /// }
    /// ```
    pub fn svd(self) -> Result<Svd<M, N, T>, LinalgError> {
        if M < N {
            return Err(LinalgError::Underdetermined);
        }
        for r in 0..M {
            for c in 0..N {
                if !self[(r, c)].is_finite() {
                    return Err(LinalgError::NonFinite);
                }
            }
        }

        let mut u = self;
        let mut v = Matrix::<N, N, T>::identity();

        // One-sided Jacobi: rotate column pairs of U until its columns are orthogonal.
        let max_sweeps = 60;
        for _ in 0..max_sweeps {
            let mut off_max = T::ZERO;
            for p in 0..N {
                for q in (p + 1)..N {
                    let cp = u.column(p);
                    let cq = u.column(q);
                    let alpha = cp.norm_squared();
                    let beta = cq.norm_squared();
                    let gamma = cp.dot(cq);
                    if alpha == T::ZERO || beta == T::ZERO {
                        continue;
                    }
                    let scale = (alpha * beta).sqrt();
                    let off = gamma.abs() / scale;
                    if off > off_max {
                        off_max = off;
                    }
                    if gamma.abs() <= T::EPSILON * scale {
                        continue;
                    }
                    // Rotation that makes columns p and q orthogonal.
                    let zeta = (beta - alpha) / (T::TWO * gamma);
                    let sign = if zeta < T::ZERO { -T::ONE } else { T::ONE };
                    let t = sign / (zeta.abs() + (T::ONE + zeta * zeta).sqrt());
                    let c = T::ONE / (T::ONE + t * t).sqrt();
                    let s = c * t;
                    for i in 0..M {
                        let up = u[(i, p)];
                        let uq = u[(i, q)];
                        u[(i, p)] = c * up - s * uq;
                        u[(i, q)] = s * up + c * uq;
                    }
                    for i in 0..N {
                        let vp = v[(i, p)];
                        let vq = v[(i, q)];
                        v[(i, p)] = c * vp - s * vq;
                        v[(i, q)] = s * vp + c * vq;
                    }
                }
            }
            if off_max <= T::EPSILON {
                break;
            }
        }

        // The column norms are the singular values; normalize U's columns by them.
        let mut singular_values = Vector::<N, T>::zeros();
        for k in 0..N {
            let sigma = u.column(k).norm();
            singular_values[k] = sigma;
            if sigma > T::ZERO {
                for i in 0..M {
                    u[(i, k)] /= sigma;
                }
            }
        }

        // Sort the singular values descending, carrying the matching U and V columns.
        for k in 0..N {
            let mut top = k;
            for j in (k + 1)..N {
                if singular_values[j] > singular_values[top] {
                    top = j;
                }
            }
            if top != k {
                let tmp = singular_values[k];
                singular_values[k] = singular_values[top];
                singular_values[top] = tmp;
                for i in 0..M {
                    let tmp = u[(i, k)];
                    u[(i, k)] = u[(i, top)];
                    u[(i, top)] = tmp;
                }
                for i in 0..N {
                    let tmp = v[(i, k)];
                    v[(i, k)] = v[(i, top)];
                    v[(i, top)] = tmp;
                }
            }
        }

        // Sign convention: the largest-magnitude entry of each U column is positive.
        for k in 0..N {
            let mut row = 0;
            let mut best = T::ZERO;
            for i in 0..M {
                let mag = u[(i, k)].abs();
                if mag > best {
                    best = mag;
                    row = i;
                }
            }
            if u[(row, k)] < T::ZERO {
                for i in 0..M {
                    u[(i, k)] = -u[(i, k)];
                }
                for i in 0..N {
                    v[(i, k)] = -v[(i, k)];
                }
            }
        }

        Ok(Svd {
            u,
            singular_values,
            v,
        })
    }

    /// The Moore–Penrose pseudo-inverse of `self`, for any shape.
    ///
    /// Tall or square inputs go straight through [`Matrix::svd`]; a wide input (`M < N`) is handled
    /// as `((Aᵀ)⁺)ᵀ`. Returns [`LinalgError::NonFinite`] if any entry is not finite.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// // A wide matrix, handled through the transpose route.
    /// let a = Matrix::<2, 3>::new([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]]);
    /// let pinv = a.pseudo_inverse().unwrap();
    /// let recon = a * pinv * a; // A·A⁺·A == A
    /// for r in 0..2 {
    ///     for c in 0..3 {
    ///         assert!((recon[(r, c)] - a[(r, c)]).abs() < 1e-12);
    ///     }
    /// }
    /// ```
    pub fn pseudo_inverse(self) -> Result<Matrix<N, M, T>, LinalgError> {
        if M >= N {
            Ok(self.svd()?.pseudo_inverse())
        } else {
            Ok(self.transpose().svd()?.pseudo_inverse().transpose())
        }
    }
}

impl<const M: usize, const N: usize, T: Numeric> Svd<M, N, T> {
    /// The singular values, descending and non-negative.
    pub fn singular_values(&self) -> Vector<N, T> {
        self.singular_values
    }

    /// The left factor `U`, with orthonormal columns.
    pub fn u(&self) -> Matrix<M, N, T> {
        self.u
    }

    /// The right factor `V`, with orthonormal columns.
    pub fn v(&self) -> Matrix<N, N, T> {
        self.v
    }

    /// The number of singular values greater than `tol`.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// // Column 2 is twice column 1, so the matrix has rank 1.
    /// let a = Matrix::<3, 2>::new([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]);
    /// assert_eq!(a.svd().unwrap().rank(1e-9), 1);
    /// ```
    #[inline]
    #[must_use]
    pub fn rank(&self, tol: T) -> usize {
        let mut count = 0;
        for k in 0..N {
            if self.singular_values[k] > tol {
                count += 1;
            }
        }
        count
    }

    /// The ratio `σ_max / σ_min`, or infinity when the smallest singular value is zero.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let a = Matrix::<2, 2>::new([[2.0, 0.0], [0.0, 1.0]]);
    /// assert!((a.svd().unwrap().condition_number() - 2.0).abs() < 1e-12);
    /// ```
    #[inline]
    #[must_use]
    pub fn condition_number(&self) -> T {
        if N == 0 {
            return T::INFINITY;
        }
        let smallest = self.singular_values[N - 1];
        if smallest <= T::ZERO {
            T::INFINITY
        } else {
            self.singular_values[0] / smallest
        }
    }

    /// The default cutoff below which a singular value counts as zero.
    fn default_tol(&self) -> T {
        if N == 0 {
            return T::ZERO;
        }
        T::from_usize(M.max(N)) * T::EPSILON * self.singular_values[0]
    }

    /// The Moore–Penrose pseudo-inverse `V · Σ⁺ · Uᵀ`, dropping singular values `<= tol`.
    pub fn pseudo_inverse_tol(&self, tol: T) -> Matrix<N, M, T> {
        Matrix::from_fn(|i, j| {
            let mut acc = T::ZERO;
            for k in 0..N {
                let sigma = self.singular_values[k];
                if sigma > tol {
                    acc += self.v[(i, k)] * self.u[(j, k)] / sigma;
                }
            }
            acc
        })
    }

    /// The Moore–Penrose pseudo-inverse, using a default cutoff from the largest singular value.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let a = Matrix::<3, 2>::new([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    /// let pinv = a.svd().unwrap().pseudo_inverse();
    /// let recon = a * pinv * a; // A·A⁺·A == A
    /// for r in 0..3 {
    ///     for c in 0..2 {
    ///         assert!((recon[(r, c)] - a[(r, c)]).abs() < 1e-12);
    ///     }
    /// }
    /// ```
    pub fn pseudo_inverse(&self) -> Matrix<N, M, T> {
        self.pseudo_inverse_tol(self.default_tol())
    }

    /// The minimum-norm least-squares solution of `A·x = b`, from `V · Σ⁺ · Uᵀ · b`.
    ///
    /// The pseudo-inverse is never formed. Singular values `<= tol` are dropped.
    ///
    /// ```
    /// use multicalc::linear_algebra::{Matrix, Vector};
    /// // Overdetermined and consistent: the exact solution is x = [1, 2].
    /// let a = Matrix::<3, 2>::new([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);
    /// let x = a.svd().unwrap().solve(Vector::new([1.0, 2.0, 3.0]));
    /// assert!((x[0] - 1.0).abs() < 1e-12);
    /// assert!((x[1] - 2.0).abs() < 1e-12);
    /// ```
    pub fn solve(&self, b: Vector<M, T>) -> Vector<N, T> {
        let tol = self.default_tol();
        let mut z = Vector::<N, T>::zeros();
        for k in 0..N {
            let sigma = self.singular_values[k];
            if sigma > tol {
                z[k] = self.u.column(k).dot(b) / sigma;
            }
        }
        self.v * z
    }
}
