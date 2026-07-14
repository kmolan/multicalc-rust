//! Discretization of continuous-time linear systems: zero-order hold, Van Loan process-noise
//! discretization, and the piecewise-constant white-noise model. All build on [`Matrix::expm`].

use crate::error::LinalgError;
use crate::linear_algebra::Matrix;
use crate::scalar::Numeric;

/// Exact zero-order-hold discretization of `ẋ = A x + B u` over step `dt`: returns `(F, G)` with
/// `F = expm(A·dt)` and `G = ∫₀^dt expm(A·τ) dτ · B`, via the augmented-matrix exponential.
///
/// `NM` MUST equal `N + M`; a mismatch is a compile error.
///
/// ```
/// use multicalc::linear_algebra::Matrix;
/// use multicalc::discretization::zoh;
/// # fn main() -> Result<(), multicalc::error::LinalgError> {
/// // Double integrator: A = [[0,1],[0,0]], B = [[0],[1]].
/// let a = Matrix::<2, 2>::new([[0.0, 1.0], [0.0, 0.0]]);
/// let b = Matrix::<2, 1>::new([[0.0], [1.0]]);
/// let (f, g) = zoh::<2, 1, 3, f64>(a, b, 0.1)?;
/// assert!((f[(0, 1)] - 0.1).abs() < 1e-9); // F = [[1, dt], [0, 1]]
/// assert!((g[(0, 0)] - 0.005).abs() < 1e-9); // G = [[dt²/2], [dt]]
/// # Ok(())
/// # }
/// ```
pub fn zoh<const N: usize, const M: usize, const NM: usize, T: Numeric>(
    a: Matrix<N, N, T>,
    b: Matrix<N, M, T>,
    dt: T,
) -> Result<(Matrix<N, N, T>, Matrix<N, M, T>), LinalgError> {
    const { assert!(NM == N + M, "zoh: NM must equal N + M") };
    // Augmented [[A, B], [0, 0]] · dt; its exponential's top blocks are F and G.
    let aug = Matrix::<NM, NM, T>::from_fn(|i, j| {
        if i < N && j < N {
            a[(i, j)] * dt
        } else if i < N {
            b[(i, j - N)] * dt
        } else {
            T::ZERO
        }
    });
    let e = aug.expm()?;
    let f = Matrix::<N, N, T>::from_fn(|i, j| e[(i, j)]);
    let g = Matrix::<N, M, T>::from_fn(|i, j| e[(i, N + j)]);
    Ok((f, g))
}

/// Van Loan (1978) discretization of the continuous system `(A, Q_c)`: returns `(F, Q_d)` with
/// `F = expm(A·dt)` and `Q_d` the discrete process-noise covariance.
///
/// `N2` MUST equal `2·N`; a mismatch is a compile error.
///
/// ```
/// use multicalc::linear_algebra::Matrix;
/// use multicalc::discretization::van_loan;
/// # fn main() -> Result<(), multicalc::error::LinalgError> {
/// let a = Matrix::<2, 2>::new([[0.0, 1.0], [0.0, 0.0]]);
/// let qc = Matrix::<2, 2>::new([[0.0, 0.0], [0.0, 1.0]]);
/// let (_f, qd) = van_loan::<2, 4, f64>(a, qc, 0.1)?;
/// assert!((qd[(0, 1)] - qd[(1, 0)]).abs() < 1e-12); // symmetric
/// # Ok(())
/// # }
/// ```
pub fn van_loan<const N: usize, const N2: usize, T: Numeric>(
    a: Matrix<N, N, T>,
    qc: Matrix<N, N, T>,
    dt: T,
) -> Result<(Matrix<N, N, T>, Matrix<N, N, T>), LinalgError> {
    const { assert!(N2 == 2 * N, "van_loan: N2 must equal 2*N") };
    // Ξ = [[-A, Q_c], [0, Aᵀ]] · dt. From expm(Ξ) = [[.., G12], [0, G22]]: F = G22ᵀ, Q_d = F · G12.
    let xi = Matrix::<N2, N2, T>::from_fn(|i, j| {
        if i < N && j < N {
            -a[(i, j)] * dt
        } else if i < N {
            qc[(i, j - N)] * dt
        } else if j >= N {
            a[(j - N, i - N)] * dt // (Aᵀ)[i-N, j-N] = A[j-N, i-N]
        } else {
            T::ZERO
        }
    });
    let e = xi.expm()?;
    let g12 = Matrix::<N, N, T>::from_fn(|i, j| e[(i, N + j)]);
    let g22 = Matrix::<N, N, T>::from_fn(|i, j| e[(N + i, N + j)]);
    let f = g22.transpose();
    let qd = f * g12;
    Ok((f, qd))
}

/// The filterpy-compatible discrete white-noise covariance for a Newtonian integrator chain of
/// `DIM` states (`DIM ∈ {2, 3, 4}`). Closed form; no matrix exponential. `variance` is the
/// continuous white-noise spectral intensity (filterpy's `var`).
///
/// `DIM` MUST be 2, 3, or 4; anything else is a compile error.
///
/// ```
/// use multicalc::discretization::q_discrete_white_noise;
/// let q = q_discrete_white_noise::<2, f64>(0.1, 2.0);
/// assert!((q[(1, 1)] - 2.0 * 0.1 * 0.1).abs() < 1e-15); // var · dt²
/// ```
pub fn q_discrete_white_noise<const DIM: usize, T: Numeric>(dt: T, variance: T) -> Matrix<DIM, DIM, T> {
    const { assert!(DIM >= 2 && DIM <= 4, "q_discrete_white_noise: DIM must be 2, 3, or 4") };
    let dt2 = dt * dt;
    let dt3 = dt2 * dt;
    let dt4 = dt3 * dt;
    let dt5 = dt4 * dt;
    let dt6 = dt5 * dt;
    // Variable indices keep this legal for every DIM (constant out-of-range indexing would be a
    // compile error). Entries match filterpy's Q_discrete_white_noise; the matrix is symmetric.
    Matrix::from_fn(|i, j| {
        let c = match (DIM, i, j) {
            (2, 0, 0) => dt4 * T::from_f64(0.25),
            (2, 0, 1) | (2, 1, 0) => dt3 * T::HALF,
            (2, 1, 1) => dt2,
            (3, 0, 0) => dt4 * T::from_f64(0.25),
            (3, 0, 1) | (3, 1, 0) => dt3 * T::HALF,
            (3, 0, 2) | (3, 2, 0) => dt2 * T::HALF,
            (3, 1, 1) => dt2,
            (3, 1, 2) | (3, 2, 1) => dt,
            (3, 2, 2) => T::ONE,
            (4, 0, 0) => dt6 / T::from_f64(36.0),
            (4, 0, 1) | (4, 1, 0) => dt5 / T::from_f64(12.0),
            (4, 0, 2) | (4, 2, 0) => dt4 / T::from_f64(6.0),
            (4, 0, 3) | (4, 3, 0) => dt3 / T::from_f64(6.0),
            (4, 1, 1) => dt4 * T::from_f64(0.25),
            (4, 1, 2) | (4, 2, 1) => dt3 * T::HALF,
            (4, 1, 3) | (4, 3, 1) => dt2 * T::HALF,
            (4, 2, 2) => dt2,
            (4, 2, 3) | (4, 3, 2) => dt,
            (4, 3, 3) => T::ONE,
            _ => T::ZERO,
        };
        c * variance
    })
}
