//! Discretization of continuous-time linear systems. All build on [`Matrix::expm`].
//!
//! - [`zoh`] — exact zero-order hold of `ẋ = A x + B u`.
//! - [`van_loan`] — Van Loan discrete process-noise covariance.
//! - [`q_discrete_white_noise`] — the piecewise-constant white-noise model.

use crate::error::LinalgError;
use crate::linear_algebra::Matrix;
use crate::scalar::Numeric;

/// Exact zero-order-hold discretization of `ẋ = A x + B u` over step `timestep`: returns `(F, G)` with
/// `F = expm(A·timestep)` and `G = ∫₀^timestep expm(A·τ) dτ · B`, via the augmented-matrix exponential.
///
/// `NM` MUST equal `N + M`; a mismatch is a compile error.
/// Returns [`LinalgError::InvalidTimestep`] if `timestep` is negative or non-finite.
///
/// ```
/// use multicalc::linear_algebra::{Matrix, Matrix2D};
/// use multicalc::discretization::zoh;
/// # fn main() -> Result<(), multicalc::error::LinalgError> {
/// // Double integrator: A = [[0,1],[0,0]], B = [[0],[1]].
/// let a = Matrix2D::new([[0.0, 1.0], [0.0, 0.0]]);
/// let b = Matrix::<2, 1>::new([[0.0], [1.0]]);
/// let timestep = 0.1;
///
/// let (f, g) = zoh::<2, 1, 3, f64>(a, b, timestep)?;
/// assert!((f[(0, 1)] - 0.1).abs() < 1e-9); // F = [[1, timestep], [0, 1]]
/// assert!((g[(0, 0)] - 0.005).abs() < 1e-9); // G = [[timestep²/2], [timestep]]
/// # Ok(())
/// # }
/// ```
pub fn zoh<const N: usize, const M: usize, const NM: usize, T: Numeric>(
    a: Matrix<N, N, T>,
    b: Matrix<N, M, T>,
    timestep: T,
) -> Result<(Matrix<N, N, T>, Matrix<N, M, T>), LinalgError> {
    const { assert!(NM == N + M, "zoh: NM must equal N + M") };
    if !timestep.is_finite() || timestep < T::ZERO {
        return Err(LinalgError::InvalidTimestep);
    }
    // Augmented [[A, B], [0, 0]] · timestep; its exponential's top blocks are F and G.
    let aug = Matrix::<NM, NM, T>::from_fn(|i, j| {
        if i < N && j < N {
            a[(i, j)] * timestep
        } else if i < N {
            b[(i, j - N)] * timestep
        } else {
            T::ZERO
        }
    });
    let exponential = aug.expm()?;
    let f = Matrix::<N, N, T>::from_fn(|i, j| exponential[(i, j)]);
    let input_matrix = Matrix::<N, M, T>::from_fn(|i, j| exponential[(i, N + j)]);
    Ok((f, input_matrix))
}

/// Van Loan (1978) discretization of the continuous system `(A, Q_c)`: returns `(F, Q_d)` with
/// `F = expm(A·timestep)` and `Q_d` the discrete process-noise covariance.
///
/// `N2` MUST equal `2·N`; a mismatch is a compile error.
/// Returns [`LinalgError::InvalidTimestep`] if `timestep` is negative or non-finite.
///
/// ```
/// use multicalc::linear_algebra::Matrix2D;
/// use multicalc::discretization::van_loan;
/// # fn main() -> Result<(), multicalc::error::LinalgError> {
/// let a = Matrix2D::new([[0.0, 1.0], [0.0, 0.0]]);
/// let qc = Matrix2D::new([[0.0, 0.0], [0.0, 1.0]]);
/// let (_f, qd) = van_loan::<2, 4, f64>(a, qc, 0.1)?;
/// assert!((qd[(0, 1)] - qd[(1, 0)]).abs() < 1e-12); // symmetric
/// # Ok(())
/// # }
/// ```
pub fn van_loan<const N: usize, const N2: usize, T: Numeric>(
    a: Matrix<N, N, T>,
    process_noise: Matrix<N, N, T>,
    timestep: T,
) -> Result<(Matrix<N, N, T>, Matrix<N, N, T>), LinalgError> {
    const { assert!(N2 == 2 * N, "van_loan: N2 must equal 2*N") };
    if !timestep.is_finite() || timestep < T::ZERO {
        return Err(LinalgError::InvalidTimestep);
    }
    // Ξ = [[-A, process_noise], [0, Aᵀ]] · timestep. From expm(Ξ) = [[.., G12], [0, G22]]: F = G22ᵀ, Q_d = F · G12.
    let augmented = Matrix::<N2, N2, T>::from_fn(|i, j| {
        if i < N && j < N {
            -a[(i, j)] * timestep
        } else if i < N {
            process_noise[(i, j - N)] * timestep
        } else if j >= N {
            a[(j - N, i - N)] * timestep // (Aᵀ)[i-N, j-N] = A[j-N, i-N]
        } else {
            T::ZERO
        }
    });
    let expm_result = augmented.expm()?;
    let g12 = Matrix::<N, N, T>::from_fn(|i, j| expm_result[(i, N + j)]);
    let g22 = Matrix::<N, N, T>::from_fn(|i, j| expm_result[(N + i, N + j)]);
    let f = g22.transpose();
    let discrete_q = f * g12;
    Ok((f, discrete_q))
}

/// The filterpy-compatible discrete white-noise covariance for a Newtonian integrator chain of
/// `DIM` states (`DIM ∈ {2, 3, 4}`). Closed form; no matrix exponential. `variance` is the
/// continuous white-noise spectral intensity (filterpy's `var`).
///
/// `DIM` MUST be 2, 3, or 4; anything else is a compile error.
///
/// ```
/// use multicalc::discretization::q_discrete_white_noise;
/// let timestep = 0.1;
/// let variance = 2.0;
///
/// let q = q_discrete_white_noise::<2, f64>(timestep, variance);
/// assert!((q[(1, 1)] - 2.0 * 0.1 * 0.1).abs() < 1e-15); // variance · timestep²
/// ```
pub fn q_discrete_white_noise<const DIM: usize, T: Numeric>(
    timestep: T,
    variance: T,
) -> Matrix<DIM, DIM, T> {
    const {
        assert!(
            DIM >= 2 && DIM <= 4,
            "q_discrete_white_noise: DIM must be 2, 3, or 4"
        )
    };
    let timestep2 = timestep * timestep;
    let timestep3 = timestep2 * timestep;
    let timestep4 = timestep3 * timestep;
    let timestep5 = timestep4 * timestep;
    let timestep6 = timestep5 * timestep;
    // Variable indices keep this legal for every DIM (constant out-of-range indexing would be a
    // compile error). Entries match filterpy's Q_discrete_white_noise; the matrix is symmetric.
    Matrix::from_fn(|i, j| {
        let entry = match (DIM, i, j) {
            (2, 0, 0) => timestep4 * T::from_f64(0.25),
            (2, 0, 1) | (2, 1, 0) => timestep3 * T::HALF,
            (2, 1, 1) => timestep2,
            (3, 0, 0) => timestep4 * T::from_f64(0.25),
            (3, 0, 1) | (3, 1, 0) => timestep3 * T::HALF,
            (3, 0, 2) | (3, 2, 0) => timestep2 * T::HALF,
            (3, 1, 1) => timestep2,
            (3, 1, 2) | (3, 2, 1) => timestep,
            (3, 2, 2) => T::ONE,
            (4, 0, 0) => timestep6 / T::from_f64(36.0),
            (4, 0, 1) | (4, 1, 0) => timestep5 / T::from_f64(12.0),
            (4, 0, 2) | (4, 2, 0) => timestep4 / T::from_f64(6.0),
            (4, 0, 3) | (4, 3, 0) => timestep3 / T::from_f64(6.0),
            (4, 1, 1) => timestep4 * T::from_f64(0.25),
            (4, 1, 2) | (4, 2, 1) => timestep3 * T::HALF,
            (4, 1, 3) | (4, 3, 1) => timestep2 * T::HALF,
            (4, 2, 2) => timestep2,
            (4, 2, 3) | (4, 3, 2) => timestep,
            (4, 3, 3) => T::ONE,
            _ => T::ZERO,
        };
        entry * variance
    })
}
