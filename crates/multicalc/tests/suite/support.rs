//! Small matrix helpers shared by more than one test file in this suite.

use multicalc::linear_algebra::Matrix;

/// Builds a symmetric positive-definite matrix from arbitrary entries as `M·Mᵀ`, ridged so the
/// factorization is well conditioned rather than merely non-singular.
pub fn symmetric_positive_definite<const N: usize>(entries: &[f64]) -> Matrix<N, N> {
    let factor = Matrix::<N, N>::from_fn(|row, column| entries[row * N + column]);
    factor * factor.transpose() + Matrix::<N, N>::identity().scale(0.25)
}

/// Sums the diagonal entries of a square matrix.
#[must_use]
pub fn trace<const N: usize>(matrix: Matrix<N, N>) -> f64 {
    (0..N).map(|index| matrix[(index, index)]).sum()
}
