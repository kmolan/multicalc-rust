//! Singular value decomposition on this crate's own [`Vector`] and [`Matrix`] types.

use crate::linear_algebra::{Matrix, Vector};
use crate::scalar::Numeric;

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
}
