//! Root finding for scalar equations and square systems.
//!
//! [`Bisection`] brackets a scalar root and halves the interval within a guaranteed budget.
//! [`Newton`] and [`NewtonSystem`] take Newton steps using a derivative from any
//! [`Derivator`](crate::numerical_derivative::derivator) (exact autodiff by default), each
//! with an optional backtracking line search. Every solver takes an iteration budget and
//! reports why it stopped as a [`RootTermination`].

pub mod bisection;
pub mod newton;
pub mod newton_system;

pub use bisection::Bisection;

use crate::scalar::Numeric;

/// Which convergence test stopped a root-finding solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RootTermination {
    /// The residual magnitude fell to or below the residual tolerance.
    ResidualTolerance,
    /// The step size fell to or below the step tolerance.
    StepTolerance,
    /// The bracket width fell to or below the step tolerance (bisection only).
    BracketWidth,
}

/// The outcome of a scalar root solve.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct RootReport<T = f64> {
    /// The final estimate of the root.
    pub root: T,
    /// The function value at the root estimate.
    pub residual: T,
    /// How many iterations ran.
    pub iterations: usize,
    /// Why the solver stopped.
    pub termination: RootTermination,
}

/// The outcome of a system root solve.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct RootReportN<const N: usize, T = f64> {
    /// The final estimate of the root.
    pub root: [T; N],
    /// The Euclidean norm of the residual at the root estimate.
    pub residual_norm: T,
    /// How many iterations ran.
    pub iterations: usize,
    /// Why the solver stopped.
    pub termination: RootTermination,
}

/// Returns `true` when `a` and `b` share a sign, treating zero as matching either.
///
/// Built from comparisons rather than multiplication so it is correct for infinities
/// and does not overflow.
pub(crate) fn same_sign<T: Numeric>(a: T, b: T) -> bool {
    (a >= T::ZERO) == (b >= T::ZERO)
}

/// Returns `true` when every element of `v` is finite.
pub(crate) fn all_finite<const K: usize, T: Numeric>(v: &[T; K]) -> bool {
    v.iter().all(|x| x.is_finite())
}
