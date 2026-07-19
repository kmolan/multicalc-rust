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
pub use newton::Newton;
pub use newton_system::NewtonSystem;

use crate::scalar::Numeric;
#[cfg(feature = "serde")]
use alloc::vec::Vec;

/// Which convergence test stopped a root-finding solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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






#[cfg(feature = "serde")]
impl<const N: usize, T: serde::Serialize> serde::Serialize for RootReportN<N, T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("RootReportN", 4)?;
        s.serialize_field("root", self.root.as_slice())?;
        s.serialize_field("residual_norm", &self.residual_norm)?;
        s.serialize_field("iterations", &self.iterations)?;
        s.serialize_field("termination", &self.termination)?;
        s.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, const N: usize, T: serde::Deserialize<'de> + Copy> serde::Deserialize<'de> for RootReportN<N, T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Helper<T> {
            root: Vec<T>,
            residual_norm: T,
            iterations: usize,
            termination: RootTermination,
        }
        let h = Helper::deserialize(deserializer)?;
        let root: [T; N] = h.root.try_into().map_err(|_| {
            serde::de::Error::custom("wrong number of elements in `root`")
        })?;
        Ok(RootReportN {
            root,
            residual_norm: h.residual_norm,
            iterations: h.iterations,
            termination: h.termination,
        })
    }
}