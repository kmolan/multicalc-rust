//! Solves `Aᵀ·P·A − P + Q = 0` for `P`, by adding the terms of its series in doubling blocks.
//!
//! The answer is the sum `P = Q + Aᵀ·Q·A + (Aᵀ)²·Q·A² + …`, which only adds up to something finite
//! when repeated application of `A` shrinks every direction. Squaring `A` on each pass doubles how
//! many terms the running total covers, so a handful of passes covers an enormous number of terms.
//! When `A` does not shrink, the running total grows without bound instead, and the solver says so
//! rather than returning a number. Reference values for the tests come from SciPy.

use crate::error::LinalgError;
use crate::linear_algebra::Matrix;
use crate::scalar::Numeric;

/// How many doubling passes to allow. Each pass doubles how many terms the total covers, so this
/// is far more than any settling problem needs; reaching it means the total is not settling.
const MAXIMUM_PASSES: usize = 64;

/// Finds the `P` that satisfies `Aᵀ·P·A − P + Q = 0`, given a `Q` that reads the same across the
/// diagonal.
///
/// This is the standard way to certify that a closed loop settles: a solution exists only when
/// repeated application of `A` shrinks every direction, so
/// [`LinalgError::DidNotConverge`](crate::error::LinalgError::DidNotConverge) is the verdict that
/// it does not, not a numerical failure. Costs `O(n³)` per pass with a budget of 64 passes, so run
/// it once at design time rather than inside a control loop.
///
/// Returns [`LinalgError::NonFinite`](crate::error::LinalgError::NonFinite) if any entry is not
/// finite, [`LinalgError::NotSymmetric`](crate::error::LinalgError::NotSymmetric) if `state_cost` does not
/// read the same across the diagonal, or
/// [`LinalgError::DidNotConverge`](crate::error::LinalgError::DidNotConverge) if the total has not
/// settled within the budget.
///
/// ```
/// use multicalc::linear_algebra::{Matrix, solve_discrete_lyapunov};
///
/// // A single state that keeps half of itself each step, with Q = 1. The series is
/// // 1 + 1/4 + 1/16 + ... = 4/3.
/// let a = Matrix::<1, 1>::new([[0.5]]);
/// let state_cost = Matrix::<1, 1>::new([[1.0]]);
/// let cost_to_go = solve_discrete_lyapunov(a, state_cost).unwrap();
/// assert!((cost_to_go[(0, 0)] - 4.0 / 3.0).abs() < 1e-12);
///
/// // A state that grows has no answer.
/// let unstable = Matrix::<1, 1>::new([[1.5]]);
/// assert!(solve_discrete_lyapunov(unstable, state_cost).is_err());
/// ```
pub fn solve_discrete_lyapunov<const N: usize, T: Numeric>(
    a: Matrix<N, N, T>,
    state_cost: Matrix<N, N, T>,
) -> Result<Matrix<N, N, T>, LinalgError> {
    if !a.is_finite() || !state_cost.is_finite() {
        return Err(LinalgError::NonFinite);
    }

    if !state_cost.is_symmetric() {
        return Err(LinalgError::NotSymmetric);
    }

    let mut total = state_cost;
    let mut power = a;
    for pass in 0..MAXIMUM_PASSES {
        let increment = power.transpose() * total * power;
        let mut next = total + increment;
        // The exact answer reads the same across the diagonal; forcing it back keeps rounding
        // from tilting the running total as the passes multiply it out.
        for row in 0..N {
            for column in (row + 1)..N {
                let averaged = (next[(row, column)] + next[(column, row)]) * T::HALF;
                next[(row, column)] = averaged;
                next[(column, row)] = averaged;
            }
        }
        total = next;
        // Squaring the entries to size them up overflows before the entries themselves do, so a
        // size that has run off the end of what a number can hold is checked for as well: it means
        // the total is growing away, which is the same verdict.
        let increment_size = increment.frobenius_norm();
        let total_size = total.frobenius_norm();
        if !total.is_finite() || !increment_size.is_finite() || !total_size.is_finite() {
            return Err(LinalgError::DidNotConverge { iters: pass + 1 });
        }
        let settled = increment_size <= T::EPSILON_X30 * total_size.max(T::ONE);
        if settled {
            return Ok(total);
        }
        power = power * power;
    }

    Err(LinalgError::DidNotConverge {
        iters: MAXIMUM_PASSES,
    })
}
