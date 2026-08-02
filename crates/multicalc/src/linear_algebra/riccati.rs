//! Solves the steady-state discrete Riccati equation, the cost-to-go behind an optimal linear
//! feedback law.
//!
//! The equation is `P = Aᵀ·P·A − Aᵀ·P·B·(R + Bᵀ·P·B)⁻¹·Bᵀ·P·A + Q`. It is solved by structured
//! doubling: each pass folds two steps of the underlying recursion into one, so the number of
//! steps covered doubles every pass and a handful of passes reaches the steady state. The answer
//! is put back into the equation before it is returned, so a pass count that ran out or a
//! near-singular intermediate is caught rather than returned as a plausible-looking matrix.
//! Reference values for the tests come from SciPy.

use crate::error::LinalgError;
use crate::linear_algebra::Matrix;
use crate::scalar::Numeric;

/// How many doubling passes to allow. Each pass doubles how many steps of the underlying
/// recursion the answer covers, so this is far more than a well-posed problem needs.
const MAXIMUM_PASSES: usize = 64;

/// Forces a matrix to read the same across the diagonal by averaging each pair, so rounding
/// cannot tilt it as the passes multiply it out.
fn symmetrize<const K: usize, T: Numeric>(m: &mut Matrix<K, K, T>) {
    for row in 0..K {
        for column in (row + 1)..K {
            let averaged = (m[(row, column)] + m[(column, row)]) * T::HALF;
            m[(row, column)] = averaged;
            m[(column, row)] = averaged;
        }
    }
}

/// Finds the steady-state `P` of `P = Aᵀ·P·A − Aᵀ·P·B·(R + Bᵀ·P·B)⁻¹·Bᵀ·P·A + Q`.
///
/// `A` and `B` describe how the state moves and how the input pushes it; `Q` and `R` say how much
/// state error and input effort each cost. The answer is what an optimal linear feedback law is
/// built from. The caller has to supply a system whose unstable directions can be reached by the
/// input and whose costly directions are visible in `Q`; without that there is no steady answer
/// and the solver reports that it did not settle.
///
/// Costs `O(n³)` per pass with a budget of 64 passes, so run it once at design time rather than
/// inside a control loop.
///
/// Returns [`LinalgError::NonFinite`](crate::error::LinalgError::NonFinite) if any entry is not
/// finite, [`LinalgError::NotSymmetric`](crate::error::LinalgError::NotSymmetric) if `q` or `r`
/// does not read the same across the diagonal,
/// [`LinalgError::NotPositiveDefinite`](crate::error::LinalgError::NotPositiveDefinite) if `r` has
/// no Cholesky factor, [`LinalgError::Singular`](crate::error::LinalgError::Singular) if an
/// intermediate cannot be inverted, or
/// [`LinalgError::DidNotConverge`](crate::error::LinalgError::DidNotConverge) if the passes run out
/// or the answer fails the check against the equation.
///
/// ```
/// use multicalc::linear_algebra::{Matrix, solve_discrete_riccati};
///
/// // One state that holds its value, one input that adds to it, unit costs. The equation
/// // reduces to p = p - p²/(1 + p) + 1, whose positive root is the golden ratio.
/// let a = Matrix::<1, 1>::new([[1.0]]);
/// let b = Matrix::<1, 1>::new([[1.0]]);
/// let q = Matrix::<1, 1>::new([[1.0]]);
/// let r = Matrix::<1, 1>::new([[1.0]]);
/// let p = solve_discrete_riccati(a, b, q, r).unwrap();
/// let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
/// assert!((p[(0, 0)] - golden_ratio).abs() < 1e-10);
/// ```
pub fn solve_discrete_riccati<const N: usize, const M: usize, T: Numeric>(
    a: Matrix<N, N, T>,
    b: Matrix<N, M, T>,
    q: Matrix<N, N, T>,
    r: Matrix<M, M, T>,
) -> Result<Matrix<N, N, T>, LinalgError> {
    if !a.is_finite() || !b.is_finite() || !q.is_finite() || !r.is_finite() {
        return Err(LinalgError::NonFinite);
    }
    if !q.is_symmetric() || !r.is_symmetric() {
        return Err(LinalgError::NotSymmetric);
    }

    // How far the input can push the state, per unit of input cost: `B·R⁻¹·Bᵀ`, formed by solving
    // against the factor of `R` rather than by inverting it.
    let input_cost = r.cholesky()?;
    let scaled_input = input_cost.solve_matrix::<N>(b.transpose());
    let mut reach = b * scaled_input;

    let mut state = a;
    let mut cost = q;
    let mut passes_taken = MAXIMUM_PASSES;
    for pass in 0..MAXIMUM_PASSES {
        let coupling = (Matrix::<N, N, T>::identity() + reach * cost).inverse()?;
        let folded = coupling * state;
        let cost_increment = state.transpose() * cost * folded;
        let reach_increment = state * (coupling * reach) * state.transpose();

        let next_state = state * folded;
        let mut next_cost = cost + cost_increment;
        let mut next_reach = reach + reach_increment;
        symmetrize(&mut next_cost);
        symmetrize(&mut next_reach);

        state = next_state;
        cost = next_cost;
        reach = next_reach;

        // Squaring the entries to size them up overflows before the entries themselves do, so a
        // size that has run off the end of what a number can hold is checked for as well: it means
        // the passes are running away rather than settling.
        let increment_size = cost_increment.frobenius_norm();
        let cost_size = cost.frobenius_norm();
        if !cost.is_finite()
            || !reach.is_finite()
            || !state.is_finite()
            || !increment_size.is_finite()
            || !cost_size.is_finite()
        {
            return Err(LinalgError::DidNotConverge { iters: pass + 1 });
        }
        if increment_size <= T::EPSILON_X30 * cost_size.max(T::ONE) {
            passes_taken = pass + 1;
            break;
        }
    }

    // Put the answer back into the equation and require what is left over to be small.
    let input_weight = r + b.transpose() * cost * b;
    let input_weight_factor = input_weight.cholesky()?;
    let coupling_term = b.transpose() * cost * a;
    let correction =
        coupling_term.transpose() * input_weight_factor.solve_matrix::<N>(coupling_term);
    let residual = a.transpose() * cost * a - cost - correction + q;

    // Deliberately looser than the settling test the passes use: the leftover is built from
    // several matrix products, so it carries far more rounding than the answer itself, and a
    // tighter bound would turn away correct answers on badly scaled systems.
    let allowed = T::EPSILON.sqrt() * cost.frobenius_norm().max(T::ONE);
    if !residual.is_finite() || residual.frobenius_norm() > allowed {
        return Err(LinalgError::DidNotConverge {
            iters: passes_taken,
        });
    }
    Ok(cost)
}
