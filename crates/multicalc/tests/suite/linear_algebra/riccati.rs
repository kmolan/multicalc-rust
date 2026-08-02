//! Riccati tests: the closed-form answer on a scalar, the equation satisfied on a double
//! integrator, rejection of costs that are lopsided or not positive, and an f32 pass.

use multicalc::error::LinalgError;
use multicalc::linear_algebra::{Matrix, solve_discrete_riccati};

/// A position and a velocity, pushed by an acceleration, over a tenth of a second. This is the
/// case most of the tests below run on.
fn double_integrator() -> (Matrix<2, 2>, Matrix<2, 1>, Matrix<2, 2>, Matrix<1, 1>) {
    let a = Matrix::<2, 2>::new([[1.0, 0.1], [0.0, 1.0]]);
    let b = Matrix::<2, 1>::new([[0.005], [0.1]]);
    let q = Matrix::<2, 2>::identity();
    let r = Matrix::<1, 1>::new([[1.0]]);
    (a, b, q, r)
}

#[test]
fn scalar_matches_the_closed_form() {
    // A state that holds its value with unit costs reduces to p = p - p²/(1 + p) + 1, whose
    // positive root is the golden ratio.
    let one = Matrix::<1, 1>::new([[1.0]]);
    let p = solve_discrete_riccati(one, one, one, one).unwrap();
    let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
    assert!((p[(0, 0)] - golden_ratio).abs() < 1e-10);
}

#[test]
fn satisfies_the_equation_double_integrator() {
    let (a, b, q, r) = double_integrator();
    let p = solve_discrete_riccati(a, b, q, r).unwrap();

    let weight = r + b.transpose() * p * b;
    let coupling = b.transpose() * p * a;
    let correction = coupling.transpose() * weight.cholesky().unwrap().solve_matrix::<2>(coupling);
    let residual = a.transpose() * p * a - p - correction + q;
    assert!(residual.frobenius_norm() < 1e-10);
}

#[test]
fn answer_is_positive_definite() {
    let (a, b, q, r) = double_integrator();
    let p = solve_discrete_riccati(a, b, q, r).unwrap();
    assert!(p.cholesky().is_ok());
}

#[test]
fn answer_reads_the_same_across_the_diagonal() {
    let (a, b, q, r) = double_integrator();
    let p = solve_discrete_riccati(a, b, q, r).unwrap();
    assert!((p[(0, 1)] - p[(1, 0)]).abs() < 1e-14);
}

#[test]
fn zero_state_cost_gives_a_zero_answer() {
    // Nothing is charged for where the state is, and it settles on its own, so the cheapest thing
    // to do is nothing.
    let a = Matrix::<2, 2>::new([[0.5, 0.0], [0.0, 0.5]]);
    let b = Matrix::<2, 1>::new([[0.0], [1.0]]);
    let p = solve_discrete_riccati(a, b, Matrix::zeros(), Matrix::identity()).unwrap();
    assert!(p.frobenius_norm() < 1e-12);
}

#[test]
fn rejects_a_non_positive_input_cost() {
    let (a, b, q, _) = double_integrator();
    let r = Matrix::<1, 1>::new([[0.0]]);
    assert_eq!(
        solve_discrete_riccati(a, b, q, r).err(),
        Some(LinalgError::NotPositiveDefinite)
    );
}

#[test]
fn rejects_a_lopsided_state_cost() {
    let (a, b, _, r) = double_integrator();
    let q = Matrix::<2, 2>::new([[1.0, 0.5], [-0.5, 1.0]]);
    assert_eq!(
        solve_discrete_riccati(a, b, q, r).err(),
        Some(LinalgError::NotSymmetric)
    );
}

#[test]
fn rejects_non_finite() {
    let (a, _, q, r) = double_integrator();
    let b = Matrix::<2, 1>::new([[f64::INFINITY], [0.1]]);
    assert_eq!(
        solve_discrete_riccati(a, b, q, r).err(),
        Some(LinalgError::NonFinite)
    );
}

#[test]
fn works_at_f32() {
    let a = Matrix::<2, 2, f32>::new([[1.0, 0.1], [0.0, 1.0]]);
    let b = Matrix::<2, 1, f32>::new([[0.005], [0.1]]);
    let q = Matrix::<2, 2, f32>::identity();
    let r = Matrix::<1, 1, f32>::new([[1.0]]);
    let p = solve_discrete_riccati(a, b, q, r).unwrap();

    let weight = r + b.transpose() * p * b;
    let coupling = b.transpose() * p * a;
    let correction = coupling.transpose() * weight.cholesky().unwrap().solve_matrix::<2>(coupling);
    let residual = a.transpose() * p * a - p - correction + q;
    assert!(residual.frobenius_norm() < 1e-4_f32);
}
