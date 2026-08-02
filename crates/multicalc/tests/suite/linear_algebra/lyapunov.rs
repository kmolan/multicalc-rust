//! Lyapunov tests: the series answer on a scalar, the equation satisfied on larger systems,
//! rejection of a system that grows, and rejection of a lopsided cost.

use multicalc::error::LinalgError;
use multicalc::linear_algebra::{Matrix, solve_discrete_lyapunov};

/// The 3×3 case the middle tests share: every direction shrinks, and the cost reads the same
/// across the diagonal.
fn shrinking_3x3() -> (Matrix<3, 3>, Matrix<3, 3>) {
    let a = Matrix::<3, 3>::new([[0.5, 0.1, 0.0], [0.0, 0.4, 0.2], [0.1, 0.0, 0.3]]);
    let q = Matrix::<3, 3>::new([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.0]]);
    (a, q)
}

#[test]
fn scalar_matches_the_closed_form() {
    // Keeping half of the state each step gives 1 + 1/4 + 1/16 + ... = 4/3.
    let a = Matrix::<1, 1>::new([[0.5]]);
    let q = Matrix::<1, 1>::new([[1.0]]);
    let p = solve_discrete_lyapunov(a, q).unwrap();
    assert!((p[(0, 0)] - 4.0 / 3.0).abs() < 1e-12);

    // In general a single state settles at q / (1 - a²).
    let a = Matrix::<1, 1>::new([[0.9]]);
    let q = Matrix::<1, 1>::new([[2.0]]);
    let p = solve_discrete_lyapunov(a, q).unwrap();
    assert!((p[(0, 0)] - 2.0 / (1.0 - 0.81)).abs() < 1e-10);
}

#[test]
fn satisfies_the_equation_3x3() {
    let (a, q) = shrinking_3x3();
    let p = solve_discrete_lyapunov(a, q).unwrap();
    let residual = a.transpose() * p * a - p + q;
    assert!(residual.frobenius_norm() < 1e-12);
}

#[test]
fn answer_is_positive_definite() {
    let (a, q) = shrinking_3x3();
    let p = solve_discrete_lyapunov(a, q).unwrap();
    assert!(p.cholesky().is_ok());
}

#[test]
fn answer_reads_the_same_across_the_diagonal() {
    let (a, q) = shrinking_3x3();
    let p = solve_discrete_lyapunov(a, q).unwrap();
    for row in 0..3 {
        for column in (row + 1)..3 {
            assert!((p[(row, column)] - p[(column, row)]).abs() < 1e-14);
        }
    }
}

#[test]
fn rejects_a_system_that_grows() {
    // The first direction grows by a fifth each step, so the series never adds up.
    let a = Matrix::<2, 2>::new([[1.2, 0.0], [0.0, 0.5]]);
    let q = Matrix::<2, 2>::identity();
    assert!(matches!(
        solve_discrete_lyapunov(a, q),
        Err(LinalgError::DidNotConverge { .. })
    ));
}

#[test]
fn rejects_a_lopsided_cost() {
    let a = Matrix::<2, 2>::new([[0.5, 0.0], [0.0, 0.5]]);
    let q = Matrix::<2, 2>::new([[1.0, 0.5], [-0.5, 1.0]]);
    assert_eq!(
        solve_discrete_lyapunov(a, q).err(),
        Some(LinalgError::NotSymmetric)
    );
}

#[test]
fn rejects_non_finite() {
    let a = Matrix::<2, 2>::new([[f64::NAN, 0.0], [0.0, 0.5]]);
    let q = Matrix::<2, 2>::identity();
    assert_eq!(
        solve_discrete_lyapunov(a, q).err(),
        Some(LinalgError::NonFinite)
    );
}

#[test]
fn works_at_f32() {
    let a = Matrix::<3, 3, f32>::new([[0.5, 0.1, 0.0], [0.0, 0.4, 0.2], [0.1, 0.0, 0.3]]);
    let q = Matrix::<3, 3, f32>::new([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.0]]);
    let p = solve_discrete_lyapunov(a, q).unwrap();
    let residual = a.transpose() * p * a - p + q;
    assert!(residual.frobenius_norm() < 1e-5_f32);
}
