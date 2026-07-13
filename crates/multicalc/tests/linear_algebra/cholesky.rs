use crate::helpers::{assert_close, assert_identity, cholesky_reconstructs};
use multicalc::error::LinalgError;
use multicalc::linear_algebra::{Matrix, Vector};

#[test]
fn cholesky_reconstructs_spd() {
    cholesky_reconstructs(Matrix::<2, 2>::new([[4.0, 2.0], [2.0, 3.0]]), 1e-12);

    // A matrix with a known exact factor: L = [[2,0,0],[6,1,0],[-8,5,3]].
    let a = Matrix::<3, 3>::new([
        [4.0, 12.0, -16.0],
        [12.0, 37.0, -43.0],
        [-16.0, -43.0, 98.0],
    ]);
    cholesky_reconstructs(a, 1e-12);
    assert_close(
        a.cholesky().unwrap().l(),
        Matrix::new([[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]]),
        1e-12,
    );

    // An M·Mᵀ product is symmetric positive-definite for full-rank M.
    let m = Matrix::<4, 4>::new([
        [2.0, 0.0, 0.0, 0.0],
        [1.0, 3.0, 0.0, 0.0],
        [-1.0, 2.0, 4.0, 0.0],
        [0.0, 1.0, -2.0, 5.0],
    ]);
    cholesky_reconstructs(m * m.transpose(), 1e-12);

    // The same code at f32.
    cholesky_reconstructs(
        Matrix::<3, 3, f32>::new([
            [4.0, 12.0, -16.0],
            [12.0, 37.0, -43.0],
            [-16.0, -43.0, 98.0],
        ]),
        1e-3,
    );
}

#[test]
fn cholesky_rejects_non_pd() {
    // Symmetric but indefinite (eigenvalues 3 and -1).
    let indefinite = Matrix::<2, 2>::new([[1.0, 2.0], [2.0, 1.0]]);
    assert_eq!(
        indefinite.cholesky().err(),
        Some(LinalgError::NotPositiveDefinite)
    );

    // Negative leading diagonal entry.
    let negative = Matrix::<2, 2>::new([[-4.0, 0.0], [0.0, 1.0]]);
    assert_eq!(
        negative.cholesky().err(),
        Some(LinalgError::NotPositiveDefinite)
    );

    // Singular: the second radicand collapses to zero.
    let singular = Matrix::<2, 2>::new([[1.0, 1.0], [1.0, 1.0]]);
    assert_eq!(
        singular.cholesky().err(),
        Some(LinalgError::NotPositiveDefinite)
    );
}

#[test]
fn cholesky_solves() {
    // Single RHS on a 3x3 SPD system: exact solution, matches LU, tiny residual.
    let a = Matrix::<3, 3>::new([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]]);
    let x_exact = Vector::new([1.0, -2.0, 3.0]);
    let b = a * x_exact;
    let x = a.cholesky().unwrap().solve(b);
    for i in 0..3 {
        assert!((x[i] - x_exact[i]).abs() < 1e-12);
    }
    assert!((a * x - b).norm() < 1e-12);
    let lu_x = a.lu().unwrap().solve(b);
    for i in 0..3 {
        assert!((x[i] - lu_x[i]).abs() < 1e-12);
    }

    // Multiple RHS: A·X == B, and each column agrees with a single-RHS solve.
    let s = Matrix::<2, 2>::new([[4.0, 2.0], [2.0, 3.0]]);
    let f = s.cholesky().unwrap();
    let rhs = Matrix::<2, 3>::new([[8.0, 6.0, 4.0], [8.0, 5.0, 3.0]]);
    let xm = f.solve_matrix(rhs);
    assert_close(s * xm, rhs, 1e-12);
    for c in 0..3 {
        let single = f.solve(rhs.column(c));
        for r in 0..2 {
            assert!((xm[(r, c)] - single[r]).abs() < 1e-12);
        }
    }
}

#[test]
fn cholesky_determinant_matches() {
    let a = Matrix::<3, 3>::new([
        [4.0, 12.0, -16.0],
        [12.0, 37.0, -43.0],
        [-16.0, -43.0, 98.0],
    ]);
    let det = a.cholesky().unwrap().determinant();
    // (2·1·3)² == 36.
    assert!((det - 36.0).abs() < 1e-9);
    assert!((det - a.determinant()).abs() < 1e-9);
    assert!((det - a.lu().unwrap().determinant()).abs() < 1e-9);
}

#[test]
fn cholesky_inverse_matches_lu() {
    let a = Matrix::<3, 3>::new([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]]);
    let inv = a.cholesky().unwrap().inverse();
    assert_identity(inv * a, 1e-12);
    assert_identity(a * inv, 1e-12);
    assert_close(inv, a.lu().unwrap().inverse(), 1e-12);
}
