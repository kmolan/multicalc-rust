use multicalc::error::LinalgError;
use multicalc::linear_algebra::{Matrix, Matrix2D, Matrix3D, Matrix4D, Vector};
use multicalc_testkit::tol::{assert_identity, assert_matrix_close, cholesky_reconstructs};

#[test]
fn cholesky_reconstructs_spd() {
    cholesky_reconstructs(Matrix2D::new([[4.0, 2.0], [2.0, 3.0]]), 1e-12);

    // A matrix with a known exact factor: L = [[2,0,0],[6,1,0],[-8,5,3]].
    let known_factor = Matrix3D::new([
        [4.0, 12.0, -16.0],
        [12.0, 37.0, -43.0],
        [-16.0, -43.0, 98.0],
    ]);
    cholesky_reconstructs(known_factor, 1e-12);
    let expected = Matrix::new([[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]]);
    assert_matrix_close(known_factor.cholesky().unwrap().l(), expected, 1e-12);

    // An M·Mᵀ product is symmetric positive-definite for full-rank M.
    let lower_factor_source = Matrix4D::new([
        [2.0, 0.0, 0.0, 0.0],
        [1.0, 3.0, 0.0, 0.0],
        [-1.0, 2.0, 4.0, 0.0],
        [0.0, 1.0, -2.0, 5.0],
    ]);
    cholesky_reconstructs(lower_factor_source * lower_factor_source.transpose(), 1e-12);

    // The same code at f32.
    cholesky_reconstructs(
        Matrix3D::<f32>::new([
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
    let indefinite = Matrix2D::new([[1.0, 2.0], [2.0, 1.0]]);
    assert_eq!(
        indefinite.cholesky().err(),
        Some(LinalgError::NotPositiveDefinite)
    );

    // Negative leading diagonal entry.
    let negative = Matrix2D::new([[-4.0, 0.0], [0.0, 1.0]]);
    assert_eq!(
        negative.cholesky().err(),
        Some(LinalgError::NotPositiveDefinite)
    );

    // Singular: the second radicand collapses to zero.
    let singular = Matrix2D::new([[1.0, 1.0], [1.0, 1.0]]);
    assert_eq!(
        singular.cholesky().err(),
        Some(LinalgError::NotPositiveDefinite)
    );
}

#[test]
fn cholesky_solves() {
    // Single RHS on a 3x3 SPD system: exact solution, matches LU, tiny residual.
    let tridiagonal = Matrix3D::<f64>::new([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]]);
    let exact_solution = Vector::new([1.0, -2.0, 3.0]);
    let right_hand_side = tridiagonal * exact_solution;
    let solution = tridiagonal.cholesky().unwrap().solve(right_hand_side);
    for index in 0..3 {
        assert!((solution[index] - exact_solution[index]).abs() < 1e-12);
    }
    assert!((tridiagonal * solution - right_hand_side).norm() < 1e-12);
    let lu_solution = tridiagonal.lu().unwrap().solve(right_hand_side);
    for index in 0..3 {
        assert!((solution[index] - lu_solution[index]).abs() < 1e-12);
    }

    // Multiple RHS: A·X == B, and each column agrees with a single-RHS solve.
    let small_spd = Matrix2D::new([[4.0, 2.0], [2.0, 3.0]]);
    let factorization = small_spd.cholesky().unwrap();
    let right_hand_sides = Matrix::<2, 3>::new([[8.0, 6.0, 4.0], [8.0, 5.0, 3.0]]);
    let matrix_solution = factorization.solve_matrix(right_hand_sides);
    assert_matrix_close(small_spd * matrix_solution, right_hand_sides, 1e-12);
    for column in 0..3 {
        let single_column_solution =
            factorization.solve(Vector::from_fn(|row| right_hand_sides[(row, column)]));
        for row in 0..2 {
            assert!((matrix_solution[(row, column)] - single_column_solution[row]).abs() < 1e-12);
        }
    }
}

#[test]
fn cholesky_determinant_matches() {
    let known_factor = Matrix3D::<f64>::new([
        [4.0, 12.0, -16.0],
        [12.0, 37.0, -43.0],
        [-16.0, -43.0, 98.0],
    ]);
    let determinant = known_factor.cholesky().unwrap().determinant();
    // (2·1·3)² == 36.
    assert!((determinant - 36.0).abs() < 1e-9);
    assert!((determinant - known_factor.determinant()).abs() < 1e-9);
    assert!((determinant - known_factor.lu().unwrap().determinant()).abs() < 1e-9);
}

#[test]
fn cholesky_inverse_matches_lu() {
    let tridiagonal = Matrix3D::new([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]]);
    let inverse = tridiagonal.cholesky().unwrap().inverse();
    assert_identity(inverse * tridiagonal, 1e-12);
    assert_identity(tridiagonal * inverse, 1e-12);
    assert_matrix_close(inverse, tridiagonal.lu().unwrap().inverse(), 1e-12);
}
