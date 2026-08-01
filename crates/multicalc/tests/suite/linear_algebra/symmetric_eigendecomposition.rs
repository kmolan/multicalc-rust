use multicalc::error::LinalgError;
use multicalc::linear_algebra::{Matrix, Matrix2D, Matrix3D, Matrix6D};
use multicalc::scalar::Dual;
use multicalc_testkit::tol::{assert_identity, assert_matrix_close};

/// Rebuilds `V·diag(λ)·Vᵀ` from a decomposition's own factors.
fn rebuilt<const N: usize>(matrix: Matrix<N, N>) -> Matrix<N, N> {
    let decomposition = matrix.symmetric_eigendecomposition().unwrap();
    let values = decomposition.eigenvalues();
    let vectors = decomposition.eigenvectors();
    Matrix::from_fn(|row, column| {
        (0..N)
            .map(|k| vectors[(row, k)] * values[k] * vectors[(column, k)])
            .sum()
    })
}

/// The proper rotation used to build matrices with a chosen spectrum.
fn rotation() -> Matrix3D<f64> {
    Matrix3D::new([
        [2.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0],
        [1.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0],
        [2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0],
    ])
}

/// A symmetric matrix with exactly the given eigenvalues.
fn with_spectrum(spectrum: [f64; 3]) -> Matrix3D<f64> {
    let r = rotation();
    r * Matrix3D::from_fn(|row, column| if row == column { spectrum[row] } else { 0.0 })
        * r.transpose()
}

#[test]
fn symmetric_eigendecomposition_known_spectrum() {
    let values = with_spectrum([6.0, 3.0, 1.0])
        .symmetric_eigendecomposition()
        .unwrap()
        .eigenvalues();
    for (index, expected) in [6.0, 3.0, 1.0].iter().enumerate() {
        assert!((values[index] - expected).abs() < 1e-12);
    }

    // Descending by value, not by magnitude, so -5 comes last rather than first.
    let mixed = with_spectrum([2.0, -5.0, 0.5])
        .symmetric_eigendecomposition()
        .unwrap()
        .eigenvalues();
    for (index, expected) in [2.0, 0.5, -5.0].iter().enumerate() {
        assert!((mixed[index] - expected).abs() < 1e-12);
    }
}

#[test]
fn symmetric_eigendecomposition_reconstructs() {
    let three = with_spectrum([6.0, 3.0, 1.0]);
    assert_matrix_close(rebuilt(three), three, 1e-12);
    assert_identity(
        three
            .symmetric_eigendecomposition()
            .unwrap()
            .eigenvectors()
            .transpose()
            * three.symmetric_eigendecomposition().unwrap().eigenvectors(),
        1e-12,
    );

    let six = Matrix6D::from_fn(|row, column| {
        if row == column {
            8.0
        } else {
            1.0 / (1.0 + (row + column) as f64)
        }
    });
    assert_matrix_close(rebuilt(six), six, 1e-12);
    let six_vectors = six.symmetric_eigendecomposition().unwrap().eigenvectors();
    assert_identity(six_vectors.transpose() * six_vectors, 1e-12);

    // Mixed signs and no particular structure, made symmetric by averaging with its transpose.
    let lopsided =
        Matrix::<4, 4>::from_fn(|row, column| ((row + 1) * (column + 1)) as f64 % 7.0 - 3.0);
    let four = (lopsided + lopsided.transpose()).scale(0.5);
    assert_matrix_close(rebuilt(four), four, 1e-12);
    let four_vectors = four.symmetric_eigendecomposition().unwrap().eigenvectors();
    assert_identity(four_vectors.transpose() * four_vectors, 1e-12);
}

#[test]
fn symmetric_eigendecomposition_matches_svd_on_positive_definite() {
    // For a positive-definite matrix the eigenvalues and the singular values are the same
    // numbers, and both come back largest first.
    let a = with_spectrum([6.0, 3.0, 1.0]);
    let eigenvalues = a.symmetric_eigendecomposition().unwrap().eigenvalues();
    let singular_values = a.svd().unwrap().singular_values();
    for index in 0..3 {
        assert!((eigenvalues[index] - singular_values[index]).abs() < 1e-10);
    }
}

#[test]
fn symmetric_eigendecomposition_degenerate_spectrum() {
    // A repeated eigenvalue leaves its two directions undetermined, so nothing is asserted about
    // the individual entries of the eigenvectors.
    let a = with_spectrum([5.0, 2.0, 2.0]);
    let decomposition = a.symmetric_eigendecomposition().unwrap();
    let values = decomposition.eigenvalues();
    for (index, expected) in [5.0, 2.0, 2.0].iter().enumerate() {
        assert!((values[index] - expected).abs() < 1e-12);
    }
    assert_matrix_close(rebuilt(a), a, 1e-12);
    let vectors = decomposition.eigenvectors();
    assert_identity(vectors.transpose() * vectors, 1e-12);
}

#[test]
fn symmetric_eigendecomposition_diagonal_and_identity() {
    let diagonal = Matrix::<4, 4>::from_fn(|row, column| {
        if row == column {
            [3.0, -5.0, 2.0, -1.0][row]
        } else {
            0.0
        }
    });
    let values = diagonal
        .symmetric_eigendecomposition()
        .unwrap()
        .eigenvalues();
    for (index, expected) in [3.0, 2.0, -1.0, -5.0].iter().enumerate() {
        assert!((values[index] - expected).abs() < 1e-12);
    }

    let identity = Matrix3D::<f64>::identity()
        .symmetric_eigendecomposition()
        .unwrap();
    for index in 0..3 {
        assert!((identity.eigenvalues()[index] - 1.0).abs() < 1e-12);
    }
    assert_identity(
        identity.eigenvectors().transpose() * identity.eigenvectors(),
        1e-12,
    );

    let zeros = Matrix3D::<f64>::zeros()
        .symmetric_eigendecomposition()
        .unwrap();
    for index in 0..3 {
        assert!(zeros.eigenvalues()[index].abs() < 1e-12);
    }
    assert!(zeros.condition_number().is_infinite());
}

#[test]
fn symmetric_eigendecomposition_determinant_and_condition_number() {
    let positive = with_spectrum([6.0, 3.0, 1.0])
        .symmetric_eigendecomposition()
        .unwrap();
    assert!((positive.determinant() - 18.0).abs() < 1e-9);
    assert!((positive.condition_number() - 6.0).abs() < 1e-9);
    assert!(positive.is_positive_definite());

    let mixed = with_spectrum([2.0, -5.0, 0.5])
        .symmetric_eigendecomposition()
        .unwrap();
    assert!((mixed.determinant() + 5.0).abs() < 1e-9);
    assert!((mixed.condition_number() - 10.0).abs() < 1e-9);
    assert!(!mixed.is_positive_definite());
}

#[test]
fn symmetric_eigendecomposition_clamped_lifts_the_spectrum() {
    let repaired = with_spectrum([2.0, -5.0, 0.5])
        .symmetric_eigendecomposition()
        .unwrap()
        .clamped(0.25);
    let decomposition = repaired.symmetric_eigendecomposition().unwrap();
    let values = decomposition.eigenvalues();
    for (index, expected) in [2.0, 0.5, 0.25].iter().enumerate() {
        assert!((values[index] - expected).abs() < 1e-9);
    }
    assert!(decomposition.is_positive_definite());

    // Exactly the same read across the diagonal, not merely close.
    for row in 0..3 {
        for column in 0..3 {
            assert_eq!(repaired[(row, column)], repaired[(column, row)]);
        }
    }

    // Nothing was below the floor, so the matrix comes back as it went in.
    let already_positive = with_spectrum([6.0, 3.0, 1.0]);
    let unchanged = already_positive
        .symmetric_eigendecomposition()
        .unwrap()
        .clamped(0.25);
    assert_matrix_close(unchanged, already_positive, 1e-12);
}

#[test]
fn symmetric_eigendecomposition_error_paths() {
    let with_nan = Matrix2D::new([[f64::NAN, 0.0], [0.0, 1.0]]);
    assert_eq!(
        with_nan.symmetric_eigendecomposition().err(),
        Some(LinalgError::NonFinite)
    );

    let with_infinity = Matrix2D::new([[f64::INFINITY, 0.0], [0.0, 1.0]]);
    assert_eq!(
        with_infinity.symmetric_eigendecomposition().err(),
        Some(LinalgError::NonFinite)
    );

    let lopsided = Matrix2D::new([[1.0, 2.0], [-2.0, 1.0]]);
    assert_eq!(
        lopsided.symmetric_eigendecomposition().err(),
        Some(LinalgError::NotSymmetric)
    );

    // The accepting side of the same check: a difference at rounding level still goes through.
    let mut drifted = with_spectrum([6.0, 3.0, 1.0]);
    drifted[(0, 1)] += 1e-15;
    assert!(drifted.symmetric_eigendecomposition().is_ok());
}

#[test]
fn symmetric_eigendecomposition_f32_reconstructs() {
    let a = Matrix3D::<f32>::new([[4.0, 1.0, 2.0], [1.0, 5.0, 3.0], [2.0, 3.0, 6.0]]);
    let decomposition = a.symmetric_eigendecomposition().unwrap();
    let values = decomposition.eigenvalues();
    let vectors = decomposition.eigenvectors();

    let reconstruction = Matrix3D::<f32>::from_fn(|row, column| {
        (0..3)
            .map(|k| vectors[(row, k)] * values[k] * vectors[(column, k)])
            .sum()
    });
    assert_matrix_close(reconstruction, a, 1e-4);
    assert_identity(vectors.transpose() * vectors, 1e-5);
}

#[test]
fn symmetric_eigendecomposition_differentiates() {
    // The eigenvalues of [[2, x], [x, 2]] are 2 + x and 2 - x, so at x = 0.5 they are 2.5 and 1.5
    // with derivatives 1 and -1.
    let a = Matrix::<2, 2, Dual<f64>>::new([
        [Dual::constant(2.0), Dual::variable(0.5)],
        [Dual::variable(0.5), Dual::constant(2.0)],
    ]);
    let values = a.symmetric_eigendecomposition().unwrap().eigenvalues();

    assert!((values[0].value - 2.5).abs() < 1e-12);
    assert!((values[0].deriv - 1.0).abs() < 1e-9);
    assert!((values[1].value - 1.5).abs() < 1e-12);
    assert!((values[1].deriv + 1.0).abs() < 1e-9);
}
