use multicalc::error::LinalgError;
use multicalc::linear_algebra::{Matrix, Matrix2D, Matrix3D, Matrix4D, Vector};
use multicalc_testkit::tol::{assert_identity, assert_matrix_close, lu_reconstructs, max_abs};
use proptest::prelude::*;
use proptest::test_runner::TestCaseError;

// ----- LU decomposition (Doolittle, partial pivoting) -----

#[test]
fn lu_reconstructs_pivoted_matrix() {
    // The largest first-column entry is in the last row, forcing a swap.
    lu_reconstructs(
        Matrix3D::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]),
        1e-12,
    );
    lu_reconstructs(
        Matrix4D::new([
            [4.0, 3.0, 2.0, 1.0],
            [3.0, 4.0, 3.0, 2.0],
            [2.0, 3.0, 4.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
        ]),
        1e-12,
    );
    // The same code at f32.
    lu_reconstructs(
        Matrix3D::<f32>::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]),
        1e-5,
    );
}

#[test]
fn lu_determinant_matches_direct() {
    // Cross-check against the direct determinant, including the pivot-sign handling.
    let pivoted = Matrix3D::<f64>::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]);
    assert!((pivoted.lu().unwrap().determinant() - pivoted.determinant()).abs() < 1e-12);

    let non_symmetric = Matrix4D::<f64>::new([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 1.0, 0.0, 1.0],
        [0.0, 3.0, 1.0, 2.0],
        [1.0, 0.0, 2.0, 1.0],
    ]);
    assert!(
        (non_symmetric.lu().unwrap().determinant() - non_symmetric.determinant()).abs() < 1e-12
    );
    assert!((non_symmetric.lu().unwrap().determinant() + 20.0).abs() < 1e-12);
}

#[test]
fn lu_rejects_singular() {
    // A zero column: the pivot search turns up only zeros.
    let zero_column = Matrix3D::new([[1.0, 0.0, 2.0], [3.0, 0.0, 4.0], [5.0, 0.0, 6.0]]);
    assert_eq!(zero_column.lu().err(), Some(LinalgError::Singular));

    // Dependent rows drive a pivot to zero during elimination.
    let dependent = Matrix2D::new([[1.0, 2.0], [2.0, 4.0]]);
    assert_eq!(dependent.lu().err(), Some(LinalgError::Singular));
}

#[test]
fn lu_rejects_a_pivot_negligible_against_the_largest() {
    // Put the negligible pivot first so the check must retain the smallest pivot seen when a
    // larger one appears later. The matrix is invertible in exact arithmetic but too
    // ill-conditioned for the documented LU/inverse contract.
    let near_singular = Matrix::<5, 5>::from_diagonal([1e-300, 1.0, 1.0, 1.0, 1.0]);
    assert_eq!(near_singular.lu().err(), Some(LinalgError::Singular));
    assert_eq!(near_singular.inverse().err(), Some(LinalgError::Singular));

    // The threshold is relative: uniformly tiny, well-conditioned matrices remain valid.
    let uniformly_scaled = Matrix::<5, 5>::from_diagonal([1e-300, 2e-300, 3e-300, 4e-300, 5e-300]);
    assert!(uniformly_scaled.lu().is_ok());
    assert!(uniformly_scaled.inverse().is_ok());
}

#[test]
fn lu_solves() {
    let matrix = Matrix3D::<f64>::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]);
    let factorization = matrix.lu().unwrap();

    // Single RHS: A·x = b has the exact solution x = [1, 2, 3], with a tiny residual.
    let right_hand_side = Vector::new([7.0, 19.0, 49.0]);
    let solution = factorization.solve(right_hand_side);
    assert!((solution[0] - 1.0).abs() < 1e-12);
    assert!((solution[1] - 2.0).abs() < 1e-12);
    assert!((solution[2] - 3.0).abs() < 1e-12);
    assert!((matrix * solution - right_hand_side).norm() < 1e-12);

    // Multiple RHS: A·X == B, and each column agrees with a single-RHS solve.
    let right_hand_sides = Matrix::<3, 2>::new([[7.0, 4.0], [19.0, 10.0], [49.0, 26.0]]);
    let matrix_solution = factorization.solve_matrix(right_hand_sides);
    assert_matrix_close(matrix * matrix_solution, right_hand_sides, 1e-12);
    for column in 0..2 {
        let single_column_solution =
            factorization.solve(Vector::from_fn(|row| right_hand_sides[(row, column)]));
        for row in 0..3 {
            assert!((matrix_solution[(row, column)] - single_column_solution[row]).abs() < 1e-12);
        }
    }
}

#[test]
fn lu_inverse_matches_reference_5x5() {
    // A non-symmetric 5×5; reference inverse from an exact rational solve. The direct inverse is
    // covered in matrix.rs; this guards the LU inverse on a larger, non-symmetric system.
    let matrix = Matrix::<5, 5>::new([
        [5.0, 1.0, 0.0, 2.0, 1.0],
        [1.0, 6.0, 2.0, 0.0, 1.0],
        [3.0, 2.0, 7.0, 1.0, 0.0],
        [2.0, 0.0, 1.0, 8.0, 2.0],
        [1.0, 4.0, 0.0, 2.0, 9.0],
    ]);
    assert!((matrix.lu().unwrap().determinant() - 10406.0).abs() < 1e-9);

    let inverse = matrix.lu().unwrap().inverse();
    let expected = Matrix::new([
        [
            0.2200653469152412,
            -0.03757447626369402,
            0.01864309052469729,
            -0.055352681145492987,
            -0.007976167595617912,
        ],
        [
            -0.005573707476455891,
            0.20488179896213723,
            -0.06073419181241591,
            0.01537574476263694,
            -0.025562175667883914,
        ],
        [
            -0.08687295790889871,
            -0.04804920238324044,
            0.15683259657889678,
            -0.0017297712857966558,
            0.01537574476263694,
        ],
        [
            -0.04093792043052085,
            0.039304247549490676,
            -0.032289064001537575,
            0.14741495291178167,
            -0.032577359215837015,
        ],
        [
            -0.012877186238708437,
            -0.09561791274264847,
            0.032096867192004615,
            -0.03344224485873534,
            0.13059773207764752,
        ],
    ]);
    assert_matrix_close(inverse, expected, 1e-12);
    assert_identity(matrix * inverse, 1e-12);
}

// ----- property: P·A = L·U on random matrices -----

/// Builds an `N`x`N` matrix from `entries` and checks `P·A = L·U` (`L` unit lower-triangular,
/// `U` upper-triangular) via [`lu_reconstructs`], at a tolerance scaled by the matrix's
/// magnitude and `f64::EPSILON`.
///
/// Rejects rather than asserts on inputs the generator turns up that are singular, or that carry
/// a pivot too small relative to the matrix's scale: partial pivoting keeps elimination growth
/// mild, but a near-zero pivot still leaves the reconstruction too ill-conditioned for a fixed
/// tolerance not to flake.
fn check_lu_property<const N: usize>(entries: Vec<f64>) -> Result<(), TestCaseError> {
    let matrix = Matrix::<N, N>::try_from_row_slice(&entries).expect("N*N entries");
    let scale = max_abs(matrix).max(1.0);

    let lu = matrix.lu();
    prop_assume!(lu.is_ok());
    let factorization = lu.unwrap();

    let min_pivot = (0..N).fold(f64::MAX, |smallest, index| {
        smallest.min(factorization.u()[(index, index)].abs())
    });
    prop_assume!(min_pivot >= 1e-6 * scale);

    let tolerance = N as f64 * scale * f64::EPSILON * 1e3;
    lu_reconstructs(matrix, tolerance);
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn proptest_lu_reconstructs_3x3(entries in prop::collection::vec(-8.0f64..8.0, 9)) {
        check_lu_property::<3>(entries)?;
    }

    #[test]
    fn proptest_lu_reconstructs_4x4(entries in prop::collection::vec(-8.0f64..8.0, 16)) {
        check_lu_property::<4>(entries)?;
    }

    #[test]
    fn proptest_lu_reconstructs_5x5(entries in prop::collection::vec(-8.0f64..8.0, 25)) {
        check_lu_property::<5>(entries)?;
    }
}
