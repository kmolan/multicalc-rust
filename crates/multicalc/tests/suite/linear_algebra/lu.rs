use multicalc::error::LinalgError;
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc_testkit::tol::{assert_identity, assert_matrix_close, lu_reconstructs, max_abs};
use proptest::prelude::*;
use proptest::test_runner::TestCaseError;

// ----- LU decomposition (Doolittle, partial pivoting) -----

#[test]
fn lu_reconstructs_pivoted_matrix() {
    // The largest first-column entry is in the last row, forcing a swap.
    lu_reconstructs(
        Matrix::<3, 3>::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]),
        1e-12,
    );
    lu_reconstructs(
        Matrix::<4, 4>::new([
            [4.0, 3.0, 2.0, 1.0],
            [3.0, 4.0, 3.0, 2.0],
            [2.0, 3.0, 4.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
        ]),
        1e-12,
    );
    // The same code at f32.
    lu_reconstructs(
        Matrix::<3, 3, f32>::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]),
        1e-5,
    );
}

#[test]
fn lu_determinant_matches_direct() {
    // Cross-check against the direct determinant, including the pivot-sign handling.
    let a = Matrix::<3, 3>::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]);
    assert!((a.lu().unwrap().determinant() - a.determinant()).abs() < 1e-12);

    let b = Matrix::<4, 4>::new([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 1.0, 0.0, 1.0],
        [0.0, 3.0, 1.0, 2.0],
        [1.0, 0.0, 2.0, 1.0],
    ]);
    assert!((b.lu().unwrap().determinant() - b.determinant()).abs() < 1e-12);
    assert!((b.lu().unwrap().determinant() + 20.0).abs() < 1e-12);
}

#[test]
fn lu_rejects_singular() {
    // A zero column: the pivot search turns up only zeros.
    let zero_col = Matrix::<3, 3>::new([[1.0, 0.0, 2.0], [3.0, 0.0, 4.0], [5.0, 0.0, 6.0]]);
    assert_eq!(zero_col.lu().err(), Some(LinalgError::Singular));

    // Dependent rows drive a pivot to zero during elimination.
    let dependent = Matrix::<2, 2>::new([[1.0, 2.0], [2.0, 4.0]]);
    assert_eq!(dependent.lu().err(), Some(LinalgError::Singular));
}

#[test]
fn lu_solves() {
    let a = Matrix::<3, 3>::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]);
    let f = a.lu().unwrap();

    // Single RHS: A·x = b has the exact solution x = [1, 2, 3], with a tiny residual.
    let b = Vector::new([7.0, 19.0, 49.0]);
    let x = f.solve(b);
    assert!((x[0] - 1.0).abs() < 1e-12);
    assert!((x[1] - 2.0).abs() < 1e-12);
    assert!((x[2] - 3.0).abs() < 1e-12);
    assert!((a * x - b).norm() < 1e-12);

    // Multiple RHS: A·X == B, and each column agrees with a single-RHS solve.
    let rhs = Matrix::<3, 2>::new([[7.0, 4.0], [19.0, 10.0], [49.0, 26.0]]);
    let xm = f.solve_matrix(rhs);
    assert_matrix_close(a * xm, rhs, 1e-12);
    for c in 0..2 {
        let single = f.solve(rhs.column(c));
        for r in 0..3 {
            assert!((xm[(r, c)] - single[r]).abs() < 1e-12);
        }
    }
}

#[test]
fn lu_inverse_matches_reference_5x5() {
    // A non-symmetric 5×5; reference inverse from an exact rational solve. The direct inverse is
    // covered in matrix.rs; this guards the LU inverse on a larger, non-symmetric system.
    let a = Matrix::<5, 5>::new([
        [5.0, 1.0, 0.0, 2.0, 1.0],
        [1.0, 6.0, 2.0, 0.0, 1.0],
        [3.0, 2.0, 7.0, 1.0, 0.0],
        [2.0, 0.0, 1.0, 8.0, 2.0],
        [1.0, 4.0, 0.0, 2.0, 9.0],
    ]);
    assert!((a.lu().unwrap().determinant() - 10406.0).abs() < 1e-9);

    let inv = a.lu().unwrap().inverse();
    assert_matrix_close(
        inv,
        Matrix::new([
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
        ]),
        1e-12,
    );
    assert_identity(a * inv, 1e-12);
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
    let a = Matrix::<N, N>::try_from_row_slice(&entries).expect("N*N entries");
    let scale = max_abs(a).max(1.0);

    let lu = a.lu();
    prop_assume!(lu.is_ok());
    let f = lu.unwrap();

    let min_pivot = (0..N).fold(f64::MAX, |acc, i| acc.min(f.u()[(i, i)].abs()));
    prop_assume!(min_pivot >= 1e-6 * scale);

    let tol = N as f64 * scale * f64::EPSILON * 1e3;
    lu_reconstructs(a, tol);
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
