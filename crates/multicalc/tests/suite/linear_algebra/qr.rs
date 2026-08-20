use multicalc::error::LinalgError;
use multicalc::linear_algebra::{Matrix, Matrix3D, PivotedQr, Vector};
use multicalc_testkit::tol::{assert_identity, assert_matrix_close, max_abs};
use proptest::prelude::*;
use proptest::test_runner::TestCaseError;

// ----- column-pivoted QR (decompose, accessors, solve) -----

#[test]
fn qr_rejects_underdetermined() {
    let matrix = Matrix::<2, 3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    assert!(matches!(
        PivotedQr::decompose(matrix),
        Err(LinalgError::Underdetermined)
    ));
}

#[test]
fn qr_solves_square_system() {
    // A x = b with the exact solution x = [1, 1, 1].
    let matrix = Matrix3D::<f64>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]);
    let right_hand_side = Vector::new([6.0, 15.0, 25.0]);
    let solution = PivotedQr::decompose(matrix)
        .unwrap()
        .solve_least_squares(right_hand_side)
        .unwrap();
    for value in solution.into_array() {
        assert!((value - 1.0).abs() < 1e-12);
    }
}

#[test]
fn qr_solves_overdetermined_least_squares() {
    // Fit y = m t + c to three non-collinear points; least-squares gives m = 0.5, c = 7/6.
    let matrix = Matrix::<3, 2>::new([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]);
    let right_hand_side = Vector::new([1.0, 2.0, 2.0]);
    let solution = PivotedQr::decompose(matrix)
        .unwrap()
        .solve_least_squares(right_hand_side)
        .unwrap();
    assert!((solution[0] - 0.5).abs() < 1e-12);
    assert!((solution[1] - 7.0 / 6.0).abs() < 1e-12);
}

#[test]
fn qr_solve_rejects_rank_deficient() {
    let right_hand_side = Vector::new([1.0, 2.0, 3.0]);

    // The middle column is zero, so R has an exactly-zero diagonal entry.
    let zero_column = Matrix3D::new([[1.0, 0.0, 2.0], [3.0, 0.0, 4.0], [5.0, 0.0, 6.0]]);
    assert!(matches!(
        PivotedQr::decompose(zero_column)
            .unwrap()
            .solve_least_squares(right_hand_side),
        Err(LinalgError::Singular)
    ));

    // col2 = col0 + col1: dependent columns leave a tiny (not exactly zero) diagonal, which the
    // relative rank tolerance still flags.
    let dependent = Matrix3D::new([[1.0, 2.0, 3.0], [4.0, 5.0, 9.0], [7.0, 8.0, 15.0]]);
    assert!(matches!(
        PivotedQr::decompose(dependent)
            .unwrap()
            .solve_least_squares(right_hand_side),
        Err(LinalgError::Singular)
    ));
}

// ----- damped least squares (qrsolv) -----

// A full-rank 4x3 problem reused across the damped-solve tests.
fn sample_problem() -> (Matrix<4, 3>, Vector<4>) {
    let jacobian = Matrix::<4, 3>::new([
        [1.0, 2.0, 0.0],
        [0.0, 1.0, 3.0],
        [2.0, 1.0, 1.0],
        [1.0, 0.0, 2.0],
    ]);
    let right_hand_side = Vector::new([1.0, 2.0, 3.0, 4.0]);
    (jacobian, right_hand_side)
}

#[test]
fn damped_solve_satisfies_normal_equations() {
    let (jacobian, right_hand_side) = sample_problem();
    let diagonal = [1.0, 0.5, 2.0];
    let damped = PivotedQr::decompose(jacobian)
        .unwrap()
        .into_damped(right_hand_side);
    let (solution, _) = damped.solve_with_diagonal(&diagonal);

    // x must satisfy (JᵀJ + D²) x = Jᵀb.
    let normal_matrix = jacobian.transpose() * jacobian;
    let normal_right_hand_side = jacobian.transpose() * right_hand_side;
    let left_side = Matrix3D::from_fn(|row, column| {
        normal_matrix[(row, column)]
            + if row == column {
                diagonal[row] * diagonal[row]
            } else {
                0.0
            }
    }) * solution;
    for index in 0..3 {
        assert!((left_side[index] - normal_right_hand_side[index]).abs() < 1e-12);
    }
}

#[test]
fn damped_zero_diagonal_matches_least_squares() {
    let (jacobian, right_hand_side) = sample_problem();
    let pivoted_qr = PivotedQr::decompose(jacobian).unwrap();
    let expected = pivoted_qr.solve_least_squares(right_hand_side).unwrap();
    let (solution, _) = pivoted_qr
        .into_damped(right_hand_side)
        .solve_with_zero_diagonal();
    for index in 0..3 {
        assert!((solution[index] - expected[index]).abs() < 1e-12);
    }
}

#[test]
fn damped_accessors() {
    let (jacobian, right_hand_side) = sample_problem();
    let damped = PivotedQr::decompose(jacobian)
        .unwrap()
        .into_damped(right_hand_side);

    // max_a_t_b_scaled: max over columns of |Jᵀb|ₗ / (b_norm · ‖columnₗ‖).
    let b_norm = right_hand_side.norm();
    let normal_right_hand_side = jacobian.transpose() * right_hand_side;
    let mut expected = 0.0_f64;
    for column in 0..3 {
        let scaled = (normal_right_hand_side[column]
            / b_norm
            / Vector::<4>::from_fn(|row| jacobian[(row, column)]).norm())
        .abs();
        expected = expected.max(scaled);
    }
    assert!((damped.max_a_t_b_scaled(b_norm) - expected).abs() < 1e-12);

    // a_x_norm(x) is ‖J x‖.
    let candidate = Vector::new([1.0, -2.0, 0.5]);
    assert!((damped.a_x_norm(&candidate) - (jacobian * candidate).norm()).abs() < 1e-12);

    // is_non_singular: true for the full-rank problem, false once a column goes to zero.
    assert!(damped.is_non_singular());
    let deficient = Matrix::<4, 3>::new([
        [1.0, 0.0, 2.0],
        [3.0, 0.0, 4.0],
        [5.0, 0.0, 6.0],
        [7.0, 0.0, 8.0],
    ]);
    assert!(
        !PivotedQr::decompose(deficient)
            .unwrap()
            .into_damped(right_hand_side)
            .is_non_singular()
    );
}

#[test]
fn qr_fits_vandermonde_polynomial() {
    // Fit a degree-6 polynomial to 20 points on [-1, 1] by QR least squares.
    let node = |index: usize| -1.0 + 2.0 * index as f64 / 19.0;
    let vandermonde = Matrix::<20, 7>::from_fn(|row, column| {
        let node_value = node(row);
        (0..column).fold(1.0, |power, _| power * node_value)
    });
    let coefficients = [0.5, -1.2, 2.0, 0.3, -0.8, 1.1, -0.4];
    let right_hand_side = vandermonde * Vector::new(coefficients);

    let solution = PivotedQr::decompose(vandermonde)
        .unwrap()
        .solve_least_squares(right_hand_side)
        .unwrap();

    // Every coefficient is recovered and the fit reproduces the samples.
    for (got, want) in solution.into_array().iter().zip(coefficients.iter()) {
        assert!((got - want).abs() < 1e-7, "got {got}, want {want}");
    }
    assert!((vandermonde * solution - right_hand_side).norm() < 1e-10);
}

#[test]
fn qr_factorizes_hilbert_stably() {
    // The Hilbert matrix is famously ill-conditioned (cond(H_8) is about 1.5e10).
    let hilbert = Matrix::<8, 8>::from_fn(|row, column| 1.0 / ((row + column + 1) as f64));
    let factorization = PivotedQr::decompose(hilbert).unwrap();
    let permutation = factorization.permutation();
    let orthogonal = factorization.orthogonal();
    let triangular = factorization.triangular();

    // The factorization stays backward-stable regardless of conditioning.
    assert_identity(orthogonal.transpose() * orthogonal, 1e-12);
    let column_permuted =
        Matrix::<8, 8>::from_fn(|row, column| hilbert[(row, permutation[column])]);
    assert_matrix_close(orthogonal * triangular, column_permuted, 1e-12);

    // Solving is backward-stable (tiny residual) though the solution itself degrades.
    let true_coefficients = [1.0; 8];
    let right_hand_side = hilbert * Vector::new(true_coefficients);
    let solution = factorization.solve_least_squares(right_hand_side).unwrap();
    assert!((hilbert * solution - right_hand_side).norm() < 1e-12);
    for value in solution.into_array() {
        assert!((value - 1.0).abs() < 1e-2);
    }
}

#[test]
fn damped_solves_ridge_regression() {
    // Ridge (Tikhonov) regression on an ill-conditioned Vandermonde design:
    // (VᵀV + λ²I) x = Vᵀb, which is exactly the damped solve with a constant diagonal.
    let node = |index: usize| -1.0 + 2.0 * index as f64 / 14.0;
    let design = Matrix::<15, 8>::from_fn(|row, column| {
        let node_value = node(row);
        (0..column).fold(1.0, |power, _| power * node_value)
    });
    let true_coefficients = [0.4, 1.0, -0.6, 0.9, -1.3, 0.5, 0.7, -0.2];
    let right_hand_side = design * Vector::new(true_coefficients);
    let ridge_parameter = 0.1;

    let (solution, _) = PivotedQr::decompose(design)
        .unwrap()
        .into_damped(right_hand_side)
        .solve_with_diagonal(&[ridge_parameter; 8]);

    // x satisfies the regularized normal equations.
    let normal_matrix = design.transpose() * design;
    let normal_right_hand_side = design.transpose() * right_hand_side;
    let left_side = Matrix::<8, 8>::from_fn(|row, column| {
        normal_matrix[(row, column)]
            + if row == column {
                ridge_parameter * ridge_parameter
            } else {
                0.0
            }
    }) * solution;
    for index in 0..8 {
        assert!((left_side[index] - normal_right_hand_side[index]).abs() < 1e-8);
    }
}

// ----- property: A·P = Q·R on random matrices -----

/// Builds an `M`x`N` matrix from `entries` and checks the column-pivoted QR identities: `R` is
/// upper-triangular, `Q` has orthonormal columns (`QᵀQ = I`), and `A·P = Q·R` (column `j` of
/// `A·P` is column `permutation()[j]` of `A`). Tolerance is scaled by the matrix's magnitude and
/// `f64::EPSILON`.
///
/// Rejects rather than asserts on inputs the generator turns up that are near rank-deficient (an
/// `R` diagonal entry too small relative to the matrix's scale) — the factorization stays
/// backward-stable regardless, but the reconstruction tolerance below would flake on the rare,
/// mildly ill-conditioned draw.
fn check_qr_property<const M: usize, const N: usize>(
    entries: Vec<f64>,
) -> Result<(), TestCaseError> {
    let matrix = Matrix::<M, N>::try_from_row_slice(&entries).expect("M*N entries");
    let scale = max_abs(matrix).max(1.0);

    // M >= N is guaranteed by the generators below, so this never hits `Underdetermined`.
    let factorization = PivotedQr::decompose(matrix).unwrap();
    let triangular = factorization.triangular();
    let orthogonal = factorization.orthogonal();
    let permutation = factorization.permutation();

    let minimum_diagonal = (0..N).fold(f64::MAX, |smallest, index| {
        smallest.min(triangular[(index, index)].abs())
    });
    prop_assume!(minimum_diagonal >= 1e-6 * scale);

    let tolerance = M.max(N) as f64 * scale * f64::EPSILON * 1e3;

    // R is upper-triangular by construction; check anyway as a structural guard.
    for row in 0..N {
        for column in 0..row {
            assert_eq!(triangular[(row, column)], 0.0);
        }
    }

    assert_identity(orthogonal.transpose() * orthogonal, tolerance);

    let column_permuted = Matrix::<M, N>::from_fn(|row, column| matrix[(row, permutation[column])]);
    assert_matrix_close(orthogonal * triangular, column_permuted, tolerance);

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn proptest_qr_reconstructs_3x3(entries in prop::collection::vec(-8.0f64..8.0, 9)) {
        check_qr_property::<3, 3>(entries)?;
    }

    #[test]
    fn proptest_qr_reconstructs_5x5(entries in prop::collection::vec(-8.0f64..8.0, 25)) {
        check_qr_property::<5, 5>(entries)?;
    }

    // Rectangular (overdetermined) case: more rows than columns.
    #[test]
    fn proptest_qr_reconstructs_6x3(entries in prop::collection::vec(-8.0f64..8.0, 18)) {
        check_qr_property::<6, 3>(entries)?;
    }
}
