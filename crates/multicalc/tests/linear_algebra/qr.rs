use crate::helpers::{assert_close, assert_identity};
use multicalc::linear_algebra::{Matrix, PivotedQr, Vector};
use multicalc::utils::error_codes::CalcError;

// ----- column-pivoted QR (decompose, accessors, solve) -----

#[test]
fn qr_rejects_underdetermined() {
    let a = Matrix::<2, 3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    assert!(matches!(
        PivotedQr::decompose(a),
        Err(CalcError::Underdetermined)
    ));
}

#[test]
fn qr_solves_square_system() {
    // A x = b with the exact solution x = [1, 1, 1].
    let a = Matrix::<3, 3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]);
    let b = Vector::new([6.0, 15.0, 25.0]);
    let x = PivotedQr::decompose(a)
        .unwrap()
        .solve_least_squares(b)
        .unwrap();
    for value in x.into_array() {
        assert!((value - 1.0).abs() < 1e-12);
    }
}

#[test]
fn qr_solves_overdetermined_least_squares() {
    // Fit y = m t + c to three non-collinear points; least-squares gives m = 0.5, c = 7/6.
    let a = Matrix::<3, 2>::new([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]);
    let b = Vector::new([1.0, 2.0, 2.0]);
    let x = PivotedQr::decompose(a)
        .unwrap()
        .solve_least_squares(b)
        .unwrap();
    assert!((x[0] - 0.5).abs() < 1e-12);
    assert!((x[1] - 7.0 / 6.0).abs() < 1e-12);
}

#[test]
fn qr_solve_rejects_rank_deficient() {
    let b = Vector::new([1.0, 2.0, 3.0]);

    // The middle column is zero, so R has an exactly-zero diagonal entry.
    let zero_col = Matrix::<3, 3>::new([[1.0, 0.0, 2.0], [3.0, 0.0, 4.0], [5.0, 0.0, 6.0]]);
    assert!(matches!(
        PivotedQr::decompose(zero_col)
            .unwrap()
            .solve_least_squares(b),
        Err(CalcError::SingularMatrix)
    ));

    // col2 = col0 + col1: dependent columns leave a tiny (not exactly zero) diagonal, which the
    // relative rank tolerance still flags.
    let dependent = Matrix::<3, 3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 9.0], [7.0, 8.0, 15.0]]);
    assert!(matches!(
        PivotedQr::decompose(dependent)
            .unwrap()
            .solve_least_squares(b),
        Err(CalcError::SingularMatrix)
    ));
}

// ----- damped least squares (qrsolv) -----

// A full-rank 4x3 problem reused across the damped-solve tests.
fn sample_problem() -> (Matrix<4, 3>, Vector<4>) {
    let j = Matrix::<4, 3>::new([
        [1.0, 2.0, 0.0],
        [0.0, 1.0, 3.0],
        [2.0, 1.0, 1.0],
        [1.0, 0.0, 2.0],
    ]);
    let b = Vector::new([1.0, 2.0, 3.0, 4.0]);
    (j, b)
}

#[test]
fn damped_solve_satisfies_normal_equations() {
    let (j, b) = sample_problem();
    let diag = [1.0, 0.5, 2.0];
    let dls = PivotedQr::decompose(j).unwrap().into_damped(b);
    let (x, _) = dls.solve_with_diagonal(&diag);

    // x must satisfy (JᵀJ + D²) x = Jᵀb.
    let jtj = j.transpose() * j;
    let jtb = j.transpose() * b;
    let lhs =
        Matrix::<3, 3>::from_fn(|r, c| jtj[(r, c)] + if r == c { diag[r] * diag[r] } else { 0.0 })
            * x;
    for i in 0..3 {
        assert!((lhs[i] - jtb[i]).abs() < 1e-12);
    }
}

#[test]
fn damped_zero_diagonal_matches_least_squares() {
    let (j, b) = sample_problem();
    let qr = PivotedQr::decompose(j).unwrap();
    let expected = qr.solve_least_squares(b).unwrap();
    let (x, _) = qr.into_damped(b).solve_with_zero_diagonal();
    for i in 0..3 {
        assert!((x[i] - expected[i]).abs() < 1e-12);
    }
}

#[test]
fn damped_accessors() {
    let (j, b) = sample_problem();
    let dls = PivotedQr::decompose(j).unwrap().into_damped(b);

    // max_a_t_b_scaled: max over columns of |Jᵀb|ₗ / (b_norm · ‖columnₗ‖).
    let b_norm = b.norm();
    let jtb = j.transpose() * b;
    let mut expected = 0.0_f64;
    for l in 0..3 {
        let scaled = (jtb[l] / b_norm / j.column(l).norm()).abs();
        expected = expected.max(scaled);
    }
    assert!((dls.max_a_t_b_scaled(b_norm) - expected).abs() < 1e-12);

    // a_x_norm(x) is ‖J x‖.
    let x = Vector::new([1.0, -2.0, 0.5]);
    assert!((dls.a_x_norm(&x) - (j * x).norm()).abs() < 1e-12);

    // is_non_singular: true for the full-rank problem, false once a column goes to zero.
    assert!(dls.is_non_singular());
    let deficient = Matrix::<4, 3>::new([
        [1.0, 0.0, 2.0],
        [3.0, 0.0, 4.0],
        [5.0, 0.0, 6.0],
        [7.0, 0.0, 8.0],
    ]);
    assert!(
        !PivotedQr::decompose(deficient)
            .unwrap()
            .into_damped(b)
            .is_non_singular()
    );
}

#[test]
fn qr_fits_vandermonde_polynomial() {
    // Fit a degree-6 polynomial to 20 points on [-1, 1] by QR least squares.
    let node = |i: usize| -1.0 + 2.0 * i as f64 / 19.0;
    let vandermonde = Matrix::<20, 7>::from_fn(|i, j| {
        let t = node(i);
        (0..j).fold(1.0, |acc, _| acc * t)
    });
    let coeffs = [0.5, -1.2, 2.0, 0.3, -0.8, 1.1, -0.4];
    let b = vandermonde * Vector::new(coeffs);

    let x = PivotedQr::decompose(vandermonde)
        .unwrap()
        .solve_least_squares(b)
        .unwrap();

    // Every coefficient is recovered and the fit reproduces the samples.
    for (got, want) in x.into_array().iter().zip(coeffs.iter()) {
        assert!((got - want).abs() < 1e-7, "got {got}, want {want}");
    }
    assert!((vandermonde * x - b).norm() < 1e-10);
}

#[test]
fn qr_factorizes_hilbert_stably() {
    // The Hilbert matrix is famously ill-conditioned (cond(H_8) is about 1.5e10).
    let hilbert = Matrix::<8, 8>::from_fn(|i, j| 1.0 / ((i + j + 1) as f64));
    let f = PivotedQr::decompose(hilbert).unwrap();
    let perm = f.permutation();
    let q = f.q();
    let r = f.r();

    // The factorization stays backward-stable regardless of conditioning.
    assert_identity(q.transpose() * q, 1e-12);
    let ap = Matrix::<8, 8>::from_fn(|i, c| hilbert[(i, perm[c])]);
    assert_close(q * r, ap, 1e-12);

    // Solving is backward-stable (tiny residual) though the solution itself degrades.
    let x_true = [1.0; 8];
    let b = hilbert * Vector::new(x_true);
    let x = f.solve_least_squares(b).unwrap();
    assert!((hilbert * x - b).norm() < 1e-12);
    for value in x.into_array() {
        assert!((value - 1.0).abs() < 1e-2);
    }
}

#[test]
fn damped_solves_ridge_regression() {
    // Ridge (Tikhonov) regression on an ill-conditioned Vandermonde design:
    // (VᵀV + λ²I) x = Vᵀb, which is exactly the damped solve with a constant diagonal.
    let node = |i: usize| -1.0 + 2.0 * i as f64 / 14.0;
    let v = Matrix::<15, 8>::from_fn(|i, j| {
        let t = node(i);
        (0..j).fold(1.0, |acc, _| acc * t)
    });
    let x_true = [0.4, 1.0, -0.6, 0.9, -1.3, 0.5, 0.7, -0.2];
    let b = v * Vector::new(x_true);
    let lambda = 0.1;

    let (x, _) = PivotedQr::decompose(v)
        .unwrap()
        .into_damped(b)
        .solve_with_diagonal(&[lambda; 8]);

    // x satisfies the regularized normal equations.
    let vtv = v.transpose() * v;
    let vtb = v.transpose() * b;
    let lhs =
        Matrix::<8, 8>::from_fn(|r, c| vtv[(r, c)] + if r == c { lambda * lambda } else { 0.0 })
            * x;
    for i in 0..8 {
        assert!((lhs[i] - vtb[i]).abs() < 1e-8);
    }
}
