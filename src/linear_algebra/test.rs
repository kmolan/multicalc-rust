use crate::linear_algebra::qr::{PivotedQr, enorm, max, min};
use crate::linear_algebra::{Matrix, Vector};
use crate::utils::error_codes::CalcError;

/// Asserts every entry of `m` is within tolerance of the identity matrix.
fn approx_identity<const N: usize>(m: Matrix<N, N>) {
    let id: Matrix<N, N> = Matrix::identity();
    for r in 0..N {
        for c in 0..N {
            assert!((m[(r, c)] - id[(r, c)]).abs() < 1e-12);
        }
    }
}

// ----- construction & access -----

#[test]
fn construct_and_access() {
    let v = Vector::new([1.0, 2.0, 3.0]);
    assert_eq!(v[0], 1.0);
    assert_eq!(v.into_array(), [1.0, 2.0, 3.0]);

    let mut w = Vector::from([4.0, 5.0]);
    w[1] = 9.0;
    assert_eq!(w, Vector::new([4.0, 9.0]));

    let z: Vector<3> = Vector::zeros();
    assert_eq!(z, Vector::new([0.0, 0.0, 0.0]));

    assert_eq!(
        Vector::<4>::from_fn(|i| i as f64),
        Vector::new([0.0, 1.0, 2.0, 3.0])
    );

    let mut m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    assert_eq!(m[(1, 0)], 3.0);
    m[(0, 1)] = 7.0;
    assert_eq!(m[(0, 1)], 7.0);

    let id: Matrix<3, 3> = Matrix::identity();
    assert_eq!(
        id.into_array(),
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    );

    assert_eq!(
        Matrix::<2, 2>::from_fn(|r, c| (r * 2 + c) as f64),
        Matrix::new([[0.0, 1.0], [2.0, 3.0]])
    );
}

#[test]
fn try_from_slice_length() {
    assert_eq!(
        Vector::<3>::try_from_slice(&[1.0, 2.0, 3.0]),
        Some(Vector::new([1.0, 2.0, 3.0]))
    );
    assert!(Vector::<3>::try_from_slice(&[1.0, 2.0]).is_none());

    assert_eq!(
        Matrix::<2, 2>::try_from_row_slice(&[1.0, 2.0, 3.0, 4.0]),
        Some(Matrix::new([[1.0, 2.0], [3.0, 4.0]]))
    );
    assert!(Matrix::<2, 2>::try_from_row_slice(&[1.0, 2.0, 3.0]).is_none());
}

// ----- vector arithmetic -----

#[test]
fn vector_arithmetic() {
    let a = Vector::new([1.0, 2.0, 3.0]);
    let b = Vector::new([4.0, 5.0, 6.0]);

    assert_eq!(a + b, Vector::new([5.0, 7.0, 9.0]));
    assert_eq!(b - a, Vector::new([3.0, 3.0, 3.0]));
    assert_eq!(-a, Vector::new([-1.0, -2.0, -3.0]));
    assert_eq!(a * 2.0, a.scale(2.0));
    assert_eq!(a.scale(2.0), Vector::new([2.0, 4.0, 6.0]));

    let mut c = a;
    c += b;
    assert_eq!(c, a + b);
    c -= b;
    assert_eq!(c, a);
}

#[test]
fn vector_dot() {
    let a: Vector<3> = Vector::new([1.0, 2.0, 3.0]);
    let b: Vector<3> = Vector::new([4.0, 5.0, 6.0]);
    assert_eq!(a.dot(b), 32.0);
    assert!((a.dot(b) - b.dot(a)).abs() < 1e-12); // symmetry
    assert_eq!(Vector::new([1.0, 0.0]).dot(Vector::new([0.0, 1.0])), 0.0); // orthogonal

    let empty: Vector<0> = Vector::zeros();
    assert_eq!(empty.dot(empty), 0.0);
}

#[test]
fn vector_norm() {
    let v: Vector<2> = Vector::new([3.0, 4.0]);
    assert_eq!(v.norm(), 5.0);
    assert!((v.norm_squared() - v.norm() * v.norm()).abs() < 1e-12);

    let z: Vector<3> = Vector::zeros();
    assert_eq!(z.norm(), 0.0);
    assert!(Vector::new([f64::INFINITY, 0.0]).norm().is_infinite());
}

// ----- cross products & scalar triple -----

#[test]
fn vector_cross_3d() {
    let x = Vector::new([1.0, 0.0, 0.0]);
    let y = Vector::new([0.0, 1.0, 0.0]);
    let z = Vector::new([0.0, 0.0, 1.0]);
    assert_eq!(x.cross(y), z);
    assert_eq!(y.cross(z), x);
    assert_eq!(z.cross(x), y);
    assert_eq!(x.cross(y), -(y.cross(x))); // anti-commutativity

    let a = Vector::new([1.0, 2.0, 3.0]);
    let b = Vector::new([4.0, 5.0, 6.0]);
    let axb = a.cross(b);
    assert_eq!(a.dot(axb), 0.0); // orthogonal to both inputs
    assert_eq!(b.dot(axb), 0.0);
}

#[test]
fn vector_cross_2d_and_scalar_triple() {
    assert_eq!(Vector::new([1.0, 0.0]).cross(Vector::new([0.0, 1.0])), 1.0);
    let a: Vector<2> = Vector::new([2.0, 3.0]);
    let b: Vector<2> = Vector::new([5.0, 7.0]);
    assert!((a.cross(b) + b.cross(a)).abs() < 1e-12); // anti-commutativity
    assert_eq!(a.cross(a), 0.0); // parallel

    let x = Vector::new([1.0, 0.0, 0.0]);
    let y = Vector::new([0.0, 1.0, 0.0]);
    let z = Vector::new([0.0, 0.0, 1.0]);
    assert_eq!(x.scalar_triple(y, z), 1.0);

    let p: Vector<3> = Vector::new([1.0, 2.0, 3.0]);
    let q: Vector<3> = Vector::new([0.0, 1.0, 4.0]);
    let r: Vector<3> = Vector::new([5.0, 6.0, 0.0]);
    assert!((p.scalar_triple(q, r) - q.scalar_triple(r, p)).abs() < 1e-12); // cyclic
    let m = Matrix::new([p.into_array(), q.into_array(), r.into_array()]);
    assert!((p.scalar_triple(q, r) - m.determinant()).abs() < 1e-12); // == det([p; q; r])
}

// ----- matrix arithmetic, multiply, transpose -----

#[test]
fn matrix_arithmetic() {
    let a = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    let b = Matrix::new([[5.0, 6.0], [7.0, 8.0]]);

    assert_eq!(a + b, Matrix::new([[6.0, 8.0], [10.0, 12.0]]));
    assert_eq!(b - a, Matrix::new([[4.0, 4.0], [4.0, 4.0]]));
    assert_eq!(-a, Matrix::new([[-1.0, -2.0], [-3.0, -4.0]]));
    assert_eq!(a * 2.0, a.scale(2.0));

    let mut c = a;
    c += b;
    assert_eq!(c, a + b);
    c -= b;
    assert_eq!(c, a);
}

#[test]
fn matrix_multiply() {
    let a = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    let b = Matrix::new([[5.0, 6.0], [7.0, 8.0]]);
    let id: Matrix<2, 2> = Matrix::identity();

    assert_eq!(a * id, a);
    assert_eq!(id * a, a);
    assert_eq!((a * b) * a, a * (b * a)); // associativity

    // non-square 2x3 * 3x2
    let p = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let q = Matrix::new([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);
    assert_eq!(p * q, Matrix::new([[58.0, 64.0], [139.0, 154.0]]));

    // matrix x vector
    assert_eq!(a * Vector::new([1.0, 1.0]), Vector::new([3.0, 7.0]));
}

#[test]
fn matrix_transpose() {
    let m = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]); // 2x3
    assert_eq!(
        m.transpose(),
        Matrix::new([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    ); // 3x2
    assert_eq!(m.transpose().transpose(), m); // involution

    let a = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    let b = Matrix::new([[5.0, 6.0], [7.0, 8.0]]);
    assert_eq!((a * b).transpose(), b.transpose() * a.transpose());
}

// ----- determinant & inverse (specialized) -----

#[test]
fn matrix_determinant() {
    let id2: Matrix<2, 2> = Matrix::identity();
    let id3: Matrix<3, 3> = Matrix::identity();
    assert_eq!(id2.determinant(), 1.0);
    assert_eq!(id3.determinant(), 1.0);

    assert_eq!(Matrix::new([[1.0, 2.0], [3.0, 4.0]]).determinant(), -2.0);
    assert_eq!(
        Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]).determinant(),
        -3.0
    );
    assert_eq!(Matrix::new([[1.0, 2.0], [2.0, 4.0]]).determinant(), 0.0); // singular
}

#[test]
fn matrix_inverse() {
    let id2: Matrix<2, 2> = Matrix::identity();
    let id3: Matrix<3, 3> = Matrix::identity();
    assert_eq!(id2.inverse(), Some(id2));
    assert_eq!(id3.inverse(), Some(id3));

    let a = Matrix::new([[4.0, 7.0], [2.0, 6.0]]);
    approx_identity(a * a.inverse().unwrap());

    let b = Matrix::new([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]]);
    approx_identity(b * b.inverse().unwrap());

    // singular -> None
    assert!(Matrix::new([[1.0, 2.0], [2.0, 4.0]]).inverse().is_none());
    let singular3 = Matrix::new([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]]);
    assert_eq!(singular3.determinant(), 0.0);
    assert!(singular3.inverse().is_none());
}

// ----- genericity: the same code at f32 -----

#[test]
fn generic_f32() {
    let a = Vector::<3, f32>::new([1.0, 2.0, 2.0]);
    let b = Vector::<3, f32>::new([2.0, 0.0, 1.0]);
    assert!((a.norm() - 3.0).abs() < 1e-6);
    assert!((a.dot(b) - 4.0).abs() < 1e-6);

    let m = Matrix::<2, 2, f32>::new([[1.0, 2.0], [3.0, 4.0]]);
    let id: Matrix<2, 2, f32> = Matrix::identity();
    assert_eq!(m * id, m);
}

// ----- overflow-safe norm (enorm) and comparison helpers -----

#[test]
fn enorm_matches_naive_norm() {
    // On ordinary values, enorm agrees with the plain sqrt-of-dot norm.
    assert!((enorm(&[3.0_f64, 4.0]) - 5.0).abs() < 1e-12);

    let v = Vector::new([1.0_f64, 2.0, 2.0]);
    assert!((enorm(v.as_array()) - v.norm()).abs() < 1e-12);
}

#[test]
fn enorm_survives_huge_components() {
    // A naive norm would overflow to infinity here; enorm stays finite.
    let result = enorm(&[3.0e200_f64, 4.0e200]);
    assert!(result.is_finite());
    assert!((result / 5.0e200 - 1.0).abs() < 1e-12);
}

#[test]
fn enorm_survives_tiny_components() {
    // A naive norm would underflow to zero here; enorm keeps the magnitude.
    let result = enorm(&[3.0e-200_f64, 4.0e-200]);
    assert!(result > 0.0);
    assert!((result / 5.0e-200 - 1.0).abs() < 1e-12);
}

#[test]
fn enorm_f32_extremes_stay_finite() {
    let big = enorm(&[3.0e30_f32, 4.0e30]);
    assert!(big.is_finite());
    assert!((big / 5.0e30 - 1.0).abs() < 1e-5);

    let small = enorm(&[3.0e-30_f32, 4.0e-30]);
    assert!(small > 0.0);
    assert!((small / 5.0e-30 - 1.0).abs() < 1e-5);
}

#[test]
fn min_max_pick_an_argument() {
    assert_eq!(max(2.0_f64, 3.0), 3.0);
    assert_eq!(max(3.0_f64, 2.0), 3.0);
    assert_eq!(min(2.0_f64, 3.0), 2.0);
    assert_eq!(min(3.0_f64, 2.0), 2.0);

    // An incomparable pair returns the first argument unchanged.
    assert_eq!(max(1.0_f64, f64::NAN), 1.0);
    assert_eq!(min(1.0_f64, f64::NAN), 1.0);
}

// ----- column-pivoted QR (decompose, accessors, solve) -----

#[test]
fn qr_reconstructs_pivoted_matrix() {
    let a = Matrix::<4, 3>::new([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 10.0],
        [2.0, -1.0, 1.0],
    ]);
    let f = PivotedQr::decompose(a).unwrap();
    let perm = f.permutation();

    // The most-normed column (index 2) pivots to the front, so |r_diag[0]| is its norm.
    assert_eq!(perm[0], 2);
    assert!((f.r_diag[0].abs() - 146.0_f64.sqrt()).abs() < 1e-12);

    // Column norms are the original ones, in original order.
    assert!((f.column_norms[0] - 70.0_f64.sqrt()).abs() < 1e-12);
    assert!((f.column_norms[1] - 94.0_f64.sqrt()).abs() < 1e-12);
    assert!((f.column_norms[2] - 146.0_f64.sqrt()).abs() < 1e-12);

    let q = f.q();
    let r = f.r();

    // R is upper-triangular.
    for row in 0..3 {
        for col in 0..row {
            assert_eq!(r[(row, col)], 0.0);
        }
    }

    // Q has orthonormal columns.
    approx_identity(q.transpose() * q);

    // A * P == Q * R.
    let ap = Matrix::<4, 3>::from_fn(|i, c| a[(i, perm[c])]);
    let product = q * r;
    for i in 0..4 {
        for c in 0..3 {
            assert!((ap[(i, c)] - product[(i, c)]).abs() < 1e-12);
        }
    }
}

#[test]
fn qr_rejects_underdetermined() {
    let a = Matrix::<2, 3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    assert!(matches!(
        PivotedQr::decompose(a),
        Err(CalcError::Underdetermined)
    ));
}

#[test]
fn qr_handles_zero_column() {
    // Column 1 is entirely zero: decompose must not divide by its zero norm.
    let a = Matrix::<4, 3>::new([
        [1.0, 0.0, 2.0],
        [3.0, 0.0, 4.0],
        [5.0, 0.0, 6.0],
        [7.0, 0.0, 8.0],
    ]);
    let f = PivotedQr::decompose(a).unwrap();
    let perm = f.permutation();

    // The zero column sorts last and carries a zero diagonal.
    assert_eq!(perm[2], 1);
    assert!(f.r_diag[2].abs() < 1e-12);

    // The factorization still reproduces A * P.
    let ap = Matrix::<4, 3>::from_fn(|i, c| a[(i, perm[c])]);
    let product = f.q() * f.r();
    for i in 0..4 {
        for c in 0..3 {
            assert!((ap[(i, c)] - product[(i, c)]).abs() < 1e-12);
        }
    }
}

#[test]
fn qr_solves_square_system() {
    // A x = b with the exact solution x = [1, 1, 1].
    let a = Matrix::<3, 3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]);
    let b = Vector::new([6.0, 15.0, 25.0]);
    let x = PivotedQr::decompose(a).unwrap().solve_least_squares(b).unwrap();
    for value in x.into_array() {
        assert!((value - 1.0).abs() < 1e-12);
    }
}

#[test]
fn qr_solves_overdetermined_least_squares() {
    // Fit y = m t + c to three non-collinear points; least-squares gives m = 0.5, c = 7/6.
    let a = Matrix::<3, 2>::new([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]);
    let b = Vector::new([1.0, 2.0, 2.0]);
    let x = PivotedQr::decompose(a).unwrap().solve_least_squares(b).unwrap();
    assert!((x[0] - 0.5).abs() < 1e-12);
    assert!((x[1] - 7.0 / 6.0).abs() < 1e-12);
}

#[test]
fn qr_solve_reports_singular() {
    // The middle column is zero, so R has an exactly-zero diagonal entry: rank-deficient.
    let a = Matrix::<3, 3>::new([[1.0, 0.0, 2.0], [3.0, 0.0, 4.0], [5.0, 0.0, 6.0]]);
    let b = Vector::new([1.0, 2.0, 3.0]);
    let f = PivotedQr::decompose(a).unwrap();
    assert!(matches!(
        f.solve_least_squares(b),
        Err(CalcError::SingularMatrix)
    ));
}

#[test]
fn qr_solve_reports_rank_deficient() {
    // col2 = col0 + col1: dependent columns leave a tiny (not exactly zero) diagonal, which
    // the relative rank tolerance still flags.
    let a = Matrix::<3, 3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 9.0], [7.0, 8.0, 15.0]]);
    let b = Vector::new([1.0, 2.0, 3.0]);
    let f = PivotedQr::decompose(a).unwrap();
    assert!(matches!(
        f.solve_least_squares(b),
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
    let lhs = Matrix::<3, 3>::from_fn(|r, c| {
        jtj[(r, c)] + if r == c { diag[r] * diag[r] } else { 0.0 }
    }) * x;
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
fn damped_max_a_t_b_scaled_matches_direct() {
    let (j, b) = sample_problem();
    let dls = PivotedQr::decompose(j).unwrap().into_damped(b);
    let b_norm = b.norm();

    // Direct: max over columns of |Jᵀb|ₗ / (b_norm · ‖columnₗ‖).
    let jtb = j.transpose() * b;
    let mut expected = 0.0_f64;
    for l in 0..3 {
        let scaled = (jtb[l] / b_norm / j.column(l).norm()).abs();
        expected = expected.max(scaled);
    }
    assert!((dls.max_a_t_b_scaled(b_norm) - expected).abs() < 1e-12);
}

#[test]
fn damped_a_x_norm_matches_direct() {
    let (j, b) = sample_problem();
    let dls = PivotedQr::decompose(j).unwrap().into_damped(b);
    let x = Vector::new([1.0, -2.0, 0.5]);
    // a_x_norm(x) is ‖J x‖.
    assert!((dls.a_x_norm(&x) - (j * x).norm()).abs() < 1e-12);
}

#[test]
fn damped_is_non_singular() {
    let (j, b) = sample_problem();
    assert!(PivotedQr::decompose(j).unwrap().into_damped(b).is_non_singular());

    // A zero column makes the problem rank-deficient.
    let deficient = Matrix::<4, 3>::new([
        [1.0, 0.0, 2.0],
        [3.0, 0.0, 4.0],
        [5.0, 0.0, 6.0],
        [7.0, 0.0, 8.0],
    ]);
    let dls = PivotedQr::decompose(deficient).unwrap().into_damped(b);
    assert!(!dls.is_non_singular());
}

#[test]
fn damped_cholesky_factor_matches_normal_matrix() {
    let (j, b) = sample_problem();
    let diag = [1.0, 0.5, 2.0];
    let dls = PivotedQr::decompose(j).unwrap().into_damped(b);
    let (_, cf) = dls.solve_with_diagonal(&diag);

    // Reconstruct S (upper triangular) from the factor.
    let s = Matrix::<3, 3>::from_fn(|row, col| {
        if row == col {
            cf.s_diag[row]
        } else if col > row {
            cf.s[(col, row)]
        } else {
            0.0
        }
    });

    // SᵀS must equal RᵀR + D², with D permuted the way qrsolv applies it.
    let sts = s.transpose() * s;
    let rtr = dls.r.transpose() * dls.r;
    for row in 0..3 {
        for col in 0..3 {
            let mut expected = rtr[(row, col)];
            if row == col {
                let d = diag[dls.permutation[row]];
                expected += d * d;
            }
            assert!((sts[(row, col)] - expected).abs() < 1e-9);
        }
    }
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
    approx_identity(q.transpose() * q);
    let ap = Matrix::<8, 8>::from_fn(|i, c| hilbert[(i, perm[c])]);
    let product = q * r;
    for i in 0..8 {
        for c in 0..8 {
            assert!((ap[(i, c)] - product[(i, c)]).abs() < 1e-12);
        }
    }

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
fn enorm_handles_extreme_dynamic_range() {
    // Twelve large components: a naive sum of squares would overflow to infinity.
    let many_large = [1.0e160_f64; 12];
    let result = enorm(&many_large);
    assert!(result.is_finite());
    assert!((result / (12.0_f64.sqrt() * 1.0e160) - 1.0).abs() < 1e-12);

    // A vector mixing all three magnitude bands; the large band sets the norm.
    let mut mixed = [0.0_f64; 16];
    mixed[0] = 3.0e160;
    mixed[1] = 4.0e160;
    mixed[2] = 1.0;
    mixed[3] = 1.0;
    mixed[4] = 3.0e-160;
    mixed[5] = 4.0e-160;
    let norm = enorm(&mixed);
    assert!(norm.is_finite());
    assert!((norm / 5.0e160 - 1.0).abs() < 1e-12);
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
    let lhs = Matrix::<8, 8>::from_fn(|r, c| {
        vtv[(r, c)] + if r == c { lambda * lambda } else { 0.0 }
    }) * x;
    for i in 0..8 {
        assert!((lhs[i] - vtb[i]).abs() < 1e-8);
    }
}
