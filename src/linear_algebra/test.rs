use crate::linear_algebra::qr::{PivotedQr, enorm, max, min};
use crate::linear_algebra::{Matrix, Vector};
use crate::scalar::Numeric;
use crate::utils::error_codes::CalcError;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::sync::atomic::{AtomicUsize, Ordering};

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
    assert_eq!(id2.inverse(), Ok(id2));
    assert_eq!(id3.inverse(), Ok(id3));

    let a = Matrix::new([[4.0, 7.0], [2.0, 6.0]]);
    approx_identity(a * a.inverse().unwrap());

    let b = Matrix::new([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]]);
    approx_identity(b * b.inverse().unwrap());

    // singular -> Err(SingularMatrix)
    let singular2 = Matrix::new([[1.0, 2.0], [2.0, 4.0]]);
    assert_eq!(singular2.inverse(), Err(CalcError::SingularMatrix));
    let singular3 = Matrix::new([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]]);
    assert_eq!(singular3.determinant(), 0.0);
    assert_eq!(singular3.inverse(), Err(CalcError::SingularMatrix));
}

#[test]
fn matrix_4x4_determinant_and_inverse() {
    // Upper-triangular: the determinant is the product of the diagonal.
    let upper = Matrix::<4, 4>::new([
        [2.0, 1.0, 1.0, 1.0],
        [0.0, 3.0, 1.0, 1.0],
        [0.0, 0.0, 4.0, 1.0],
        [0.0, 0.0, 0.0, 5.0],
    ]);
    assert_eq!(upper.determinant(), 120.0);

    // Reference determinant and inverse from an exact rational solve.
    let a = Matrix::<4, 4>::new([
        [4.0, 3.0, 2.0, 1.0],
        [3.0, 4.0, 3.0, 2.0],
        [2.0, 3.0, 4.0, 3.0],
        [1.0, 2.0, 3.0, 4.0],
    ]);
    assert_eq!(a.determinant(), 20.0);

    let expected_inv = [
        [0.6, -0.5, 0.0, 0.1],
        [-0.5, 1.0, -0.5, 0.0],
        [0.0, -0.5, 1.0, -0.5],
        [0.1, 0.0, -0.5, 0.6],
    ];
    let inv = a.inverse().unwrap();
    for r in 0..4 {
        for c in 0..4 {
            assert!((inv[(r, c)] - expected_inv[r][c]).abs() < 1e-12);
        }
    }
    approx_identity(a * inv);

    // A non-symmetric matrix, so its (non-symmetric) inverse catches any transpose error in
    // the adjugate. Reference from an exact rational solve.
    let b = Matrix::<4, 4>::new([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 1.0, 0.0, 1.0],
        [0.0, 3.0, 1.0, 2.0],
        [1.0, 0.0, 2.0, 1.0],
    ]);
    assert_eq!(b.determinant(), -20.0);

    let expected_b_inv = [
        [-0.15, 0.45, -0.05, 0.25],
        [-0.35, 0.05, 0.55, 0.25],
        [-0.25, -0.25, 0.25, 0.75],
        [0.65, 0.05, -0.45, -0.75],
    ];
    let b_inv = b.inverse().unwrap();
    for r in 0..4 {
        for c in 0..4 {
            assert!((b_inv[(r, c)] - expected_b_inv[r][c]).abs() < 1e-12);
        }
    }
    approx_identity(b * b_inv);
    approx_identity(b_inv * b);

    // Rows in arithmetic progression are rank-deficient.
    let singular = Matrix::<4, 4>::new([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ]);
    assert_eq!(singular.determinant(), 0.0);
    assert_eq!(singular.inverse(), Err(CalcError::SingularMatrix));

    // The same code at f32 round-trips to the identity.
    let af = Matrix::<4, 4, f32>::new([
        [4.0, 3.0, 2.0, 1.0],
        [3.0, 4.0, 3.0, 2.0],
        [2.0, 3.0, 4.0, 3.0],
        [1.0, 2.0, 3.0, 4.0],
    ]);
    let pf = af * af.inverse().unwrap();
    let idf: Matrix<4, 4, f32> = Matrix::identity();
    for r in 0..4 {
        for c in 0..4 {
            assert!((pf[(r, c)] - idf[(r, c)]).abs() < 1e-5);
        }
    }
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
    assert!(
        PivotedQr::decompose(j)
            .unwrap()
            .into_damped(b)
            .is_non_singular()
    );

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
    let lhs =
        Matrix::<8, 8>::from_fn(|r, c| vtv[(r, c)] + if r == c { lambda * lambda } else { 0.0 })
            * x;
    for i in 0..8 {
        assert!((lhs[i] - vtb[i]).abs() < 1e-8);
    }
}

// ----- LU decomposition (Doolittle, partial pivoting) -----

// Factorizes `a` and checks the factors are triangular and reconstruct `P·A`.
fn lu_reconstructs<const N: usize>(a: Matrix<N, N>) {
    let f = a.lu().unwrap();
    let l = f.l();
    let u = f.u();
    let perm = f.permutation();

    // L is unit lower-triangular; U is upper-triangular.
    for r in 0..N {
        assert_eq!(l[(r, r)], 1.0);
        for c in (r + 1)..N {
            assert_eq!(l[(r, c)], 0.0);
        }
        for c in 0..r {
            assert_eq!(u[(r, c)], 0.0);
        }
    }

    // P·A == L·U.
    let pa = Matrix::<N, N>::from_fn(|i, c| a[(perm[i], c)]);
    let prod = l * u;
    for i in 0..N {
        for c in 0..N {
            assert!((pa[(i, c)] - prod[(i, c)]).abs() < 1e-12);
        }
    }
}

#[test]
fn lu_reconstructs_pivoted_matrix() {
    // The largest first-column entry is in the last row, forcing a swap.
    lu_reconstructs(Matrix::<3, 3>::new([
        [2.0, 1.0, 1.0],
        [4.0, 3.0, 3.0],
        [8.0, 7.0, 9.0],
    ]));
    lu_reconstructs(Matrix::<4, 4>::new([
        [4.0, 3.0, 2.0, 1.0],
        [3.0, 4.0, 3.0, 2.0],
        [2.0, 3.0, 4.0, 3.0],
        [1.0, 2.0, 3.0, 4.0],
    ]));
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
    assert_eq!(zero_col.lu().err(), Some(CalcError::SingularMatrix));

    // Dependent rows drive a pivot to zero during elimination.
    let dependent = Matrix::<2, 2>::new([[1.0, 2.0], [2.0, 4.0]]);
    assert_eq!(dependent.lu().err(), Some(CalcError::SingularMatrix));
}

#[test]
fn lu_f32_reconstructs() {
    let a = Matrix::<3, 3, f32>::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]);
    let f = a.lu().unwrap();
    let pa = Matrix::<3, 3, f32>::from_fn(|i, c| a[(f.permutation()[i], c)]);
    let prod = f.l() * f.u();
    for i in 0..3 {
        for c in 0..3 {
            assert!((pa[(i, c)] - prod[(i, c)]).abs() < 1e-5);
        }
    }
}

#[test]
fn lu_solves_system() {
    let a = Matrix::<3, 3>::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]);
    // A·x = b has the exact solution x = [1, 2, 3].
    let b = Vector::new([7.0, 19.0, 49.0]);
    let x = a.lu().unwrap().solve(b);
    assert!((x[0] - 1.0).abs() < 1e-12);
    assert!((x[1] - 2.0).abs() < 1e-12);
    assert!((x[2] - 3.0).abs() < 1e-12);
    // The residual is tiny.
    assert!((a * x - b).norm() < 1e-12);
}

#[test]
fn lu_solve_matrix_multi_rhs() {
    let a = Matrix::<3, 3>::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]);
    let f = a.lu().unwrap();
    let rhs = Matrix::<3, 2>::new([[7.0, 4.0], [19.0, 10.0], [49.0, 26.0]]);
    let x = f.solve_matrix(rhs);
    // A·X == B.
    let prod = a * x;
    for r in 0..3 {
        for c in 0..2 {
            assert!((prod[(r, c)] - rhs[(r, c)]).abs() < 1e-12);
        }
    }
    // Each column agrees with a single-RHS solve.
    for c in 0..2 {
        let single = f.solve(rhs.column(c));
        for r in 0..3 {
            assert!((x[(r, c)] - single[r]).abs() < 1e-12);
        }
    }
}

#[test]
fn lu_inverse_matches_direct() {
    // LU inverse agrees with the direct closed-form inverses for 2×2, 3×3, and 4×4.
    let a2 = Matrix::<2, 2>::new([[4.0, 7.0], [2.0, 6.0]]);
    approx_identity(a2.lu().unwrap().inverse() * a2);
    let d2 = a2.inverse().unwrap();
    for r in 0..2 {
        for c in 0..2 {
            assert!((a2.lu().unwrap().inverse()[(r, c)] - d2[(r, c)]).abs() < 1e-12);
        }
    }

    let a3 = Matrix::<3, 3>::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]);
    let d3 = a3.inverse().unwrap();
    let lu3 = a3.lu().unwrap().inverse();
    for r in 0..3 {
        for c in 0..3 {
            assert!((lu3[(r, c)] - d3[(r, c)]).abs() < 1e-12);
        }
    }

    let a4 = Matrix::<4, 4>::new([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 1.0, 0.0, 1.0],
        [0.0, 3.0, 1.0, 2.0],
        [1.0, 0.0, 2.0, 1.0],
    ]);
    let d4 = a4.inverse().unwrap();
    let lu4 = a4.lu().unwrap().inverse();
    for r in 0..4 {
        for c in 0..4 {
            assert!((lu4[(r, c)] - d4[(r, c)]).abs() < 1e-12);
        }
    }
}

#[test]
fn lu_inverse_matches_reference_5x5() {
    // A non-symmetric 5×5; reference inverse from an exact rational solve.
    let a = Matrix::<5, 5>::new([
        [5.0, 1.0, 0.0, 2.0, 1.0],
        [1.0, 6.0, 2.0, 0.0, 1.0],
        [3.0, 2.0, 7.0, 1.0, 0.0],
        [2.0, 0.0, 1.0, 8.0, 2.0],
        [1.0, 4.0, 0.0, 2.0, 9.0],
    ]);
    assert!((a.lu().unwrap().determinant() - 10406.0).abs() < 1e-9);

    let expected = [
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
    ];
    let inv = a.lu().unwrap().inverse();
    for r in 0..5 {
        for c in 0..5 {
            assert!((inv[(r, c)] - expected[r][c]).abs() < 1e-12);
        }
    }
    approx_identity(a * inv);
}

// ----- Cholesky -----

fn cholesky_reconstructs<const N: usize>(a: Matrix<N, N>) {
    let l = a.cholesky().unwrap().l();

    // L is lower-triangular with a strictly positive diagonal.
    for r in 0..N {
        assert!(l[(r, r)] > 0.0);
        for c in (r + 1)..N {
            assert_eq!(l[(r, c)], 0.0);
        }
    }

    // L·Lᵀ == A.
    let prod = l * l.transpose();
    for r in 0..N {
        for c in 0..N {
            assert!((prod[(r, c)] - a[(r, c)]).abs() < 1e-12);
        }
    }
}

#[test]
fn cholesky_reconstructs_spd() {
    cholesky_reconstructs(Matrix::<2, 2>::new([[4.0, 2.0], [2.0, 3.0]]));
    cholesky_reconstructs(Matrix::<3, 3>::new([
        [4.0, 12.0, -16.0],
        [12.0, 37.0, -43.0],
        [-16.0, -43.0, 98.0],
    ]));
    // An M·Mᵀ product is symmetric positive-definite for full-rank M.
    let m = Matrix::<4, 4>::new([
        [2.0, 0.0, 0.0, 0.0],
        [1.0, 3.0, 0.0, 0.0],
        [-1.0, 2.0, 4.0, 0.0],
        [0.0, 1.0, -2.0, 5.0],
    ]);
    cholesky_reconstructs(m * m.transpose());
}

#[test]
fn cholesky_matches_reference() {
    let a = Matrix::<3, 3>::new([
        [4.0, 12.0, -16.0],
        [12.0, 37.0, -43.0],
        [-16.0, -43.0, 98.0],
    ]);
    let l = a.cholesky().unwrap().l();
    let expected = [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]];
    for r in 0..3 {
        for c in 0..3 {
            assert!((l[(r, c)] - expected[r][c]).abs() < 1e-12);
        }
    }
}

#[test]
fn cholesky_rejects_non_pd() {
    // Symmetric but indefinite (eigenvalues 3 and -1).
    let indefinite = Matrix::<2, 2>::new([[1.0, 2.0], [2.0, 1.0]]);
    assert_eq!(
        indefinite.cholesky().err(),
        Some(CalcError::NotPositiveDefinite)
    );

    // Negative leading diagonal entry.
    let negative = Matrix::<2, 2>::new([[-4.0, 0.0], [0.0, 1.0]]);
    assert_eq!(
        negative.cholesky().err(),
        Some(CalcError::NotPositiveDefinite)
    );

    // Singular: the second radicand collapses to zero.
    let singular = Matrix::<2, 2>::new([[1.0, 1.0], [1.0, 1.0]]);
    assert_eq!(
        singular.cholesky().err(),
        Some(CalcError::NotPositiveDefinite)
    );
}

#[test]
fn cholesky_solves_system() {
    let a = Matrix::<3, 3>::new([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]]);
    let x_exact = Vector::new([1.0, -2.0, 3.0]);
    let b = a * x_exact;
    let x = a.cholesky().unwrap().solve(b);
    for i in 0..3 {
        assert!((x[i] - x_exact[i]).abs() < 1e-12);
    }
    assert!((a * x - b).norm() < 1e-12);

    // Agrees with the LU solve on the same SPD system.
    let lu_x = a.lu().unwrap().solve(b);
    for i in 0..3 {
        assert!((x[i] - lu_x[i]).abs() < 1e-12);
    }
}

#[test]
fn cholesky_solve_matrix_multi_rhs() {
    let a = Matrix::<2, 2>::new([[4.0, 2.0], [2.0, 3.0]]);
    let f = a.cholesky().unwrap();
    let rhs = Matrix::<2, 3>::new([[8.0, 6.0, 4.0], [8.0, 5.0, 3.0]]);
    let x = f.solve_matrix(rhs);
    // A·X == B.
    let prod = a * x;
    for r in 0..2 {
        for c in 0..3 {
            assert!((prod[(r, c)] - rhs[(r, c)]).abs() < 1e-12);
        }
    }
    // Each column agrees with a single-RHS solve.
    for c in 0..3 {
        let single = f.solve(rhs.column(c));
        for r in 0..2 {
            assert!((x[(r, c)] - single[r]).abs() < 1e-12);
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
    approx_identity(inv * a);
    approx_identity(a * inv);

    let lu_inv = a.lu().unwrap().inverse();
    for r in 0..3 {
        for c in 0..3 {
            assert!((inv[(r, c)] - lu_inv[(r, c)]).abs() < 1e-12);
        }
    }
}

#[test]
fn matrix_solve_agrees_with_lu() {
    let a = Matrix::<3, 3>::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]);
    let b = Vector::new([7.0, 19.0, 49.0]);
    let x = a.solve(b).unwrap();

    // The convenience solver matches an explicit LU solve.
    let lu_x = a.lu().unwrap().solve(b);
    for i in 0..3 {
        assert!((x[i] - lu_x[i]).abs() < 1e-12);
    }
    assert!((a * x - b).norm() < 1e-12);

    // A singular system is rejected.
    let singular = Matrix::<2, 2>::new([[1.0, 2.0], [2.0, 4.0]]);
    assert_eq!(
        singular.solve(Vector::new([1.0, 2.0])).err(),
        Some(CalcError::SingularMatrix)
    );
}

#[test]
fn cholesky_f32_reconstructs() {
    let a = Matrix::<3, 3, f32>::new([
        [4.0, 12.0, -16.0],
        [12.0, 37.0, -43.0],
        [-16.0, -43.0, 98.0],
    ]);
    let l = a.cholesky().unwrap().l();
    let prod = l * l.transpose();
    for r in 0..3 {
        for c in 0..3 {
            assert!((prod[(r, c)] - a[(r, c)]).abs() < 1e-3);
        }
    }
}

// ----- work-count regression guard -----

// A scalar that tallies every multiply and divide it performs, so a test can pin the arithmetic
// work of a factorization to a fixed count. That count is a deterministic function of the matrix
// size, independent of wall-clock timing, so it fails if an algorithm starts doing more work than
// the benches/linear_algebra.md figures assume. Only `factorization_work_counts` touches the
// counter, and it
// runs single-threaded, so the shared tally needs no synchronization beyond atomicity.
static MUL_DIV_OPS: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
struct Counted(f64);

impl Counted {
    fn tick() {
        MUL_DIV_OPS.fetch_add(1, Ordering::Relaxed);
    }
}

impl Add for Counted {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Counted(self.0 + rhs.0)
    }
}
impl Sub for Counted {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Counted(self.0 - rhs.0)
    }
}
impl Mul for Counted {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::tick();
        Counted(self.0 * rhs.0)
    }
}
impl Div for Counted {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self::tick();
        Counted(self.0 / rhs.0)
    }
}
impl Neg for Counted {
    type Output = Self;
    fn neg(self) -> Self {
        Counted(-self.0)
    }
}
impl AddAssign for Counted {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}
impl SubAssign for Counted {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}
impl MulAssign for Counted {
    fn mul_assign(&mut self, rhs: Self) {
        Self::tick();
        self.0 *= rhs.0;
    }
}
impl DivAssign for Counted {
    fn div_assign(&mut self, rhs: Self) {
        Self::tick();
        self.0 /= rhs.0;
    }
}

impl Numeric for Counted {
    const ZERO: Self = Counted(0.0);
    const ONE: Self = Counted(1.0);
    const TWO: Self = Counted(2.0);
    const HALF: Self = Counted(0.5);
    const PI: Self = Counted(core::f64::consts::PI);
    const EPSILON: Self = Counted(f64::EPSILON);
    const NAN: Self = Counted(f64::NAN);
    const INFINITY: Self = Counted(f64::INFINITY);
    const NEG_INFINITY: Self = Counted(f64::NEG_INFINITY);
    const MAX: Self = Counted(f64::MAX);
    const MIN_POSITIVE: Self = Counted(f64::MIN_POSITIVE);

    fn from_f64(value: f64) -> Self {
        Counted(value)
    }
    fn from_u64(value: u64) -> Self {
        Counted(value as f64)
    }
    fn from_usize(value: usize) -> Self {
        Counted(value as f64)
    }

    fn abs(self) -> Self {
        Counted(libm::fabs(self.0))
    }
    fn sqrt(self) -> Self {
        Counted(libm::sqrt(self.0))
    }
    fn sin(self) -> Self {
        Counted(libm::sin(self.0))
    }
    fn cos(self) -> Self {
        Counted(libm::cos(self.0))
    }
    fn tan(self) -> Self {
        Counted(libm::tan(self.0))
    }
    fn exp(self) -> Self {
        Counted(libm::exp(self.0))
    }
    fn ln(self) -> Self {
        Counted(libm::log(self.0))
    }

    fn is_nan(self) -> bool {
        self.0.is_nan()
    }
    fn is_finite(self) -> bool {
        self.0.is_finite()
    }
}

#[test]
fn factorization_work_counts() {
    // Symmetric positive-definite and invertible, so LU, Cholesky, and the direct 4x4 inverse all
    // apply. The multiply/divide counts below are fixed functions of the size — for a 4x4:
    //   LU:       divisions N(N-1)/2 = 6, multiplications sum_{p<N} p^2 = 14  -> 20
    //   Cholesky: multiplications 10, divisions 6                            -> 16
    // and the direct inverse is a cofactor expansion. If any of these change, revisit
    // benches/linear_algebra.md.
    let a =
        Matrix::<4, 4, Counted>::from_fn(|i, j| if i == j { Counted(4.0) } else { Counted(1.0) });

    let measure = |f: &dyn Fn()| {
        MUL_DIV_OPS.store(0, Ordering::Relaxed);
        f();
        MUL_DIV_OPS.load(Ordering::Relaxed)
    };

    let lu = measure(&|| {
        let _ = a.lu().unwrap();
    });
    let cholesky = measure(&|| {
        let _ = a.cholesky().unwrap();
    });
    let inverse = measure(&|| {
        let _ = a.inverse().unwrap();
    });

    // One-sided Jacobi SVD on a fixed 3x3 (a deterministic sweep sequence).
    let a3 = Matrix::<3, 3, Counted>::from_fn(|i, j| {
        Counted([[4.0, 1.0, 2.0], [1.0, 5.0, 3.0], [2.0, 3.0, 6.0]][i][j])
    });
    let svd = measure(&|| {
        let _ = a3.svd().unwrap();
    });

    assert_eq!((lu, cholesky, inverse), (20, 16, 95));
    // One-sided Jacobi converges in a fixed number of sweeps for this input.
    assert_eq!(svd, 441);
}

// ----- SVD -----

fn svd_reconstructs<const M: usize, const N: usize>(a: Matrix<M, N>) {
    let f = a.svd().unwrap();
    let (u, s, v) = (f.u(), f.singular_values(), f.v());

    // Singular values are non-negative and descending.
    for k in 0..N {
        assert!(s[k] >= 0.0);
        if k + 1 < N {
            assert!(s[k] >= s[k + 1]);
        }
    }

    // U and V have orthonormal columns.
    approx_identity(u.transpose() * u);
    approx_identity(v.transpose() * v);

    // U · diag(σ) · Vᵀ == A.
    for r in 0..M {
        for c in 0..N {
            let mut acc = 0.0;
            for k in 0..N {
                acc += u[(r, k)] * s[k] * v[(c, k)];
            }
            assert!((acc - a[(r, c)]).abs() < 1e-12);
        }
    }
}

#[test]
fn svd_reconstructs_various() {
    svd_reconstructs(Matrix::<3, 2>::new([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]));
    svd_reconstructs(Matrix::<3, 3>::new([
        [4.0, 1.0, 2.0],
        [1.0, 5.0, 3.0],
        [2.0, 3.0, 6.0],
    ]));
    svd_reconstructs(Matrix::<4, 3>::new([
        [1.0, 0.0, 2.0],
        [0.0, 3.0, 1.0],
        [4.0, 1.0, 0.0],
        [2.0, 1.0, 5.0],
    ]));
    // Larger, well-conditioned tall matrices.
    svd_reconstructs(Matrix::<12, 6>::from_fn(|i, j| {
        if i == j {
            10.0
        } else {
            1.0 / (1.0 + (i + j) as f64)
        }
    }));
    svd_reconstructs(Matrix::<20, 6>::from_fn(|i, j| {
        if i == j {
            8.0
        } else {
            (i as f64 - j as f64) / (5.0 + (i + j) as f64)
        }
    }));
}

#[test]
fn svd_matches_reference() {
    // A symmetric matrix built from a known spectrum: A = R · diag(σ) · Rᵀ with R a proper
    // rotation, so the singular values are exactly [6, 3, 1].
    let r = Matrix::<3, 3>::new([
        [2.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0],
        [1.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0],
        [2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0],
    ]);
    let d = Matrix::<3, 3>::new([[6.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]]);
    let a = r * d * r.transpose();
    let s = a.svd().unwrap().singular_values();
    let expected = [6.0, 3.0, 1.0];
    for k in 0..3 {
        assert!((s[k] - expected[k]).abs() < 1e-12);
    }
}

#[test]
fn svd_diagonal_singular_values() {
    // Diagonal input: singular values are the sorted absolute diagonal.
    let a = Matrix::<4, 4>::from_fn(|i, j| {
        if i == j {
            [3.0, -5.0, 2.0, -1.0][i]
        } else {
            0.0
        }
    });
    let s = a.svd().unwrap().singular_values();
    let expected = [5.0, 3.0, 2.0, 1.0];
    for k in 0..4 {
        assert!((s[k] - expected[k]).abs() < 1e-12);
    }
}

fn svd_moore_penrose<const M: usize, const N: usize>(a: Matrix<M, N>) {
    let ap = a.pseudo_inverse().unwrap();
    let aapa = a * ap * a;
    let papap = ap * a * ap;
    let aap = a * ap;
    let apa = ap * a;
    for r in 0..M {
        for c in 0..N {
            assert!((aapa[(r, c)] - a[(r, c)]).abs() < 1e-10);
        }
    }
    for r in 0..N {
        for c in 0..M {
            assert!((papap[(r, c)] - ap[(r, c)]).abs() < 1e-10);
        }
    }
    for r in 0..M {
        for c in 0..M {
            assert!((aap[(r, c)] - aap[(c, r)]).abs() < 1e-12);
        }
    }
    for r in 0..N {
        for c in 0..N {
            assert!((apa[(r, c)] - apa[(c, r)]).abs() < 1e-12);
        }
    }
}

#[test]
fn svd_pseudo_inverse_conditions() {
    svd_moore_penrose(Matrix::<3, 2>::new([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]));
    svd_moore_penrose(Matrix::<2, 3>::new([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]]));
    svd_moore_penrose(Matrix::<3, 3>::new([
        [4.0, 1.0, 2.0],
        [1.0, 5.0, 3.0],
        [2.0, 3.0, 6.0],
    ]));
    svd_moore_penrose(Matrix::<12, 6>::from_fn(|i, j| {
        if i == j {
            10.0
        } else {
            1.0 / (1.0 + (i + j) as f64)
        }
    }));
}

#[test]
fn svd_rank_deficient() {
    // Column 2 is twice column 1: rank 1, one exactly-zero singular value.
    let a = Matrix::<3, 2>::new([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]);
    let f = a.svd().unwrap();
    assert_eq!(f.rank(1e-9), 1);
    assert!(f.condition_number().is_infinite());

    // Truncated pseudo-inverse stays finite.
    let ap = f.pseudo_inverse();
    for r in 0..2 {
        for c in 0..3 {
            assert!(ap[(r, c)].is_finite());
        }
    }

    // Consistent system: the min-norm solution reproduces the range and equals A⁺·b.
    let b = a * Vector::new([1.0, 0.5]);
    let x = f.solve(b);
    assert!((a * x - b).norm() < 1e-10);
    let x_pinv = ap * b;
    for i in 0..2 {
        assert!((x[i] - x_pinv[i]).abs() < 1e-12);
    }
}

#[test]
fn svd_f32_reconstructs() {
    let a = Matrix::<3, 3, f32>::new([[4.0, 1.0, 2.0], [1.0, 5.0, 3.0], [2.0, 3.0, 6.0]]);
    let f = a.svd().unwrap();
    let (u, s, v) = (f.u(), f.singular_values(), f.v());
    for r in 0..3 {
        for c in 0..3 {
            let mut acc = 0.0f32;
            for k in 0..3 {
                acc += u[(r, k)] * s[k] * v[(c, k)];
            }
            assert!((acc - a[(r, c)]).abs() < 1e-4);
        }
    }
}

#[test]
fn svd_error_paths() {
    // A wide matrix is rejected by the raw svd().
    let wide = Matrix::<2, 3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    assert_eq!(wide.svd().err(), Some(CalcError::Underdetermined));

    // Non-finite entries are rejected.
    let mut nan = Matrix::<3, 2>::new([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    nan[(2, 1)] = f64::NAN;
    assert_eq!(nan.svd().err(), Some(CalcError::NonFiniteValue));
    let mut inf = Matrix::<3, 2>::new([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    inf[(0, 0)] = f64::INFINITY;
    assert_eq!(inf.svd().err(), Some(CalcError::NonFiniteValue));
}

#[test]
fn svd_kabsch_rotation_recovery() {
    // A proper rotation (orthonormal, determinant +1).
    let rot = Matrix::<3, 3>::new([
        [2.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0],
        [1.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0],
        [2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0],
    ]);
    // Body points spanning 3D.
    let pts = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [2.0, -1.0, 0.5],
        [-1.0, 0.5, 2.0],
    ];
    // Cross-covariance H = Σ (R·p) pᵀ.
    let mut h = Matrix::<3, 3>::zeros();
    for p in pts {
        let pv = Vector::new(p);
        let q = rot * pv;
        for i in 0..3 {
            for j in 0..3 {
                h[(i, j)] += q[i] * pv[j];
            }
        }
    }
    let f = h.svd().unwrap();
    let (u, v) = (f.u(), f.v());
    let mut rhat = u * v.transpose();
    // Reflection fix: guarantee a proper rotation.
    if rhat.determinant() < 0.0 {
        let mut uf = u;
        for i in 0..3 {
            uf[(i, 2)] = -uf[(i, 2)];
        }
        rhat = uf * v.transpose();
    }
    // Recovered rotation matches, is orthonormal, and has determinant +1.
    for i in 0..3 {
        for j in 0..3 {
            assert!((rhat[(i, j)] - rot[(i, j)]).abs() < 1e-9);
        }
    }
    approx_identity(rhat.transpose() * rhat);
    assert!((rhat.determinant() - 1.0).abs() < 1e-9);
}

#[test]
fn svd_redundant_jacobian_pseudo_inverse() {
    // A 6x7 manipulator Jacobian (7 joints -> 6-D task twist), full row rank.
    let j = Matrix::<6, 7>::from_fn(|r, c| {
        if c < 6 {
            if r == c {
                2.0
            } else {
                0.3 / (1.0 + (r + c) as f64)
            }
        } else {
            0.5 * (r as f64 + 1.0)
        }
    });
    let jp = j.pseudo_inverse().unwrap();

    // Moore–Penrose: J·J⁺·J == J and J·J⁺ symmetric.
    let jjpj = j * jp * j;
    for r in 0..6 {
        for c in 0..7 {
            assert!((jjpj[(r, c)] - j[(r, c)]).abs() < 1e-9);
        }
    }
    let jjp = j * jp;
    for r in 0..6 {
        for c in 0..6 {
            assert!((jjp[(r, c)] - jjp[(c, r)]).abs() < 1e-12);
        }
    }

    // Minimum-norm resolution: J⁺·v beats any other solution of J·x = v.
    let vtwist = Vector::<6>::from_fn(|i| i as f64 - 2.5);
    let x_min = jp * vtwist;
    let n = Vector::<7>::from_fn(|i| (i as f64 - 3.0) * 0.7);
    let jpjn = (jp * j) * n;
    let x_other = Vector::<7>::from_fn(|i| x_min[i] + n[i] - jpjn[i]);
    assert!((j * x_min - vtwist).norm() < 1e-9);
    assert!((j * x_other - vtwist).norm() < 1e-9);
    assert!(x_min.norm() <= x_other.norm() + 1e-9);
}

#[test]
fn svd_near_singular_jacobian() {
    // A 6x6 Jacobian at a near-singularity: column 5 nearly equals column 4.
    let mut j = Matrix::<6, 6>::from_fn(|i, jj| {
        if i == jj {
            1.0 + 0.1 * jj as f64
        } else {
            0.2 / (1.0 + (i + jj) as f64)
        }
    });
    for r in 0..6 {
        j[(r, 5)] = j[(r, 4)] + 1e-8 * (r as f64 + 1.0);
    }
    let f = j.svd().unwrap();
    let s = f.singular_values();
    let tol = 1e-4 * s[0];

    assert!(f.condition_number() > 1e6);
    assert_eq!(f.rank(tol), 5);

    // Truncated pseudo-inverse and solve stay finite (no blow-up on the tiny σ).
    let jp = f.pseudo_inverse_tol(tol);
    for r in 0..6 {
        for c in 0..6 {
            assert!(jp[(r, c)].is_finite());
        }
    }
    let x = f.solve(Vector::from_fn(|i| i as f64 + 1.0));
    for i in 0..6 {
        assert!(x[i].is_finite());
    }
}

#[test]
fn svd_overdetermined_least_squares() {
    // Fit a plane z = a·x + b·y + c to 30 sampled points; solve vs the normal equations.
    let sample = |i: usize| ((i as f64 * 0.37).sin(), (i as f64 * 0.53).cos());
    let design = Matrix::<30, 3>::from_fn(|i, c| {
        let (x, y) = sample(i);
        match c {
            0 => x,
            1 => y,
            _ => 1.0,
        }
    });
    let rhs = Vector::<30>::from_fn(|i| {
        let (x, y) = sample(i);
        0.5 * x - 1.2 * y + 2.0 + 1e-3 * (i as f64 * 1.7).sin()
    });
    let x_svd = design.svd().unwrap().solve(rhs);
    let x_ne = (design.transpose() * design)
        .solve(design.transpose() * rhs)
        .unwrap();
    for i in 0..3 {
        assert!((x_svd[i] - x_ne[i]).abs() < 1e-9);
    }
}
