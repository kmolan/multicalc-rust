use crate::linear_algebra::qr::{enorm, max, min};
use crate::linear_algebra::{Matrix, Vector};

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
