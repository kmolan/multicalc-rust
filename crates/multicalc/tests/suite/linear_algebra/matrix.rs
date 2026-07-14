use multicalc::error::LinalgError;
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc_testkit::tol::{assert_identity, assert_matrix_close};

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
    assert_identity(a * a.inverse().unwrap(), 1e-12);

    let b = Matrix::new([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]]);
    assert_identity(b * b.inverse().unwrap(), 1e-12);

    // singular -> Err(SingularMatrix)
    let singular2 = Matrix::new([[1.0, 2.0], [2.0, 4.0]]);
    assert_eq!(singular2.inverse(), Err(LinalgError::Singular));
    let singular3 = Matrix::new([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]]);
    assert_eq!(singular3.determinant(), 0.0);
    assert_eq!(singular3.inverse(), Err(LinalgError::Singular));
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

    let inv = a.inverse().unwrap();
    assert_matrix_close(
        inv,
        Matrix::new([
            [0.6, -0.5, 0.0, 0.1],
            [-0.5, 1.0, -0.5, 0.0],
            [0.0, -0.5, 1.0, -0.5],
            [0.1, 0.0, -0.5, 0.6],
        ]),
        1e-12,
    );
    assert_identity(a * inv, 1e-12);

    // A non-symmetric matrix, so its (non-symmetric) inverse catches any transpose error in
    // the adjugate. Reference from an exact rational solve.
    let b = Matrix::<4, 4>::new([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 1.0, 0.0, 1.0],
        [0.0, 3.0, 1.0, 2.0],
        [1.0, 0.0, 2.0, 1.0],
    ]);
    assert_eq!(b.determinant(), -20.0);

    let b_inv = b.inverse().unwrap();
    assert_matrix_close(
        b_inv,
        Matrix::new([
            [-0.15, 0.45, -0.05, 0.25],
            [-0.35, 0.05, 0.55, 0.25],
            [-0.25, -0.25, 0.25, 0.75],
            [0.65, 0.05, -0.45, -0.75],
        ]),
        1e-12,
    );
    assert_identity(b * b_inv, 1e-12);
    assert_identity(b_inv * b, 1e-12);

    // Rows in arithmetic progression are rank-deficient.
    let singular = Matrix::<4, 4>::new([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ]);
    assert_eq!(singular.determinant(), 0.0);
    assert_eq!(singular.inverse(), Err(LinalgError::Singular));

    // The same code at f32 round-trips to the identity.
    let af = Matrix::<4, 4, f32>::new([
        [4.0, 3.0, 2.0, 1.0],
        [3.0, 4.0, 3.0, 2.0],
        [2.0, 3.0, 4.0, 3.0],
        [1.0, 2.0, 3.0, 4.0],
    ]);
    assert_identity(af * af.inverse().unwrap(), 1e-5_f32);
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
        Some(LinalgError::Singular)
    );
}

// ----- genericity: the same code at f32 -----

#[test]
fn genericity_f32() {
    let a = Vector::<3, f32>::new([1.0, 2.0, 2.0]);
    let b = Vector::<3, f32>::new([2.0, 0.0, 1.0]);
    assert!((a.norm() - 3.0).abs() < 1e-6);
    assert!((a.dot(b) - 4.0).abs() < 1e-6);

    let m = Matrix::<2, 2, f32>::new([[1.0, 2.0], [3.0, 4.0]]);
    let id: Matrix<2, 2, f32> = Matrix::identity();
    assert_eq!(m * id, m);
}
