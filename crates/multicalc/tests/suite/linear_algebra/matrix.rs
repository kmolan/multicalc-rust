use multicalc::error::LinalgError;
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc_testkit::tol::{Tol, assert_identity, assert_matrix_close, assert_scalar_close};
use proptest::prelude::*;

// A strategy for producing matrices in property-based tests.
fn matrix_strategy<const ROWS: usize, const COLS: usize, S>(
    num_strategy: S,
) -> impl Strategy<Value = Matrix<ROWS, COLS>>
where
    S: Strategy<Value = f64>,
{
    prop::array::uniform::<_, ROWS>(prop::array::uniform::<_, COLS>(num_strategy))
        .prop_map(Matrix::new)
}

// ----- matrix arithmetic, multiply, transpose -----

#[test]
fn matrix_arithmetic() {
    let left = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    let right = Matrix::new([[5.0, 6.0], [7.0, 8.0]]);

    assert_eq!(left + right, Matrix::new([[6.0, 8.0], [10.0, 12.0]]));
    assert_eq!(right - left, Matrix::new([[4.0, 4.0], [4.0, 4.0]]));
    assert_eq!(-left, Matrix::new([[-1.0, -2.0], [-3.0, -4.0]]));
    assert_eq!(left * 2.0, left.scale(2.0));
    assert_eq!(left / 2.0, left.scale(0.5));

    let mut accumulated = left;
    accumulated += right;
    assert_eq!(accumulated, left + right);
    accumulated -= right;
    assert_eq!(accumulated, left);

    // Check division by zero behavior
    let a = Matrix::new([
        [1.0, -1.0, 0.0],
        [f64::INFINITY, f64::NEG_INFINITY, f64::NAN],
    ]);
    let div_zero = a / 0.0;
    let expected = Matrix::new([
        [f64::INFINITY, f64::NEG_INFINITY, f64::NAN],
        [f64::INFINITY, f64::NEG_INFINITY, f64::NAN],
    ]);
    div_zero
        .into_array()
        .into_iter()
        .flatten()
        .zip(expected.into_array().into_iter().flatten())
        .for_each(|(got, want)| {
            assert_eq!(
                got.total_cmp(&want),
                core::cmp::Ordering::Equal,
                "{got} != {want}"
            );
        })
}

#[test]
fn try_row_column() {
    let matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);

    assert_eq!(matrix.try_row(1), Some(Vector::new([3.0, 4.0])));
    assert_eq!(matrix.try_row(2), None);

    assert_eq!(matrix.try_column(0), Some(Vector::new([1.0, 3.0])));
    assert_eq!(matrix.try_column(2), None);

    let empty: Matrix<0, 3> = Matrix::zeros();
    assert_eq!(empty.try_column(0), Some(Vector::<0, f64>::zeros()));
    assert_eq!(empty.try_column(3), None);
}

#[test]
fn matrix_multiply() {
    let left = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    let right = Matrix::new([[5.0, 6.0], [7.0, 8.0]]);
    let identity: Matrix<2, 2> = Matrix::identity();

    assert_eq!(left * identity, left);
    assert_eq!(identity * left, left);
    assert_eq!((left * right) * left, left * (right * left)); // associativity

    // non-square 2x3 * 3x2
    let wide = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let tall = Matrix::new([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);
    assert_eq!(wide * tall, Matrix::new([[58.0, 64.0], [139.0, 154.0]]));

    // matrix x vector
    assert_eq!(left * Vector::new([1.0, 1.0]), Vector::new([3.0, 7.0]));
}

#[test]
fn matrix_transpose() {
    let matrix = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]); // 2x3
    assert_eq!(
        matrix.transpose(),
        Matrix::new([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    ); // 3x2
    assert_eq!(matrix.transpose().transpose(), matrix); // involution

    let left = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    let right = Matrix::new([[5.0, 6.0], [7.0, 8.0]]);
    assert_eq!(
        (left * right).transpose(),
        right.transpose() * left.transpose()
    );
}

fn check_matrix_is_finite<const ROWS: usize, const COLS: usize>(m: Matrix<ROWS, COLS>) {
    assert_eq!(
        m.is_finite(),
        m.into_array().iter().flatten().all(|x| x.is_finite())
    );
}

// Check the Frobenius norm implementation using the alternate definition
// `||A||_F = sqrt(Tr(A * A^T))`.
fn check_matrix_frobenius_norm<const ROWS: usize, const COLS: usize>(m: Matrix<ROWS, COLS>) {
    let norm = m.frobenius_norm();
    let alt_def = (m * m.transpose()).trace().sqrt();
    assert_eq!(
        norm.is_finite(),
        alt_def.is_finite(),
        "{} != {}",
        norm,
        alt_def
    );
    if norm.is_finite() {
        assert_scalar_close(
            norm,
            alt_def,
            Tol {
                abs: 0.0,
                rel: 1e-8,
            },
        );
    }
}

fn check_matrix_trace<const N: usize>(m: Matrix<N, N>) {
    assert_eq!(m.trace(), (0..N).fold(0.0, |acc, i| acc + m[(i, i)]));
}

fn check_matrix_from_diagonal<const N: usize>(diag: [f64; N]) {
    let m = Matrix::from_diagonal(diag);
    for i in 0..N {
        assert_eq!(m[(i, i)], diag[i]);
        for j in 0..N {
            if i != j {
                assert_eq!(m[(i, j)], 0.0);
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn matrix_is_finite_1x1(m in matrix_strategy::<1, 1, _>(prop::num::f64::ANY)) {
        check_matrix_is_finite(m);
    }

    #[test]
    fn matrix_is_finite_2x2(m in matrix_strategy::<2, 2, _>(prop::num::f64::ANY)) {
        check_matrix_is_finite(m);
    }

    #[test]
    fn matrix_is_finite_3x3(m in matrix_strategy::<3, 3, _>(prop::num::f64::ANY)) {
        check_matrix_is_finite(m);
    }

    #[test]
    fn matrix_is_finite_4x3(m in matrix_strategy::<4, 3, _>(prop::num::f64::ANY)) {
        check_matrix_is_finite(m);
    }

    #[test]
    fn matrix_is_finite_3x4(m in matrix_strategy::<3, 4, _>(prop::num::f64::ANY)) {
        check_matrix_is_finite(m);
    }

    #[test]
    fn matrix_is_finite_4x4(m in matrix_strategy::<4, 4, _>(prop::num::f64::ANY)) {
        check_matrix_is_finite(m);
    }

    #[test]
    fn matrix_frobenius_norm_1x1(m in matrix_strategy::<1, 1, _>(prop::num::f64::ANY)) {
        check_matrix_frobenius_norm(m);
    }

    #[test]
    fn matrix_frobenius_norm_2x2(m in matrix_strategy::<2, 2, _>(prop::num::f64::ANY)) {
        check_matrix_frobenius_norm(m);
    }

    #[test]
    fn matrix_frobenius_norm_3x3(m in matrix_strategy::<3, 3, _>(prop::num::f64::ANY)) {
        check_matrix_frobenius_norm(m);
    }

    #[test]
    fn matrix_frobenius_norm_4x3(m in matrix_strategy::<4, 3, _>(prop::num::f64::ANY)) {
        check_matrix_frobenius_norm(m);
    }

    #[test]
    fn matrix_frobenius_norm_3x4(m in matrix_strategy::<3, 4, _>(prop::num::f64::ANY)) {
        check_matrix_frobenius_norm(m);
    }

    #[test]
    fn matrix_frobenius_norm_4x4(m in matrix_strategy::<4, 4, _>(prop::num::f64::ANY)) {
        check_matrix_frobenius_norm(m);
    }

    // Note: using `prop::num::f64::NORMAL` as opposed to `prop::num::f64::ANY` because
    // if `NaN` or `Infinity` are involved then the equality check will fail (`NaN != NaN`).
    #[test]
    fn matrix_trace_1x1(m in matrix_strategy::<1, 1, _>(prop::num::f64::NORMAL)) {
        check_matrix_trace(m);
    }

    #[test]
    fn matrix_trace_2x2(m in matrix_strategy::<2, 2, _>(prop::num::f64::NORMAL)) {
        check_matrix_trace(m);
    }

    #[test]
    fn matrix_trace_3x3(m in matrix_strategy::<3, 3, _>(prop::num::f64::NORMAL)) {
        check_matrix_trace(m);
    }

    #[test]
    fn matrix_trace_4x4(m in matrix_strategy::<4, 4, _>(prop::num::f64::NORMAL)) {
        check_matrix_trace(m);
    }

    #[test]
    fn matrix_from_diagonal_1x1(diag in prop::array::uniform1(prop::num::f64::NORMAL)) {
        check_matrix_from_diagonal(diag);
    }

    #[test]
    fn matrix_from_diagonal_2x2(diag in prop::array::uniform2(prop::num::f64::NORMAL)) {
        check_matrix_from_diagonal(diag);
    }

    #[test]
    fn matrix_from_diagonal_3x3(diag in prop::array::uniform3(prop::num::f64::NORMAL)) {
        check_matrix_from_diagonal(diag);
    }

    #[test]
    fn matrix_from_diagonal_4x4(diag in prop::array::uniform4(prop::num::f64::NORMAL)) {
        check_matrix_from_diagonal(diag);
    }
}

// ----- determinant & inverse (specialized) -----

#[test]
fn matrix_determinant() {
    let identity_2x2: Matrix<2, 2> = Matrix::identity();
    let identity_3x3: Matrix<3, 3> = Matrix::identity();
    assert_eq!(identity_2x2.determinant(), 1.0);
    assert_eq!(identity_3x3.determinant(), 1.0);

    assert_eq!(Matrix::new([[1.0, 2.0], [3.0, 4.0]]).determinant(), -2.0);

    let three_by_three = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]);
    let expected = -3.0;
    assert_eq!(three_by_three.determinant(), expected);

    assert_eq!(Matrix::new([[1.0, 2.0], [2.0, 4.0]]).determinant(), 0.0); // singular
}

#[test]
fn matrix_inverse() {
    let identity_2x2: Matrix<2, 2> = Matrix::identity();
    let identity_3x3: Matrix<3, 3> = Matrix::identity();
    assert_eq!(identity_2x2.inverse(), Ok(identity_2x2));
    assert_eq!(identity_3x3.inverse(), Ok(identity_3x3));

    let invertible_2x2 = Matrix::new([[4.0, 7.0], [2.0, 6.0]]);
    assert_identity(invertible_2x2 * invertible_2x2.inverse().unwrap(), 1e-12);

    let invertible_3x3 = Matrix::new([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]]);
    assert_identity(invertible_3x3 * invertible_3x3.inverse().unwrap(), 1e-12);

    // singular -> Err(SingularMatrix)
    let singular2 = Matrix::new([[1.0, 2.0], [2.0, 4.0]]);
    assert_eq!(singular2.inverse(), Err(LinalgError::Singular));
    let singular3 = Matrix::new([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]]);
    assert_eq!(singular3.determinant(), 0.0);
    assert_eq!(singular3.inverse(), Err(LinalgError::Singular));

    // Near-singular: det is tiny but nonzero under exact compare.
    let near_singular_2x2 = Matrix::new([[1.0, 1.0], [1.0, 1.0 + f64::EPSILON]]);
    assert_ne!(near_singular_2x2.determinant(), 0.0);
    assert_eq!(near_singular_2x2.inverse(), Err(LinalgError::Singular));

    let near_singular_3x3 =
        Matrix::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, f64::EPSILON]]);
    assert_ne!(near_singular_3x3.determinant(), 0.0);
    assert_eq!(near_singular_3x3.inverse(), Err(LinalgError::Singular));

    // Tiny but full-rank still inverts (scaled threshold, not absolute eps).
    let tiny = Matrix::<2, 2>::identity().scale(1e-8);
    assert!(tiny.inverse().is_ok());
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
    let symmetric = Matrix::<4, 4>::new([
        [4.0, 3.0, 2.0, 1.0],
        [3.0, 4.0, 3.0, 2.0],
        [2.0, 3.0, 4.0, 3.0],
        [1.0, 2.0, 3.0, 4.0],
    ]);
    assert_eq!(symmetric.determinant(), 20.0);

    let inverse = symmetric.inverse().unwrap();
    let expected = Matrix::new([
        [0.6, -0.5, 0.0, 0.1],
        [-0.5, 1.0, -0.5, 0.0],
        [0.0, -0.5, 1.0, -0.5],
        [0.1, 0.0, -0.5, 0.6],
    ]);
    assert_matrix_close(inverse, expected, 1e-12);
    assert_identity(symmetric * inverse, 1e-12);

    // A non-symmetric matrix, so its (non-symmetric) inverse catches any transpose error in
    // the adjugate. Reference from an exact rational solve.
    let non_symmetric = Matrix::<4, 4>::new([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 1.0, 0.0, 1.0],
        [0.0, 3.0, 1.0, 2.0],
        [1.0, 0.0, 2.0, 1.0],
    ]);
    assert_eq!(non_symmetric.determinant(), -20.0);

    let non_symmetric_inverse = non_symmetric.inverse().unwrap();
    let expected = Matrix::new([
        [-0.15, 0.45, -0.05, 0.25],
        [-0.35, 0.05, 0.55, 0.25],
        [-0.25, -0.25, 0.25, 0.75],
        [0.65, 0.05, -0.45, -0.75],
    ]);
    assert_matrix_close(non_symmetric_inverse, expected, 1e-12);
    assert_identity(non_symmetric * non_symmetric_inverse, 1e-12);
    assert_identity(non_symmetric_inverse * non_symmetric, 1e-12);

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
    let single_precision = Matrix::<4, 4, f32>::new([
        [4.0, 3.0, 2.0, 1.0],
        [3.0, 4.0, 3.0, 2.0],
        [2.0, 3.0, 4.0, 3.0],
        [1.0, 2.0, 3.0, 4.0],
    ]);
    assert_identity(
        single_precision * single_precision.inverse().unwrap(),
        1e-5_f32,
    );

    // Near-singular: det is tiny but nonzero under exact compare.
    let near_singular_4x4 = Matrix::<4, 4>::new([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, f64::EPSILON],
    ]);
    assert_ne!(near_singular_4x4.determinant(), 0.0);
    assert_eq!(near_singular_4x4.inverse(), Err(LinalgError::Singular));
}

#[test]
fn matrix_5x5_determinant_and_inverse() {
    let subject = Matrix::<5, 5>::new([
        [1.0, 2.0, 3.0, 4.0, 0.0],
        [2.0, 1.0, 0.0, 1.0, 2.0],
        [0.0, 3.0, 1.0, 2.0, 1.0],
        [1.0, 0.0, 2.0, 1.0, 2.0],
        [3.0, 1.0, 0.0, 3.0, 1.0],
    ]);

    assert!((subject.determinant() - -27.0).abs() < 1e-12);

    // Each row is the running sum of the row above it.
    let cumulative_sums = Matrix::<5, 5>::new([
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1.0, 3.0, 6.0, 10.0, 15.0],
        [1.0, 4.0, 10.0, 20.0, 35.0],
        [1.0, 5.0, 15.0, 35.0, 70.0],
    ]);

    let expected = Matrix::<5, 5>::new([
        [5.0, -10.0, 10.0, -5.0, 1.0],
        [-10.0, 30.0, -35.0, 19.0, -4.0],
        [10.0, -35.0, 46.0, -27.0, 6.0],
        [-5.0, 19.0, -27.0, 17.0, -4.0],
        [1.0, -4.0, 6.0, -4.0, 1.0],
    ]);
    assert_matrix_close(cumulative_sums.inverse().unwrap(), expected, 1e-12);
}

#[test]
fn matrix_solve_agrees_with_lu() {
    let matrix = Matrix::<3, 3>::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]);
    let right_hand_side = Vector::new([7.0, 19.0, 49.0]);
    let solution = matrix.solve(right_hand_side).unwrap();

    // The convenience solver matches an explicit LU solve.
    let lu_solution = matrix.lu().unwrap().solve(right_hand_side);
    for index in 0..3 {
        assert!((solution[index] - lu_solution[index]).abs() < 1e-12);
    }
    assert!((matrix * solution - right_hand_side).norm() < 1e-12);

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
    let first = Vector::<3, f32>::new([1.0, 2.0, 2.0]);
    let second = Vector::<3, f32>::new([2.0, 0.0, 1.0]);
    assert!((first.norm() - 3.0).abs() < 1e-6);
    assert!((first.dot(second) - 4.0).abs() < 1e-6);

    let matrix = Matrix::<2, 2, f32>::new([[1.0, 2.0], [3.0, 4.0]]);
    let identity: Matrix<2, 2, f32> = Matrix::identity();
    assert_eq!(matrix * identity, matrix);
}
