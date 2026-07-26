use core::f64::consts::{E, PI};
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc_testkit::tol::{Tol, assert_scalar_close};
use proptest::prelude::*;
use proptest::test_runner::TestCaseError;

// A strategy for producing vectors in property-based tests.
fn vector_strategy<const N: usize, S>(num_strategy: S) -> impl Strategy<Value = Vector<N>>
where
    S: Strategy<Value = f64>,
{
    prop::array::uniform::<_, N>(num_strategy).prop_map(Vector::new)
}

// ----- construction & access -----

#[test]
fn construct_and_access() {
    let vector = Vector::new([1.0, 2.0, 3.0]);
    assert_eq!(vector.get(0), Some(&1.0));
    assert_eq!(vector.into_array(), [1.0, 2.0, 3.0]);

    let mut mutable = Vector::from([4.0, 5.0]);
    if let Some(entry) = mutable.get_mut(1) {
        *entry = 9.0;
    }
    assert_eq!(mutable, Vector::new([4.0, 9.0]));

    let zeros: Vector<3> = Vector::zeros();
    assert_eq!(zeros, Vector::new([0.0, 0.0, 0.0]));

    assert_eq!(
        Vector::<4>::from_fn(|index| index as f64),
        Vector::new([0.0, 1.0, 2.0, 3.0])
    );

    let mut matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    assert_eq!(matrix.get(1, 0), Some(&3.0));
    if let Some(entry) = matrix.get_mut(0, 1) {
        *entry = 7.0;
    }
    assert_eq!(matrix.get(0, 1), Some(&7.0));

    let identity: Matrix<3, 3> = Matrix::identity();
    assert_eq!(
        identity.into_array(),
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    );

    assert_eq!(
        Matrix::<2, 2>::from_fn(|row, column| (row * 2 + column) as f64),
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

#[test]
fn get_checked_access() {
    let vector = Vector::new([1.0, 2.0, 3.0]);
    assert_eq!(vector.get(0), Some(&1.0));
    assert_eq!(vector.get(3), None);

    let mut mutable = Vector::new([4.0, 5.0]);
    if let Some(entry) = mutable.get_mut(1) {
        *entry = 9.0;
    }
    assert_eq!(mutable.get(1), Some(&9.0));

    let mut matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    assert_eq!(matrix.get(1, 0), Some(&3.0));
    assert_eq!(matrix.get(2, 0), None);
    assert_eq!(matrix.get(0, 2), None);
    if let Some(entry) = matrix.get_mut(0, 1) {
        *entry = 7.0;
    }
    assert_eq!(matrix.get(0, 1), Some(&7.0));

    matrix.as_mut_slice_rows()[1][1] = 8.0;
    assert_eq!(matrix.get(1, 1), Some(&8.0));
}

fn check_vector_map<const N: usize, F: Fn(f64) -> f64>(v: Vector<N>, f: F) {
    let u = v.map(&f);
    for (a, b) in u.as_slice().iter().zip(v.as_slice()) {
        assert_eq!(*a, f(*b));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn vector_map_1(v in vector_strategy::<1, _>(prop::num::f64::NORMAL)) {
        check_vector_map(v, |x| x + PI);
    }

    #[test]
    fn vector_map_2(v in vector_strategy::<2, _>(prop::num::f64::NORMAL)) {
        check_vector_map(v, |x| 2.0 * x);
    }

    #[test]
    fn vector_map_3(v in vector_strategy::<3, _>(prop::num::f64::NORMAL)) {
        check_vector_map(v, |x| x - E);
    }

    #[test]
    fn vector_map_4(v in vector_strategy::<4, _>(prop::num::f64::NORMAL)) {
        check_vector_map(v, |x| x * x);
    }
}

// ----- vector arithmetic -----

#[test]
fn vector_arithmetic() {
    let left = Vector::new([1.0, 2.0, 3.0]);
    let right = Vector::new([4.0, 5.0, 6.0]);

    assert_eq!(left + right, Vector::new([5.0, 7.0, 9.0]));
    assert_eq!(right - left, Vector::new([3.0, 3.0, 3.0]));
    assert_eq!(-left, Vector::new([-1.0, -2.0, -3.0]));
    assert_eq!(left * 2.0, left.scale(2.0));
    assert_eq!(left.scale(2.0), Vector::new([2.0, 4.0, 6.0]));
    assert_eq!(left / 2.0, left.scale(0.5));

    let mut accumulated = left;
    accumulated += right;
    assert_eq!(accumulated, left + right);
    accumulated -= right;
    assert_eq!(accumulated, left);

    // Check division by zero behavior
    let a = Vector::new([
        1.0,
        -1.0,
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ]);
    let div_zero = a / 0.0;
    let expected = Vector::new([
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ]);
    div_zero
        .into_array()
        .into_iter()
        .zip(expected.as_array())
        .for_each(|(got, want)| {
            assert_eq!(
                got.total_cmp(want),
                core::cmp::Ordering::Equal,
                "{got} != {want}"
            );
        })
}

#[test]
fn vector_dot_and_norm() {
    let left: Vector<3> = Vector::new([1.0, 2.0, 3.0]);
    let right: Vector<3> = Vector::new([4.0, 5.0, 6.0]);
    assert_eq!(left.dot(right), 32.0);
    assert!((left.dot(right) - right.dot(left)).abs() < 1e-12); // symmetry
    assert_eq!(Vector::new([1.0, 0.0]).dot(Vector::new([0.0, 1.0])), 0.0); // orthogonal

    let empty: Vector<0> = Vector::zeros();
    assert_eq!(empty.dot(empty), 0.0);

    let vector: Vector<2> = Vector::new([3.0, 4.0]);
    assert_eq!(vector.norm(), 5.0);
    assert!((vector.norm_squared() - vector.norm() * vector.norm()).abs() < 1e-12);

    let zeros: Vector<3> = Vector::zeros();
    assert_eq!(zeros.norm(), 0.0);
    assert!(Vector::new([f64::INFINITY, 0.0]).norm().is_infinite());
}

#[test]
fn vector_is_finite() {
    assert!(Vector::new([1.0, -2.0, 3.0]).is_finite());
    assert!(Vector::<0>::zeros().is_finite()); // vacuously true
    assert!(!Vector::new([1.0, f64::NAN]).is_finite());
    assert!(!Vector::new([f64::INFINITY, 0.0]).is_finite());
    assert!(!Vector::new([0.0, f64::NEG_INFINITY]).is_finite());
}

fn check_vector_normalized<const N: usize>(mut v: Vector<N>) -> Result<(), TestCaseError> {
    prop_assume!(v.norm().is_finite());
    prop_assume!(v.norm() > 1e-16);

    let tol = Tol {
        abs: 0.0,
        rel: 1e-8,
    };

    let normalized = v.normalized();
    assert_scalar_close(normalized.norm(), 1.0, tol);

    // The normalized and try_normalized operations are identical when normalization is possible.
    assert_eq!(Some(normalized), v.try_normalized());

    // Create a copy of the vector to test both mutable operations.
    let mut v_copy = v;

    // In-place normalization gives the same result as returning a copy.
    v.normalize();
    assert_eq!(v, normalized);

    // Fallible in-place normalization also agrees;
    assert!(v_copy.try_normalize().is_ok());
    assert_eq!(v_copy, normalized);

    Ok(())
}

#[test]
fn zero_vector_normalize() {
    // A vector with any finite non-zero entry (even EPSILON) can still be normalized
    let mut near_zero = Vector::new([<f64 as multicalc::Numeric>::EPSILON, 0.0, 0.0]);
    assert_eq!(near_zero.normalized(), Vector::new([1.0f64, 0.0, 0.0]));
    near_zero.normalize();
    assert_eq!(near_zero, Vector::new([1.0f64, 0.0, 0.0]));

    let mut zero = Vector::new([0.0f64, 0.0, 0.0]);

    // Fallible normalization rejects the zero vector
    assert_eq!(zero.try_normalized(), None);
    assert!(zero.try_normalize().is_err());
    assert_eq!(zero, Vector::new([0.0f64, 0.0, 0.0]));

    // The zero vector will normalize to NAN via the unchecked functions
    assert!(
        zero.normalized()
            .into_array()
            .into_iter()
            .all(|x| x.total_cmp(&f64::NAN).is_eq())
    );
    zero.normalize();
    assert!(
        zero.into_array()
            .into_iter()
            .all(|x| x.total_cmp(&f64::NAN).is_eq())
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn vector_normalize_1(v in vector_strategy::<1, _>(prop::num::f64::NORMAL)) {
        check_vector_normalized(v)?;
    }

    #[test]
    fn vector_normalize_2(v in vector_strategy::<2, _>(prop::num::f64::NORMAL)) {
        check_vector_normalized(v)?;
    }

    #[test]
    fn vector_normalize_3(v in vector_strategy::<3, _>(prop::num::f64::NORMAL)) {
        check_vector_normalized(v)?;
    }

    #[test]
    fn vector_normalize_4(v in vector_strategy::<4, _>(prop::num::f64::NORMAL)) {
        check_vector_normalized(v)?;
    }
}

// ----- cross products & scalar triple -----

#[test]
fn vector_cross_3d() {
    let unit_x = Vector::new([1.0, 0.0, 0.0]);
    let unit_y = Vector::new([0.0, 1.0, 0.0]);
    let unit_z = Vector::new([0.0, 0.0, 1.0]);
    assert_eq!(unit_x.cross(unit_y), unit_z);
    assert_eq!(unit_y.cross(unit_z), unit_x);
    assert_eq!(unit_z.cross(unit_x), unit_y);
    assert_eq!(unit_x.cross(unit_y), -(unit_y.cross(unit_x))); // anti-commutativity

    let left = Vector::new([1.0, 2.0, 3.0]);
    let right = Vector::new([4.0, 5.0, 6.0]);
    let cross_product = left.cross(right);
    assert_eq!(left.dot(cross_product), 0.0); // orthogonal to both inputs
    assert_eq!(right.dot(cross_product), 0.0);
}

#[test]
fn vector_cross_2d_and_scalar_triple() {
    assert_eq!(Vector::new([1.0, 0.0]).cross(Vector::new([0.0, 1.0])), 1.0);
    let left: Vector<2> = Vector::new([2.0, 3.0]);
    let right: Vector<2> = Vector::new([5.0, 7.0]);
    assert!((left.cross(right) + right.cross(left)).abs() < 1e-12); // anti-commutativity
    assert_eq!(left.cross(left), 0.0); // parallel

    let unit_x = Vector::new([1.0, 0.0, 0.0]);
    let unit_y = Vector::new([0.0, 1.0, 0.0]);
    let unit_z = Vector::new([0.0, 0.0, 1.0]);
    assert_eq!(unit_x.scalar_triple(unit_y, unit_z), 1.0);

    let first: Vector<3> = Vector::new([1.0, 2.0, 3.0]);
    let second: Vector<3> = Vector::new([0.0, 1.0, 4.0]);
    let third: Vector<3> = Vector::new([5.0, 6.0, 0.0]);
    // cyclic
    assert!(
        (first.scalar_triple(second, third) - second.scalar_triple(third, first)).abs() < 1e-12
    );
    let rows_as_matrix = Matrix::new([first.into_array(), second.into_array(), third.into_array()]);
    // equals the determinant of the matrix whose rows are the three vectors
    assert!((first.scalar_triple(second, third) - rows_as_matrix.determinant()).abs() < 1e-12);
}
