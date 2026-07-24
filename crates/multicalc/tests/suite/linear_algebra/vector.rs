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
    let a = Vector::new([1.0, 2.0, 3.0]);
    let b = Vector::new([4.0, 5.0, 6.0]);

    assert_eq!(a + b, Vector::new([5.0, 7.0, 9.0]));
    assert_eq!(b - a, Vector::new([3.0, 3.0, 3.0]));
    assert_eq!(-a, Vector::new([-1.0, -2.0, -3.0]));
    assert_eq!(a * 2.0, a.scale(2.0));
    assert_eq!(a / 2.0, a.scale(0.5));
    assert_eq!(a.scale(2.0), Vector::new([2.0, 4.0, 6.0]));

    let mut c = a;
    c += b;
    assert_eq!(c, a + b);
    c -= b;
    assert_eq!(c, a);
}

#[test]
fn vector_dot_and_norm() {
    let a: Vector<3> = Vector::new([1.0, 2.0, 3.0]);
    let b: Vector<3> = Vector::new([4.0, 5.0, 6.0]);
    assert_eq!(a.dot(b), 32.0);
    assert!((a.dot(b) - b.dot(a)).abs() < 1e-12); // symmetry
    assert_eq!(Vector::new([1.0, 0.0]).dot(Vector::new([0.0, 1.0])), 0.0); // orthogonal

    let empty: Vector<0> = Vector::zeros();
    assert_eq!(empty.dot(empty), 0.0);

    let v: Vector<2> = Vector::new([3.0, 4.0]);
    assert_eq!(v.norm(), 5.0);
    assert!((v.norm_squared() - v.norm() * v.norm()).abs() < 1e-12);

    let z: Vector<3> = Vector::zeros();
    assert_eq!(z.norm(), 0.0);
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

    v.normalize();
    assert_scalar_close(v.norm(), 1.0, tol);

    Ok(())
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
