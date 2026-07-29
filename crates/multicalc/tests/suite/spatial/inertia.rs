//! Rigid-body inertia tests: constructor checks, moving the reference point, combining bodies,
//! and scalar-generic behaviour at f32 and under autodiff.

use multicalc::error::SpatialError;
use multicalc::linear_algebra::{Matrix, Matrix3D, Vector, Vector3D};
use multicalc::scalar::Dual;
use multicalc::spatial::SpatialInertia;

fn unit_inertia() -> Matrix3D {
    Matrix::from_diagonal([1.0, 1.0, 1.0])
}

fn origin() -> Vector3D {
    Vector::new([0.0, 0.0, 0.0])
}

#[test]
fn rejects_non_finite_values() {
    let mass = SpatialInertia::new(f64::INFINITY, origin(), unit_inertia());
    assert_eq!(mass, Err(SpatialError::NonFinite));

    let center = SpatialInertia::new(1.0, Vector::new([f64::NAN, 0.0, 0.0]), unit_inertia());
    assert_eq!(center, Err(SpatialError::NonFinite));

    let inertia = SpatialInertia::new(1.0, origin(), Matrix::from_diagonal([1.0, f64::NAN, 1.0]));
    assert_eq!(inertia, Err(SpatialError::NonFinite));
}

#[test]
fn rejects_non_positive_mass() {
    assert_eq!(
        SpatialInertia::new(0.0, origin(), unit_inertia()),
        Err(SpatialError::NonPositiveMass)
    );
    assert_eq!(
        SpatialInertia::new(-1.0, origin(), unit_inertia()),
        Err(SpatialError::NonPositiveMass)
    );
}

#[test]
fn rejects_inertia_that_differs_across_the_diagonal() {
    let lopsided = Matrix::new([[1.0, 0.5, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]);
    assert_eq!(
        SpatialInertia::new(1.0, origin(), lopsided),
        Err(SpatialError::NotSymmetric)
    );
}

#[test]
fn accepts_inertia_off_by_one_rounding_step() {
    // The symmetry check is scaled by the entries, not an exact comparison, so a pair that differs
    // by a single rounding step is still the same number as far as this type is concerned.
    let nudged = Matrix::new([
        [1.0, 0.25, 0.0],
        [0.25 + f64::EPSILON, 2.0, 0.0],
        [0.0, 0.0, 3.0],
    ]);
    assert!(SpatialInertia::new(1.0, origin(), nudged).is_ok());
}

#[test]
fn rejects_non_positive_inertia_diagonal() {
    assert_eq!(
        SpatialInertia::new(1.0, origin(), Matrix::from_diagonal([1.0, 0.0, 1.0])),
        Err(SpatialError::NonPositiveInertia)
    );
    assert_eq!(
        SpatialInertia::new(1.0, origin(), Matrix::from_diagonal([1.0, 1.0, -2.0])),
        Err(SpatialError::NonPositiveInertia)
    );
}

#[test]
fn diagonal_constructor_matches_the_full_one() {
    let from_diagonal =
        SpatialInertia::from_diagonal_inertia(2.0, origin(), Vector::new([1.0, 2.0, 3.0])).unwrap();
    let from_full =
        SpatialInertia::new(2.0, origin(), Matrix::from_diagonal([1.0, 2.0, 3.0])).unwrap();
    assert_eq!(from_diagonal, from_full);
}

#[test]
fn inertia_about_the_balance_point_is_unchanged() {
    let center = Vector::new([0.3, -0.2, 0.5]);
    let body = SpatialInertia::new(2.0, center, unit_inertia()).unwrap();
    assert_eq!(body.inertia_about(center), unit_inertia());
}

#[test]
fn inertia_about_grows_with_distance() {
    let body = SpatialInertia::new(1.0, origin(), unit_inertia()).unwrap();
    // A metre along x leaves spinning about x alone and doubles the other two.
    let shifted = body.inertia_about(Vector::new([1.0, 0.0, 0.0]));
    assert_eq!(shifted.diagonal(), [1.0, 2.0, 2.0]);
    assert_eq!(shifted[(0, 1)], 0.0);
    assert_eq!(shifted[(0, 2)], 0.0);
    assert_eq!(shifted[(1, 2)], 0.0);
}

#[test]
fn inertia_about_couples_axes_for_a_diagonal_offset() {
    let body = SpatialInertia::new(2.0, origin(), unit_inertia()).unwrap();
    let offset = Vector::new([1.0, 2.0, 0.0]);
    let shifted = body.inertia_about(offset);
    // Each entry picks up the mass times the offset spread: the diagonal by how far the point is
    // overall, the off-diagonal by the product of the two directions.
    assert!((shifted[(0, 0)] - (1.0 + 2.0 * 4.0)).abs() < 1e-12);
    assert!((shifted[(1, 1)] - (1.0 + 2.0 * 1.0)).abs() < 1e-12);
    assert!((shifted[(2, 2)] - (1.0 + 2.0 * 5.0)).abs() < 1e-12);
    assert!((shifted[(0, 1)] - (-2.0 * 2.0)).abs() < 1e-12);
    assert_eq!(shifted[(0, 1)], shifted[(1, 0)]);
}

#[test]
fn reports_whether_every_number_is_finite() {
    let body = SpatialInertia::new(1.0, origin(), unit_inertia()).unwrap();
    assert!(body.is_finite());
}

#[test]
fn works_at_single_precision() {
    let body = SpatialInertia::new(
        2.0_f32,
        Vector::new([0.0_f32, 0.0, 0.0]),
        Matrix::from_diagonal([1.0_f32, 1.0, 1.0]),
    )
    .unwrap();

    // Mass 2 at an offset of (1, 2, 0), which is 5 away squared: each diagonal entry picks up the
    // mass times what is left after taking out its own direction, and the xy pair picks up the
    // product of the two.
    let shifted = body.inertia_about(Vector::new([1.0_f32, 2.0, 0.0]));
    assert!((shifted[(0, 0)] - 9.0).abs() < 1e-5);
    assert!((shifted[(1, 1)] - 3.0).abs() < 1e-5);
    assert!((shifted[(2, 2)] - 11.0).abs() < 1e-5);
    assert!((shifted[(0, 1)] + 4.0).abs() < 1e-5);
    assert_eq!(shifted[(0, 1)], shifted[(1, 0)]);
}

#[test]
fn differentiates_through_the_mass() {
    // Seed the mass with a unit derivative and read it back out of the shifted inertia. Moving the
    // reference point a metre along x adds the mass itself to the y entry, so the derivative there
    // is exactly one.
    let mass = Dual::variable(2.0_f64);
    let center = Vector::new([Dual::constant(0.0); 3]);
    let inertia = Matrix::from_diagonal([Dual::constant(1.0); 3]);
    let body = SpatialInertia::new(mass, center, inertia).unwrap();

    let offset = Vector::new([
        Dual::constant(1.0),
        Dual::constant(0.0),
        Dual::constant(0.0),
    ]);
    let shifted = body.inertia_about(offset);

    assert!((shifted[(1, 1)].value - 3.0).abs() < 1e-12);
    assert!((shifted[(1, 1)].deriv - 1.0).abs() < 1e-12);
    // Spinning about the direction the point moved in does not change, so nothing depends on the
    // mass there.
    assert!((shifted[(0, 0)].deriv - 0.0).abs() < 1e-12);
}
