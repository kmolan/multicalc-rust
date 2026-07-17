//! Differential-drive tests: map round trips, degenerate motions, constructor rejection, the
//! nonholonomic tangent seam, and f32 identity coverage.

use multicalc::error::KinematicsError;
use multicalc::kinematics::{
    BodyArc, BodyTwist, DifferentialDrive, WheelRotations, WheelVelocities,
};
use multicalc::linear_algebra::Vector;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const TOL: f64 = 1e-15;

// ---- helpers ----------------------------------------------------------------

/// The round trips are bijections, so the only slack is floating-point rounding. Both maps scale by
/// the wheel/body magnitudes, and the reconstruction cancels two terms of that size to recover a
/// possibly much smaller one, so the bound tracks the largest value in play rather than the result.
fn assert_exact(got: f64, want: f64, scale: f64, what: &str) {
    let bound = 8.0 * f64::EPSILON * scale.abs().max(1.0);
    assert!(
        (got - want).abs() <= bound,
        "{what}: got {got}, want {want}, bound {bound}"
    );
}

fn rand_geometry(rng: &mut StdRng) -> DifferentialDrive<f64> {
    DifferentialDrive::new(rng.gen_range(0.01..0.5), rng.gen_range(0.05..1.0)).unwrap()
}

fn rand_wheels(rng: &mut StdRng) -> WheelVelocities<f64> {
    WheelVelocities::new(rng.gen_range(-20.0..20.0), rng.gen_range(-20.0..20.0))
}

fn rand_body_twist(rng: &mut StdRng) -> BodyTwist<f64> {
    BodyTwist::new(rng.gen_range(-5.0..5.0), rng.gen_range(-5.0..5.0))
}

/// Error scales for reconstructing a body pair from the wheel pair it maps to. Recovering the
/// linear term cancels the two wheel terms and rescales by `r`; the angular term does the same and
/// then divides by `b`, which amplifies by `1/b` for a narrow track.
fn body_scales(dd: DifferentialDrive<f64>, w: WheelVelocities<f64>) -> (f64, f64) {
    let linear = w.left().abs().max(w.right().abs()) * dd.wheel_radius();
    (linear, linear / dd.wheelbase())
}

// ---- round trips ------------------------------------------------------------

#[test]
fn wheels_to_body_to_wheels_round_trip() {
    let mut rng = StdRng::seed_from_u64(0x9966_1234);
    for _ in 0..1000 {
        let dd = rand_geometry(&mut rng);
        let w = rand_wheels(&mut rng);
        let back = dd.inverse(dd.forward(w));
        let scale = w.left().abs().max(w.right().abs());
        assert_exact(back.left(), w.left(), scale, "left");
        assert_exact(back.right(), w.right(), scale, "right");
    }
}

#[test]
fn body_twist_to_wheels_to_body_twist_round_trip() {
    let mut rng = StdRng::seed_from_u64(0x9966_5678);
    for _ in 0..1000 {
        let dd = rand_geometry(&mut rng);
        let c = rand_body_twist(&mut rng);
        let w = dd.inverse(c);
        let back = dd.forward(w);
        let (linear_scale, angular_scale) = body_scales(dd, w);
        assert_exact(back.linear(), c.linear(), linear_scale, "linear");
        assert_exact(back.angular(), c.angular(), angular_scale, "angular");
    }
}

#[test]
fn wheel_rotations_round_trip() {
    let mut rng = StdRng::seed_from_u64(0x9966_abcd);
    for _ in 0..1000 {
        let dd = rand_geometry(&mut rng);
        let w = rand_wheels(&mut rng);
        let d = WheelRotations::new(w.left(), w.right());
        let back = dd.inverse_arc(dd.forward_arc(d));
        let scale = d.left().abs().max(d.right().abs());
        assert_exact(back.left(), d.left(), scale, "left");
        assert_exact(back.right(), d.right(), scale, "right");
    }
}

#[test]
fn body_arc_round_trip() {
    let mut rng = StdRng::seed_from_u64(0x9966_ef01);
    for _ in 0..1000 {
        let dd = rand_geometry(&mut rng);
        let c = rand_body_twist(&mut rng);
        let d = BodyArc::new(c.linear(), c.angular());
        let w = dd.inverse_arc(d);
        let back = dd.forward_arc(w);
        let (linear_scale, angular_scale) =
            body_scales(dd, WheelVelocities::new(w.left(), w.right()));
        assert_exact(back.linear(), d.linear(), linear_scale, "linear");
        assert_exact(back.angular(), d.angular(), angular_scale, "angular");
    }
}

// ---- degenerate motions -----------------------------------------------------

#[test]
fn straight_line_has_zero_angular() {
    let dd = DifferentialDrive::new(0.036_f64, 0.235).unwrap();
    for x in [0.0, 1.0, -3.5, 20.0] {
        assert_eq!(dd.forward(WheelVelocities::new(x, x)).angular(), 0.0);
    }
}

#[test]
fn spin_in_place_has_zero_linear() {
    let dd = DifferentialDrive::new(0.036_f64, 0.235).unwrap();
    for x in [0.0, 1.0, -3.5, 20.0] {
        assert_eq!(dd.forward(WheelVelocities::new(-x, x)).linear(), 0.0);
    }
}

#[test]
fn known_values() {
    let dd = DifferentialDrive::new(0.036_f64, 0.235).unwrap();

    let straight = dd.forward(WheelVelocities::new(10.0, 10.0));
    assert!((straight.linear() - 0.36).abs() < TOL);
    assert!((straight.angular() - 0.0).abs() < TOL);

    let turning = dd.forward(WheelVelocities::new(0.0, 10.0));
    assert!((turning.linear() - 0.18).abs() < TOL);
    assert!((turning.angular() - 0.36 / 0.235).abs() < TOL);
}

// ---- constructor ------------------------------------------------------------

#[test]
fn new_rejects_non_positive_radius() {
    assert_eq!(
        DifferentialDrive::new(0.0_f64, 0.235),
        Err(KinematicsError::NonPositiveParameter)
    );
    assert_eq!(
        DifferentialDrive::new(-1.0_f64, 0.235),
        Err(KinematicsError::NonPositiveParameter)
    );
}

#[test]
fn new_rejects_non_positive_wheelbase() {
    assert_eq!(
        DifferentialDrive::new(0.036_f64, 0.0),
        Err(KinematicsError::NonPositiveParameter)
    );
    assert_eq!(
        DifferentialDrive::new(0.036_f64, -1.0),
        Err(KinematicsError::NonPositiveParameter)
    );
}

#[test]
fn new_rejects_nan() {
    // NaN fails `<= 0`, so without the finiteness check first it would be accepted as valid.
    assert_eq!(
        DifferentialDrive::new(f64::NAN, 0.235),
        Err(KinematicsError::NonFinite)
    );
    assert_eq!(
        DifferentialDrive::new(0.036, f64::NAN),
        Err(KinematicsError::NonFinite)
    );
}

#[test]
fn new_rejects_infinite() {
    assert_eq!(
        DifferentialDrive::new(f64::INFINITY, 0.235),
        Err(KinematicsError::NonFinite)
    );
    assert_eq!(
        DifferentialDrive::new(0.036, f64::NEG_INFINITY),
        Err(KinematicsError::NonFinite)
    );
}

// ---- the nonholonomic tangent seam ------------------------------------------

#[test]
fn to_tangent_has_zero_lateral() {
    let xi = BodyTwist::new(1.0, 2.0).to_tangent();
    assert_eq!(xi, Vector::new([1.0, 0.0, 2.0]));
}

#[test]
fn project_tangent_discards_lateral() {
    let c = BodyTwist::project_tangent(Vector::new([1.0, 9.9, 2.0]));
    assert_eq!(c.linear(), 1.0);
    assert_eq!(c.angular(), 2.0);
}

#[test]
fn tangent_slip_reports_lateral() {
    assert_eq!(BodyTwist::tangent_slip(Vector::new([1.0, 9.9, 2.0])), 9.9);
}

#[test]
fn project_round_trips_only_without_slip() {
    let clean = Vector::new([1.0, 0.0, 2.0]);
    assert_eq!(BodyTwist::project_tangent(clean).to_tangent(), clean);

    let slipping = Vector::new([1.0, 9.9, 2.0]);
    assert_ne!(BodyTwist::project_tangent(slipping).to_tangent(), slipping);
}

#[test]
fn integrate_over_scales_both() {
    let d = BodyTwist::new(2.0, 4.0).integrate_over(0.5);
    assert_eq!(d.linear(), 1.0);
    assert_eq!(d.angular(), 2.0);
}

// ---- the travel seam --------------------------------------------------------

#[test]
fn travel_converters_round_trip() {
    let mut rng = StdRng::seed_from_u64(0x9966_2222);
    for _ in 0..1000 {
        let dd = rand_geometry(&mut rng);
        let (a, b) = (rng.gen_range(-5.0..5.0), rng.gen_range(-5.0..5.0));
        let (left, right) = dd.wheel_travel(dd.wheel_rotations_from_travel(a, b));
        let scale = a.abs().max(b.abs());
        assert_exact(left, a, scale, "left travel");
        assert_exact(right, b, scale, "right travel");
    }
}

// ---- f32 --------------------------------------------------------------------

#[test]
fn round_trips_hold_at_f32() {
    let dd = DifferentialDrive::new(0.036_f32, 0.235).unwrap();
    let bound = 8.0 * f32::EPSILON;

    let w = WheelVelocities::new(1.5_f32, -2.5);
    let back = dd.inverse(dd.forward(w));
    assert!((back.left() - w.left()).abs() <= bound * 2.5);
    assert!((back.right() - w.right()).abs() <= bound * 2.5);

    let c = BodyTwist::new(0.3_f32, 0.9);
    let back = dd.forward(dd.inverse(c));
    assert!((back.linear() - c.linear()).abs() <= bound * 0.9);
    assert!((back.angular() - c.angular()).abs() <= bound * 0.9);
}
