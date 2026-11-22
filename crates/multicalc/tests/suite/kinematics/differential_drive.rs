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

fn random_geometry(rng: &mut StdRng) -> DifferentialDrive<f64> {
    DifferentialDrive::new(rng.gen_range(0.01..0.5), rng.gen_range(0.05..1.0)).unwrap()
}

fn random_wheels(rng: &mut StdRng) -> WheelVelocities<f64> {
    WheelVelocities::new(rng.gen_range(-20.0..20.0), rng.gen_range(-20.0..20.0))
}

fn random_body_twist(rng: &mut StdRng) -> BodyTwist<f64> {
    BodyTwist::new(rng.gen_range(-5.0..5.0), rng.gen_range(-5.0..5.0))
}

/// Error scales for reconstructing a body pair from the wheel pair it maps to. Recovering the
/// linear term cancels the two wheel terms and rescales by the wheel radius; the angular term does
/// the same and then divides by the wheelbase, which amplifies by `1/wheelbase` for a narrow track.
fn body_scales(drive: DifferentialDrive<f64>, wheels: WheelVelocities<f64>) -> (f64, f64) {
    let linear = wheels.left().abs().max(wheels.right().abs()) * drive.wheel_radius();
    (linear, linear / drive.wheelbase())
}

// ---- round trips ------------------------------------------------------------

#[test]
fn wheels_to_body_to_wheels_round_trip() {
    let mut rng = StdRng::seed_from_u64(0x9966_1234);
    for _ in 0..1000 {
        let drive = random_geometry(&mut rng);
        let wheels = random_wheels(&mut rng);
        let recovered = drive.inverse(drive.forward(wheels));
        let scale = wheels.left().abs().max(wheels.right().abs());
        assert_exact(recovered.left(), wheels.left(), scale, "left");
        assert_exact(recovered.right(), wheels.right(), scale, "right");
    }
}

#[test]
fn body_twist_to_wheels_to_body_twist_round_trip() {
    let mut rng = StdRng::seed_from_u64(0x9966_5678);
    for _ in 0..1000 {
        let drive = random_geometry(&mut rng);
        let body = random_body_twist(&mut rng);
        let wheels = drive.inverse(body);
        let recovered = drive.forward(wheels);
        let (linear_scale, angular_scale) = body_scales(drive, wheels);
        assert_exact(recovered.linear(), body.linear(), linear_scale, "linear");
        assert_exact(
            recovered.angular(),
            body.angular(),
            angular_scale,
            "angular",
        );
    }
}

#[test]
fn wheel_rotations_round_trip() {
    let mut rng = StdRng::seed_from_u64(0x9966_abcd);
    for _ in 0..1000 {
        let drive = random_geometry(&mut rng);
        let wheels = random_wheels(&mut rng);
        let rotations = WheelRotations::new(wheels.left(), wheels.right());
        let recovered = drive.inverse_arc(drive.forward_arc(rotations));
        let scale = rotations.left().abs().max(rotations.right().abs());
        assert_exact(recovered.left(), rotations.left(), scale, "left");
        assert_exact(recovered.right(), rotations.right(), scale, "right");
    }
}

#[test]
fn body_arc_round_trip() {
    let mut rng = StdRng::seed_from_u64(0x9966_ef01);
    for _ in 0..1000 {
        let drive = random_geometry(&mut rng);
        let body = random_body_twist(&mut rng);
        let arc = BodyArc::new(body.linear(), body.angular());
        let wheels = drive.inverse_arc(arc);
        let recovered = drive.forward_arc(wheels);
        let (linear_scale, angular_scale) =
            body_scales(drive, WheelVelocities::new(wheels.left(), wheels.right()));
        assert_exact(recovered.linear(), arc.linear(), linear_scale, "linear");
        assert_exact(recovered.angular(), arc.angular(), angular_scale, "angular");
    }
}

// ---- degenerate motions -----------------------------------------------------

#[test]
fn straight_line_has_zero_angular() {
    let drive = DifferentialDrive::new(0.036_f64, 0.235).unwrap();
    for speed in [0.0, 1.0, -3.5, 20.0] {
        assert_eq!(
            drive.forward(WheelVelocities::new(speed, speed)).angular(),
            0.0
        );
    }
}

#[test]
fn spin_in_place_has_zero_linear() {
    let drive = DifferentialDrive::new(0.036_f64, 0.235).unwrap();
    for speed in [0.0, 1.0, -3.5, 20.0] {
        assert_eq!(
            drive.forward(WheelVelocities::new(-speed, speed)).linear(),
            0.0
        );
    }
}

#[test]
fn known_values() {
    let drive = DifferentialDrive::new(0.036_f64, 0.235).unwrap();

    let straight = drive.forward(WheelVelocities::new(10.0, 10.0));
    assert!((straight.linear() - 0.36).abs() < TOL);
    assert!((straight.angular() - 0.0).abs() < TOL);

    let turning = drive.forward(WheelVelocities::new(0.0, 10.0));
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
    let tangent = BodyTwist::new(1.0, 2.0).to_tangent();
    assert_eq!(tangent, Vector::new([1.0, 0.0, 2.0]));
}

#[test]
fn project_tangent_discards_lateral() {
    let body = BodyTwist::project_tangent(Vector::new([1.0, 9.9, 2.0]));
    assert_eq!(body.linear(), 1.0);
    assert_eq!(body.angular(), 2.0);
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
    let arc = BodyTwist::new(2.0, 4.0).integrate_over(0.5);
    assert_eq!(arc.linear(), 1.0);
    assert_eq!(arc.angular(), 2.0);
}

// ---- the travel seam --------------------------------------------------------

#[test]
fn travel_converters_round_trip() {
    let mut rng = StdRng::seed_from_u64(0x9966_2222);
    for _ in 0..1000 {
        let drive = random_geometry(&mut rng);
        let left_travel = rng.gen_range(-5.0..5.0);
        let right_travel = rng.gen_range(-5.0..5.0);
        let (left, right) =
            drive.wheel_travel(drive.wheel_rotations_from_travel(left_travel, right_travel));
        let scale = left_travel.abs().max(right_travel.abs());
        assert_exact(left, left_travel, scale, "left travel");
        assert_exact(right, right_travel, scale, "right travel");
    }
}

// ---- f32 --------------------------------------------------------------------

#[test]
fn round_trips_hold_at_f32() {
    let drive = DifferentialDrive::new(0.036_f32, 0.235).unwrap();
    let bound = 8.0 * f32::EPSILON;

    let wheels = WheelVelocities::new(1.5_f32, -2.5);
    let recovered = drive.inverse(drive.forward(wheels));
    assert!((recovered.left() - wheels.left()).abs() <= bound * 2.5);
    assert!((recovered.right() - wheels.right()).abs() <= bound * 2.5);

    let body = BodyTwist::new(0.3_f32, 0.9);
    let recovered = drive.forward(drive.inverse(body));
    assert!((recovered.linear() - body.linear()).abs() <= bound * 0.9);
    assert!((recovered.angular() - body.angular()).abs() <= bound * 0.9);
}
