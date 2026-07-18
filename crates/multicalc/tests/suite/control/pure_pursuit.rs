//! Pure-pursuit tests: sign and magnitude of the curvature, the closed-form value, frame
//! invariance, error handling, the body-twist conversion, and an autodiff sanity check.

use multicalc::control::{Curvature, pure_pursuit_curvature};
use multicalc::error::ControlError;
use multicalc::linear_algebra::Vector;
use multicalc::scalar::{Dual, Numeric};
use multicalc::spatial::{SE2, SO2};

#[test]
fn straight_ahead_gives_zero_curvature() {
    let curvature = pure_pursuit_curvature(SE2::identity(), Vector::new([2.0_f64, 0.0]), 2.0).unwrap();
    assert!(curvature.value().abs() < 1e-12);
}

#[test]
fn mirror_points_give_opposite_curvature() {
    let pose = SE2::identity();
    let lookahead_distance = 2.0_f64;
    let left = pure_pursuit_curvature(pose, Vector::new([1.5, 0.8]), lookahead_distance).unwrap();
    let right = pure_pursuit_curvature(pose, Vector::new([1.5, -0.8]), lookahead_distance).unwrap();
    assert!(left.value() > 0.0 && right.value() < 0.0);
    assert!((left.value() + right.value()).abs() < 1e-12);
}

#[test]
fn side_point_matches_closed_form() {
    let lookahead_distance = 3.0_f64;
    // A point exactly `lookahead_distance` to the left: lateral offset equals the distance.
    let curvature = pure_pursuit_curvature(
        SE2::identity(),
        Vector::new([0.0, lookahead_distance]),
        lookahead_distance,
    )
    .unwrap();
    assert!((curvature.value() - 2.0 / lookahead_distance).abs() < 1e-12);
}

#[test]
fn curvature_is_frame_invariant() {
    let lookahead_distance = 2.5_f64;
    let body_point = Vector::new([1.8_f64, 0.6]);
    let at_origin =
        pure_pursuit_curvature(SE2::identity(), body_point, lookahead_distance).unwrap();

    // The same target expressed in the world frame of a rotated, translated pose.
    let pose = SE2::from_parts(SO2::exp(0.7), Vector::new([3.0, -1.2]));
    let moved =
        pure_pursuit_curvature(pose, pose.act(body_point), lookahead_distance).unwrap();
    assert!((at_origin.value() - moved.value()).abs() < 1e-10);
}

#[test]
fn non_positive_lookahead_and_non_finite_point_are_errors() {
    let pose = SE2::identity();
    assert_eq!(
        pure_pursuit_curvature(pose, Vector::new([1.0_f64, 0.0]), 0.0).err(),
        Some(ControlError::NonPositiveLookaheadDistance)
    );
    assert_eq!(
        pure_pursuit_curvature(pose, Vector::new([1.0_f64, 0.0]), -1.0).err(),
        Some(ControlError::NonPositiveLookaheadDistance)
    );
    assert_eq!(
        pure_pursuit_curvature(pose, Vector::new([f64::NAN, 0.0]), 1.0).err(),
        Some(ControlError::NonFinite)
    );
}

#[test]
fn to_body_twist_uses_speed_and_curvature() {
    let twist = Curvature::new(0.5_f64).to_body_twist(2.0);
    assert_eq!(twist.linear(), 2.0);
    assert_eq!(twist.angular(), 1.0);
}

// Curvature as a function of the lateral offset of the lookahead point, for a pose at the origin.
fn curvature_of_lateral<T: Numeric>(lateral: T) -> T {
    pure_pursuit_curvature(
        SE2::identity(),
        Vector::new([T::from_f64(1.0), lateral]),
        T::from_f64(2.0),
    )
    .unwrap()
    .value()
}

#[test]
fn curvature_derivative_matches_finite_difference() {
    let lateral = 0.5_f64;
    let autodiff = curvature_of_lateral(Dual::variable(lateral)).deriv;
    let h = 1e-6;
    let finite_difference =
        (curvature_of_lateral(lateral + h) - curvature_of_lateral(lateral - h)) / (2.0 * h);
    assert!(
        (autodiff - finite_difference).abs() < 1e-7,
        "autodiff {autodiff}, finite difference {finite_difference}"
    );
}
