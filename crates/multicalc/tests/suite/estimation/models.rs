//! The ready-made process and measurement models.

use multicalc::error::EstimationError;
use multicalc::estimation::{
    ConstantTurnAndSpeed, DirectMeasurement, residual_with_wrapped_angles,
};
use multicalc::linear_algebra::Vector;
use multicalc::scalar::{Dual, VectorFn};

const HALF_TURN: f64 = core::f64::consts::PI;
const WHOLE_TURN: f64 = core::f64::consts::TAU;
const QUARTER_TURN: f64 = core::f64::consts::FRAC_PI_2;

#[test]
fn a_vehicle_not_turning_runs_straight() {
    let timestep = 0.1;
    let speed = 2.0;
    let motion = ConstantTurnAndSpeed { timestep };
    let facing_along_x = [0.0, 0.0, 0.0, speed, 0.0];
    let moved = motion.eval(&facing_along_x);
    assert!((moved[0] - speed * timestep).abs() < 1e-12);
    assert!(moved[1].abs() < 1e-12);
    assert!(moved[2].abs() < 1e-12);
    assert!((moved[3] - speed).abs() < 1e-12, "speed is carried across");
    assert!(moved[4].abs() < 1e-12, "turn rate is carried across");
}

/// The straight-line branch: a turn rate under the guard must not divide by nearly nothing.
#[test]
fn a_turn_rate_too_small_to_use_is_run_straight_instead() {
    let timestep = 0.1;
    let speed = 2.0_f64;
    let motion = ConstantTurnAndSpeed { timestep };
    for turn_rate in [0.0, 1e-12, 1e-9, -1e-9] {
        let barely_turning = [0.0, 0.0, 0.0, speed, turn_rate];
        let moved = motion.eval(&barely_turning);
        assert!(moved[0].is_finite() && moved[1].is_finite(), "{turn_rate}");
        assert!((moved[0] - speed * timestep).abs() < 1e-9, "{turn_rate}");
        assert!(moved[1].abs() < 1e-9, "{turn_rate}");
    }
}

#[test]
fn a_vehicle_turning_traces_an_arc() {
    // A whole quarter turn in one tick at unit speed: the arc radius is 1 / (π/2).
    let motion = ConstantTurnAndSpeed { timestep: 1.0 };
    let speed = 1.0;
    let turning_left = [0.0, 0.0, 0.0, speed, QUARTER_TURN];
    let moved = motion.eval(&turning_left);
    let radius = speed / QUARTER_TURN;
    assert!((moved[0] - radius).abs() < 1e-12, "{moved:?}");
    assert!((moved[1] - radius).abs() < 1e-12, "{moved:?}");
    assert!((moved[2] - QUARTER_TURN).abs() < 1e-12);
}

#[test]
fn turning_right_curves_the_other_way() {
    let motion = ConstantTurnAndSpeed { timestep: 1.0 };
    let turning_right = [0.0, 0.0, 0.0, 1.0, -QUARTER_TURN];
    let moved = motion.eval(&turning_right);
    assert!(moved[1] < 0.0, "a right turn carries the vehicle downward");
    assert!(moved[0] > 0.0);
}

#[test]
fn the_heading_is_folded_back_into_range() {
    let motion = ConstantTurnAndSpeed { timestep: 1.0 };
    let nearly_facing_backward = [0.0, 0.0, 3.0, 1.0, 1.0];
    let moved = motion.eval(&nearly_facing_backward);
    assert!(
        (moved[2] - (4.0 - WHOLE_TURN)).abs() < 1e-12,
        "{}",
        moved[2]
    );
    assert!(moved[2] > -HALF_TURN && moved[2] <= HALF_TURN);
}

/// The filters differentiate this model to get their Jacobians, so the derivative has to survive
/// the branch inside it.
#[test]
fn derivatives_flow_through_the_model() {
    let timestep = 0.1;
    let motion = ConstantTurnAndSpeed { timestep };
    // Seed the speed with a derivative and read how far x moves per unit of speed.
    let seeded_on_speed = [
        Dual::new(0.0, 0.0),
        Dual::new(0.0, 0.0),
        Dual::new(0.0, 0.0),
        Dual::new(2.0, 1.0),
        Dual::new(0.0, 0.0),
    ];
    let moved = motion.eval(&seeded_on_speed);
    // Facing along x with no turn, x advances by speed times the tick, so the slope is the tick.
    assert!((moved[0].deriv - timestep).abs() < 1e-12);
}

#[test]
fn a_direct_measurement_reads_the_components_it_was_given() {
    let state = [1.0, 2.0, 3.0, 4.0, 5.0];
    let position_fix = DirectMeasurement::<5, 2>::try_new([0, 1]).unwrap();
    let encoders = DirectMeasurement::<5, 2>::try_new([3, 4]).unwrap();
    assert_eq!(position_fix.eval(&state), [1.0, 2.0]);
    assert_eq!(encoders.eval(&state), [4.0, 5.0]);
    assert_eq!(position_fix.indices(), [0, 1]);
}

#[test]
fn a_direct_measurement_keeps_the_order_it_was_given() {
    let state = [1.0, 2.0, 3.0, 4.0, 5.0];
    let backward = DirectMeasurement::<5, 2>::try_new([4, 3]).unwrap();
    assert_eq!(backward.eval(&state), [5.0, 4.0]);
    // A component may be read more than once.
    let twice = DirectMeasurement::<5, 3>::try_new([4, 0, 4]).unwrap();
    assert_eq!(twice.eval(&state), [5.0, 1.0, 5.0]);
}

#[test]
fn a_direct_measurement_rejects_a_component_the_state_does_not_have() {
    assert_eq!(
        DirectMeasurement::<5, 2>::try_new([0, 5]).unwrap_err(),
        EstimationError::StateIndexOutOfRange
    );
    assert_eq!(
        DirectMeasurement::<5, 1>::try_new([usize::MAX]).unwrap_err(),
        EstimationError::StateIndexOutOfRange
    );
    // The last component of a five-component state is index four.
    assert!(DirectMeasurement::<5, 1>::try_new([4]).is_ok());
}

#[test]
fn a_residual_folds_only_the_components_named_as_angles() {
    let measured = Vector::new([3.0, 10.0]);
    let predicted = Vector::new([-3.0, 4.0]);
    let heading_only = residual_with_wrapped_angles(measured, predicted, &[0]);
    assert!((heading_only[0] - (6.0 - WHOLE_TURN)).abs() < 1e-12);
    assert!((heading_only[1] - 6.0).abs() < 1e-12);
}

#[test]
fn a_residual_with_no_angles_is_a_plain_subtraction() {
    let measured = Vector::new([3.0_f64, 10.0]);
    let predicted = Vector::new([-3.0, 4.0]);
    let plain = residual_with_wrapped_angles(measured, predicted, &[]);
    assert!((plain[0] - 6.0).abs() < 1e-12);
    assert!((plain[1] - 6.0).abs() < 1e-12);
}

#[test]
fn a_residual_can_fold_every_component() {
    let measured = Vector::new([3.0, 3.0]);
    let predicted = Vector::new([-3.0, -3.0]);
    let both = residual_with_wrapped_angles(measured, predicted, &[0, 1]);
    for component in 0..2 {
        assert!((both[component] - (6.0 - WHOLE_TURN)).abs() < 1e-12);
    }
}

#[test]
fn a_small_difference_is_left_alone() {
    let measured = Vector::new([0.2_f64]);
    let predicted = Vector::new([0.1]);
    let residual = residual_with_wrapped_angles(measured, predicted, &[0]);
    assert!((residual[0] - 0.1).abs() < 1e-12);
}

#[test]
fn the_models_work_at_f32() {
    let state = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let model = DirectMeasurement::<5, 2>::try_new([0, 3]).unwrap();
    assert_eq!(model.eval(&state), [1.0_f32, 4.0]);

    let residual = residual_with_wrapped_angles(Vector::new([3.0_f32]), Vector::new([-3.0]), &[0]);
    assert!((residual[0] - (6.0 - core::f32::consts::TAU)).abs() < 1e-5);

    let motion = ConstantTurnAndSpeed { timestep: 0.1 };
    let moved = motion.eval(&[0.0_f32, 0.0, 0.0, 2.0, 0.0]);
    assert!((moved[0] - 0.2).abs() < 1e-6);
}
