use multicalc::linear_algebra::Vector;
use multicalc::scalar::VectorFn;
use multicalc_demos::sim::geometry::wrap_angle;
use multicalc_demos::sim::kalman_filter_models::{
    AttitudeHeadingModel, CoordinatedTurnModel, GlobalPositionModel, WheelOdometryModel,
    attitude_residual, diagonal,
};
use std::f64::consts::{FRAC_PI_2, PI};

#[test]
fn a_straight_arc_advances_along_the_heading() {
    // Zero turn rate: the vehicle slides forward along its heading, which does not change.
    let model = CoordinatedTurnModel { timestep: 0.5 };
    let next = model.eval::<f64>(&[0.0, 0.0, 0.0, 2.0, 0.0]);
    assert!((next[0] - 1.0).abs() < 1e-12, "x: {}", next[0]);
    assert!(next[1].abs() < 1e-12, "y: {}", next[1]);
    assert!(next[2].abs() < 1e-12, "heading: {}", next[2]);
    assert_eq!([next[3], next[4]], [2.0, 0.0]);
}

#[test]
fn a_quarter_turn_lands_on_the_arc() {
    // Speed 1, turn rate 1, over a quarter-turn's worth of time: heading sweeps 0 → π/2 and the
    // unit-radius arc lands at (1, 1).
    let model = CoordinatedTurnModel {
        timestep: FRAC_PI_2,
    };
    let next = model.eval::<f64>(&[0.0, 0.0, 0.0, 1.0, 1.0]);
    assert!((next[0] - 1.0).abs() < 1e-12, "x: {}", next[0]);
    assert!((next[1] - 1.0).abs() < 1e-12, "y: {}", next[1]);
    assert!((next[2] - FRAC_PI_2).abs() < 1e-12, "heading: {}", next[2]);
}

#[test]
fn the_process_model_folds_the_output_heading() {
    // A turn that sums past π should come back folded into range, not as a bare 4 radians.
    let model = CoordinatedTurnModel { timestep: 4.0 };
    let next = model.eval::<f64>(&[0.0, 0.0, 0.0, 0.0, 1.0]);
    assert!(next[2] > -PI && next[2] <= PI, "out of range: {}", next[2]);
    assert!(
        (next[2] - wrap_angle(4.0)).abs() < 1e-12,
        "heading: {}",
        next[2]
    );
}

#[test]
fn each_measurement_model_reads_the_right_components() {
    let state: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    assert_eq!(WheelOdometryModel.eval(&state), [4.0, 5.0]);
    assert_eq!(AttitudeHeadingModel.eval(&state), [3.0, 5.0]);
    assert_eq!(GlobalPositionModel.eval(&state), [1.0, 2.0]);
}

#[test]
fn attitude_residual_folds_the_heading() {
    // A reading just past -π against a prediction just under +π: the true gap is small, so the
    // folded residual must be small too, not near -2π.
    let residual = attitude_residual(
        Vector::new([-PI + 0.05, 0.5]),
        Vector::new([PI - 0.05, 0.3]),
    );
    assert!(
        residual[0].abs() < 0.11,
        "heading residual not folded: {}",
        residual[0]
    );
    assert!(
        (residual[1] - 0.2).abs() < 1e-12,
        "turn-rate residual: {}",
        residual[1]
    );
}

#[test]
fn diagonal_places_values_on_the_diagonal() {
    let matrix = diagonal([1.0, 2.0, 3.0]);
    for row in 0..3 {
        for column in 0..3 {
            let expected = if row == column {
                [1.0, 2.0, 3.0][row]
            } else {
                0.0
            };
            assert_eq!(matrix[(row, column)], expected, "at ({row}, {column})");
        }
    }
}
