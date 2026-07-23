//! The EKF's coordinated-turn process model and the odometry, attitude-heading, and global-position
//! measurement models.

use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::scalar::{Numeric, VectorFn};

use super::inertial_measurement_unit::InertialReading;
use super::wrap_angle;

/// Rolls the state `[x, y, heading, speed, turn_rate]` forward one tick along a turning arc.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoordinatedTurnModel {
    /// The length of one tick, in seconds.
    pub timestep: f64,
}

impl VectorFn<5, 5> for CoordinatedTurnModel {
    fn eval<S: Numeric>(&self, state: &[S; 5]) -> [S; 5] {
        let [x, y, heading, speed, turn_rate] = *state;
        let dt = S::from_f64(self.timestep);
        let next_heading = heading + turn_rate * dt;
        // Follow the arc, but straighten it when the turn rate is tiny so `speed / turn_rate` cannot
        // blow up. The branch is on the plain value, so the model still differentiates cleanly.
        let (next_x, next_y) = if turn_rate.abs() > S::from_f64(1e-6) {
            let radius = speed / turn_rate;
            (
                x + radius * (next_heading.sin() - heading.sin()),
                y + radius * (heading.cos() - next_heading.cos()),
            )
        } else {
            (
                x + speed * heading.cos() * dt,
                y + speed * heading.sin() * dt,
            )
        };
        // Fold the output heading back into range by subtracting whole turns.
        let wrapped = next_heading - S::TWO_PI * (next_heading / S::TWO_PI).round();
        [next_x, next_y, wrapped, speed, turn_rate]
    }
}

/// Wheel odometry sees the forward speed and turn rate: the last two state components.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WheelOdometryModel;
impl VectorFn<5, 2> for WheelOdometryModel {
    fn eval<S: Numeric>(&self, state: &[S; 5]) -> [S; 2] {
        [state[3], state[4]]
    }
}

/// The attitude-and-heading sensor sees the heading and the turn rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttitudeHeadingModel;
impl VectorFn<5, 2> for AttitudeHeadingModel {
    fn eval<S: Numeric>(&self, state: &[S; 5]) -> [S; 2] {
        [state[2], state[4]]
    }
}

/// GPS sees the position: the first two state components.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlobalPositionModel;
impl VectorFn<5, 2> for GlobalPositionModel {
    fn eval<S: Numeric>(&self, state: &[S; 5]) -> [S; 2] {
        [state[0], state[1]]
    }
}

/// The difference between an attitude reading and the prediction, heading folded into (-π, π].
pub fn attitude_residual(measured: InertialReading, predicted: Vector<2, f64>) -> Vector<2, f64> {
    Vector::new([
        wrap_angle(measured.heading - predicted[0]),
        measured.yaw_rate - predicted[1],
    ])
}

/// A square matrix with `values` on the diagonal and zeros elsewhere.
pub fn diagonal<const N: usize>(values: [f64; N]) -> Matrix<N, N, f64> {
    Matrix::from_fn(|row, column| if row == column { values[row] } else { 0.0 })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    #[test]
    fn a_straight_arc_advances_along_the_heading() {
        // Zero turn rate: the vehicle slides forward along its heading, which does not change.
        let model = CoordinatedTurnModel { timestep: 0.5 };
        let next = model.eval(&[0.0, 0.0, 0.0, 2.0, 0.0]);
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
        let next = model.eval(&[0.0, 0.0, 0.0, 1.0, 1.0]);
        assert!((next[0] - 1.0).abs() < 1e-12, "x: {}", next[0]);
        assert!((next[1] - 1.0).abs() < 1e-12, "y: {}", next[1]);
        assert!((next[2] - FRAC_PI_2).abs() < 1e-12, "heading: {}", next[2]);
    }

    #[test]
    fn the_process_model_folds_the_output_heading() {
        // A turn that sums past π should come back folded into range, not as a bare 4 radians.
        let model = CoordinatedTurnModel { timestep: 4.0 };
        let next = model.eval(&[0.0, 0.0, 0.0, 0.0, 1.0]);
        assert!(next[2] > -PI && next[2] <= PI, "out of range: {}", next[2]);
        assert!(
            (next[2] - wrap_angle(4.0)).abs() < 1e-12,
            "heading: {}",
            next[2]
        );
    }

    #[test]
    fn each_measurement_model_reads_the_right_components() {
        let state = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(WheelOdometryModel.eval(&state), [4.0, 5.0]);
        assert_eq!(AttitudeHeadingModel.eval(&state), [3.0, 5.0]);
        assert_eq!(GlobalPositionModel.eval(&state), [1.0, 2.0]);
    }

    #[test]
    fn attitude_residual_folds_the_heading() {
        // A reading just past -π against a prediction just under +π: the true gap is small, so the
        // folded residual must be small too, not near -2π.
        let reading = InertialReading {
            heading: -PI + 0.05,
            yaw_rate: 0.5,
        };
        let residual = attitude_residual(reading, Vector::new([PI - 0.05, 0.3]));
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
}
