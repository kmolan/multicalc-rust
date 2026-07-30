//! The motion and measurement models a Kalman filter needs to track a ground vehicle: a turning-arc
//! process model, and what wheel odometry, an attitude-and-heading sensor, and GPS each see of it.

use multicalc::linear_algebra::{Matrix, Vector, Vector2D};
use multicalc::scalar::{Numeric, VectorFn};

use super::geometry::wrap_angle;

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
        let wrapped = next_heading.wrap_to_pi();
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

/// The difference between an attitude reading `[heading, turn_rate]` and the prediction, with the
/// heading part folded into (-π, π].
pub fn attitude_residual(measured: Vector2D, predicted: Vector2D) -> Vector2D {
    Vector::new([
        wrap_angle(measured[0] - predicted[0]),
        measured[1] - predicted[1],
    ])
}

/// A square matrix with `values` on the diagonal and zeros elsewhere.
pub fn diagonal<const N: usize>(values: [f64; N]) -> Matrix<N, N, f64> {
    Matrix::from_fn(|row, column| if row == column { values[row] } else { 0.0 })
}
