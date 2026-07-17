//! Extended Kalman filter goldens, invariants, and error paths.

use multicalc::scalar::{Numeric, VectorFn};

/// Unicycle motion: [x, y, heading] driven by a forward and an angular velocity over one step.
struct UnicycleMotion {
    timestep: f64,
    forward_velocity: f64,
    angular_velocity: f64,
}

impl VectorFn<3, 3> for UnicycleMotion {
    fn eval<S: Numeric>(&self, state: &[S; 3]) -> [S; 3] {
        let timestep = S::from_f64(self.timestep);
        let forward_velocity = S::from_f64(self.forward_velocity);
        let angular_velocity = S::from_f64(self.angular_velocity);
        let heading = state[2];
        [
            state[0] + forward_velocity * heading.cos() * timestep,
            state[1] + forward_velocity * heading.sin() * timestep,
            heading + angular_velocity * timestep,
        ]
    }
}

/// Range and bearing to a known landmark, from a [x, y, heading] pose.
struct LandmarkRangeAndBearing {
    landmark_x: f64,
    landmark_y: f64,
}

impl VectorFn<3, 2> for LandmarkRangeAndBearing {
    fn eval<S: Numeric>(&self, state: &[S; 3]) -> [S; 2] {
        let to_landmark_x = S::from_f64(self.landmark_x) - state[0];
        let to_landmark_y = S::from_f64(self.landmark_y) - state[1];
        [
            (to_landmark_x * to_landmark_x + to_landmark_y * to_landmark_y).sqrt(),
            to_landmark_y.atan2(to_landmark_x) - state[2],
        ]
    }
}

/// Wraps an angle to (−π, π].
fn wrap_to_pi<T: Numeric>(angle: T) -> T {
    angle.sin().atan2(angle.cos())
}
