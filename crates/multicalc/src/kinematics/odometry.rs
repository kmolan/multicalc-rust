//! Pose integration on SE(2) and the odometry process model.

use crate::kinematics::differential_drive::BodyArc;
use crate::linear_algebra::Vector;
use crate::scalar::{Numeric, VectorFn};
use crate::spatial::{SE2, SO2};

/// Advances `pose` by one odometry increment along the exact constant-twist arc,
/// `pose · exp([Δs, 0, Δθ])`.
///
/// Assumes the body twist is constant across the tick (a zero-order hold on the wheel velocities).
/// Under that assumption the result is exact at any step size — the residual is the hold itself,
/// not integration error.
///
/// Straight-line motion is handled by [`SE2::exp`]'s Taylor branch. Its guard is on the heading
/// *increment*, not the yaw rate, so the tick rate selects the branch: at 1 kHz the branch engages
/// below roughly 1e-3 rad/s. The series is accurate and derivative-continuous there.
///
/// ```
/// use multicalc::kinematics::{BodyArc, integrate};
/// use multicalc::spatial::SE2;
/// // Two ticks of 0.05 rad turn each leave the pose rotated by 0.1 rad.
/// let d = BodyArc::new(0.1_f64, 0.05);
/// let pose = integrate(integrate(SE2::identity(), d), d);
/// assert!((pose.rotation().log() - 0.1).abs() < 1e-12);
/// ```
#[inline]
pub fn integrate<T: Numeric>(pose: SE2<T>, d: BodyArc<T>) -> SE2<T> {
    pose * SE2::exp(Vector::new([d.linear(), T::ZERO, d.angular()]))
}

/// The odometry process model, `[x, y, θ, Δs, Δθ] → [x', y', θ']`.
///
/// Autodiff through this gives both Jacobians a filter needs from one function: columns 0..3 are the
/// state Jacobian, columns 3..5 the control Jacobian.
///
/// The output heading is wrapped to `(−π, π]`, so a filter consuming this must wrap its heading
/// innovation to the same interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OdometryStep;

impl VectorFn<5, 3> for OdometryStep {
    fn eval<S: Numeric>(&self, p: &[S; 5]) -> [S; 3] {
        let pose = SE2::from_parts(SO2::exp(p[2]), Vector::new([p[0], p[1]]));
        let next = pose * SE2::exp(Vector::new([p[3], S::ZERO, p[4]]));
        let t = next.translation();
        [t[0], t[1], next.rotation().log()]
    }
}
