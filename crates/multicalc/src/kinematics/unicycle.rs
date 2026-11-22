//! The unicycle plant as an ODE right-hand side.

use crate::kinematics::differential_drive::BodyTwist;
use crate::linear_algebra::{Vector, Vector3D};
use crate::scalar::Numeric;

/// The unicycle plant at a held body twist: `f(t, [x, y, θ]) = [velocity cosθ, velocity sinθ, ω]`.
///
/// Time-invariant; `t` is present to match the [`Rk4`](crate::Rk4) and [`Rk45`](crate::Rk45)
/// closure shape.
///
/// ```
/// use multicalc::kinematics::{BodyTwist, Unicycle};
/// use multicalc::linear_algebra::Vector;
/// use multicalc::ode::Rk4;
/// let command = BodyTwist::new(1.0_f64, 0.0);   // 1 m/s forward, no turn
/// let plant = Unicycle::new(command);
///
/// let start_time = 0.0;
/// let start_pose = Vector::new([0.0, 0.0, 0.0]);
/// let timestep = 0.1;
/// let state = Rk4::step(&plant.field(), start_time, &start_pose, timestep);
/// assert!((state[0] - 0.1).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Unicycle<T: Numeric = f64> {
    twist: BodyTwist<T>,
}

impl<T: Numeric> Unicycle<T> {
    /// A plant holding `twist`.
    #[inline]
    #[must_use]
    pub fn new(twist: BodyTwist<T>) -> Self {
        Unicycle { twist }
    }

    /// The state derivative at `state = [x, y, θ]`.
    #[inline]
    pub fn derivative(self, state: &Vector3D<T>) -> Vector3D<T> {
        let velocity = self.twist.linear();
        let theta = state[2];
        Vector::new([
            velocity * theta.cos(),
            velocity * theta.sin(),
            self.twist.angular(),
        ])
    }

    /// The derivative as an [`Rk4`](crate::Rk4)/[`Rk45`](crate::Rk45) closure.
    #[inline]
    pub fn field(self) -> impl Fn(T, &Vector3D<T>) -> Vector3D<T> {
        move |_t, y| self.derivative(y)
    }
}
