//! The unicycle plant as an ODE right-hand side.

use crate::kinematics::differential_drive::BodyTwist;
use crate::linear_algebra::Vector;
use crate::scalar::Numeric;

/// The unicycle plant at a held body twist: `f(t, [x, y, θ]) = [v cosθ, v sinθ, ω]`.
///
/// Time-invariant; `t` is present to match the [`Rk4`](crate::Rk4) and [`Rk45`](crate::Rk45)
/// closure shape.
///
/// ```
/// use multicalc::kinematics::{BodyTwist, Unicycle};
/// use multicalc::linear_algebra::Vector;
/// use multicalc::ode::Rk4;
/// let plant = Unicycle::new(BodyTwist::new(1.0_f64, 0.0));
/// let state = Rk4::step(&plant.field(), 0.0, &Vector::new([0.0, 0.0, 0.0]), 0.1);
/// let [x, _, _] = *state.as_array();
/// assert!((x - 0.1).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Unicycle<T: Numeric> {
    twist: BodyTwist<T>,
}

impl<T: Numeric> Unicycle<T> {
    /// A plant holding `twist`.
    #[inline]
    pub fn new(twist: BodyTwist<T>) -> Self {
        Unicycle { twist }
    }

    /// The state derivative at `state = [x, y, θ]`.
    #[inline]
    pub fn derivative(self, state: &Vector<3, T>) -> Vector<3, T> {
        let v = self.twist.linear();
        let [_, _, theta] = *state.as_array();
        Vector::new([v * theta.cos(), v * theta.sin(), self.twist.angular()])
    }

    /// The derivative as an [`Rk4`](crate::Rk4)/[`Rk45`](crate::Rk45) closure.
    #[inline]
    pub fn field(self) -> impl Fn(T, &Vector<3, T>) -> Vector<3, T> {
        move |_t, y| self.derivative(y)
    }
}
