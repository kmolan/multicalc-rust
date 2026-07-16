//! The unicycle plant as an ODE right-hand side.

use crate::kinematics::diff_drive::ChassisRate;
use crate::linear_algebra::Vector;
use crate::scalar::Numeric;

/// The unicycle plant at a held chassis rate: `f(t, [x, y, θ]) = [v cosθ, v sinθ, ω]`.
///
/// Time-invariant; `t` is present to match the [`Rk4`](crate::Rk4) and [`Rk45`](crate::Rk45)
/// closure shape.
///
/// ```
/// use multicalc::kinematics::{ChassisRate, Unicycle};
/// use multicalc::linear_algebra::Vector;
/// use multicalc::ode::Rk4;
/// let plant = Unicycle::new(ChassisRate::new(1.0_f64, 0.0));
/// let state = Rk4::step(&plant.field(), 0.0, &Vector::new([0.0, 0.0, 0.0]), 0.1);
/// assert!((state[0] - 0.1).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Unicycle<T: Numeric> {
    rate: ChassisRate<T>,
}

impl<T: Numeric> Unicycle<T> {
    /// A plant holding `rate`.
    #[inline]
    pub fn new(rate: ChassisRate<T>) -> Self {
        Unicycle { rate }
    }

    /// The state derivative at `state = [x, y, θ]`.
    #[inline]
    pub fn derivative(self, state: &Vector<3, T>) -> Vector<3, T> {
        let v = self.rate.linear();
        let theta = state[2];
        Vector::new([v * theta.cos(), v * theta.sin(), self.rate.angular()])
    }

    /// The derivative as an [`Rk4`](crate::Rk4)/[`Rk45`](crate::Rk45) closure.
    #[inline]
    pub fn field(self) -> impl Fn(T, &Vector<3, T>) -> Vector<3, T> {
        move |_t, y| self.derivative(y)
    }
}
