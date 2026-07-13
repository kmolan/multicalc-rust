//! Ordinary differential equation integrators.
//!
//! [`Rk4`] is a fixed-step classic Runge–Kutta method; [`Rk45`] is an adaptive
//! Dormand–Prince 5(4) method with PI step control and cubic-Hermite dense output.
//! Both are generic over the state `Vector<N, T>` and any [`Numeric`](crate::Numeric)
//! scalar, so the same integrator runs at `f32`/`f64` or through an autodiff scalar.

mod rk4;
mod rk45;
mod tableau;

pub use rk4::Rk4;
pub use rk45::{Rk45, Step};
