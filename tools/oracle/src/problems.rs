//! Named-problem registry.
//!
//! The generic integrands and least-squares residuals live in `multicalc-testkit`
//! so the host oracle and the bare-metal smoke firmware share one definition; they
//! are re-exported here so fixtures and tests still name them under
//! `multicalc_oracle::problems`. The ODE right-hand sides stay local because they
//! carry the integrator's concrete `f64`/`Vector` signature.

pub use multicalc_testkit::problems::*;

use multicalc::linear_algebra::Vector;

// ODE right-hand sides `y' = f(t, y)`, with the integrator's exact signature so the
// oracle test can pass `&fn`. Each key is mirrored in the Python generator.

/// `y' = -y`; reference `y(t) = e^{-t}`.
pub fn ode_exp_decay(_t: f64, y: &Vector<1>) -> Vector<1> {
    Vector::new([-y[0]])
}

/// Harmonic oscillator `y1' = y2, y2' = -y1`; reference `[cos t, -sin t]`.
pub fn ode_harmonic(_t: f64, y: &Vector<2>) -> Vector<2> {
    Vector::new([y[1], -y[0]])
}

/// Two-body orbit `[x, y, vx, vy]` with `GM = 1`: unit circular orbit of period 2π.
pub fn ode_two_body(_t: f64, y: &Vector<4>) -> Vector<4> {
    let r = (y[0] * y[0] + y[1] * y[1]).sqrt();
    let r3 = r * r * r;
    Vector::new([y[2], y[3], -y[0] / r3, -y[1] / r3])
}

/// Van der Pol oscillator with `μ = 1`: `y1' = y2, y2' = μ(1 − y1²) y2 − y1`.
pub fn ode_van_der_pol_mild(_t: f64, y: &Vector<2>) -> Vector<2> {
    let mu = 1.0;
    Vector::new([y[1], mu * (1.0 - y[0] * y[0]) * y[1] - y[0]])
}
