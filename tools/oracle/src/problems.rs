//! Named-problem registry.
//!
//! Quadrature integrands and least-squares residuals are functions, so they
//! cannot live in a JSON fixture. Instead a fixture names a problem by a stable
//! string key, and both this module and the Python generator implement that key
//! with the identical formula. Adding a problem means adding it on both sides
//! under the same key.

use multicalc::scalar::{Numeric, VectorFn};

/// Returns the `f64` integrand for a quadrature key. Panics on an unknown key.
///
/// Gauss-Hermite folds an `e^{-x^2}` weight and Gauss-Laguerre an `e^{-x}` weight
/// around this integrand; Legendre and the iterative rules integrate it directly.
pub fn integrand_f64(key: &str) -> fn(f64) -> f64 {
    match key {
        "two_x" => |x| 2.0 * x,
        "quartic" => |x| 4.0 * x * x * x - 3.0 * x * x,
        "cube" => |x| x * x * x,
        "x_squared" => |x| x * x,
        "inv_1px2" => |x| 1.0 / (1.0 + x * x),
        "exp_neg" => |x| (-x).exp(),
        other => unreachable!("unknown integrand key {other:?}"),
    }
}

/// Returns the `f32` integrand for a quadrature key. Panics on an unknown key.
pub fn integrand_f32(key: &str) -> fn(f32) -> f32 {
    match key {
        "two_x" => |x| 2.0 * x,
        "quartic" => |x| 4.0 * x * x * x - 3.0 * x * x,
        "cube" => |x| x * x * x,
        "x_squared" => |x| x * x,
        "inv_1px2" => |x| 1.0 / (1.0 + x * x),
        "exp_neg" => |x| (-x).exp(),
        other => unreachable!("unknown integrand key {other:?}"),
    }
}

/// Rosenbrock residual `[10*(x1 - x0^2), 1 - x0]`; the minimum is `x = [1, 1]`.
pub struct Rosenbrock;

impl VectorFn<2, 2> for Rosenbrock {
    fn eval<S: Numeric>(&self, x: &[S; 2]) -> [S; 2] {
        [S::from_f64(10.0) * (x[1] - x[0] * x[0]), S::ONE - x[0]]
    }
}

/// Moré-Garbow-Hillstrom trigonometric function (problem 26) in six variables.
/// Its global minimum is zero.
pub struct Trigonometric6;

impl VectorFn<6, 6> for Trigonometric6 {
    fn eval<S: Numeric>(&self, x: &[S; 6]) -> [S; 6] {
        let n = S::from_f64(6.0);
        let mut cos_sum = S::ZERO;
        for &xj in x {
            cos_sum += xj.cos();
        }
        core::array::from_fn(|i| {
            n - cos_sum + S::from_f64((i + 1) as f64) * (S::ONE - x[i].cos()) - x[i].sin()
        })
    }
}

// Circle-fit target: 40 points sampled exactly on the circle of center (2, -1),
// radius 3. The same formula is mirrored in the Python generator.
const CIRCLE_POINTS: usize = 40;

fn circle_px(i: usize) -> f64 {
    let angle = std::f64::consts::TAU * i as f64 / CIRCLE_POINTS as f64;
    2.0 + 3.0 * angle.cos()
}

fn circle_py(i: usize) -> f64 {
    let angle = std::f64::consts::TAU * i as f64 / CIRCLE_POINTS as f64;
    -1.0 + 3.0 * angle.sin()
}

/// Fit a circle `[cx, cy, r]` to 40 fixed points, minimizing the geometric
/// distance residual `sqrt((x-cx)^2 + (y-cy)^2) - r`. The recovered geometry is
/// center `(2, -1)`, radius `3`.
pub struct CircleFit;

impl VectorFn<3, CIRCLE_POINTS> for CircleFit {
    fn eval<S: Numeric>(&self, p: &[S; 3]) -> [S; CIRCLE_POINTS] {
        let (cx, cy, r) = (p[0], p[1], p[2]);
        core::array::from_fn(|i| {
            let dx = S::from_f64(circle_px(i)) - cx;
            let dy = S::from_f64(circle_py(i)) - cy;
            (dx * dx + dy * dy).sqrt() - r
        })
    }
}

// Gaussian-peaks target: two Gaussians [a, mu, sigma] sampled at 50 points.
const GAUSS_POINTS: usize = 50;
const GAUSS_TRUTH: [f64; 6] = [2.0, 3.0, 0.8, 1.5, 7.0, 1.2];

fn gauss_t(i: usize) -> f64 {
    i as f64 * 10.0 / (GAUSS_POINTS as f64 - 1.0)
}

fn gauss_y(i: usize) -> f64 {
    let t = gauss_t(i);
    let mut y = 0.0;
    for k in 0..2 {
        let a = GAUSS_TRUTH[3 * k];
        let mu = GAUSS_TRUTH[3 * k + 1];
        let sigma = GAUSS_TRUTH[3 * k + 2];
        let z = (t - mu) / sigma;
        y += a * (-(z * z)).exp();
    }
    y
}

/// Fit two Gaussian peaks `[a, mu, sigma]` to a spectrum sampled at 50 points.
/// The residual is `model(p) - y`, with `y` the two-peak signal at the true
/// parameters `[2, 3, 0.8, 1.5, 7, 1.2]`.
pub struct GaussianPeaks;

impl VectorFn<6, GAUSS_POINTS> for GaussianPeaks {
    fn eval<S: Numeric>(&self, p: &[S; 6]) -> [S; GAUSS_POINTS] {
        core::array::from_fn(|i| {
            let t = S::from_f64(gauss_t(i));
            let mut model = S::ZERO;
            for k in 0..2 {
                let a = p[3 * k];
                let mu = p[3 * k + 1];
                let sigma = p[3 * k + 2];
                let z = (t - mu) / sigma;
                model += a * (-(z * z)).exp();
            }
            model - S::from_f64(gauss_y(i))
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    #[test]
    fn integrands_evaluate() {
        assert_eq!(integrand_f64("two_x")(3.0), 6.0);
        assert_eq!(integrand_f64("x_squared")(4.0), 16.0);
        assert_eq!(integrand_f32("cube")(2.0), 8.0);
    }

    #[test]
    fn residuals_vanish_at_the_solution() {
        // Each problem is a zero-residual fit at its true parameters.
        let r = Rosenbrock.eval(&[1.0, 1.0]);
        assert!(r.iter().all(|v: &f64| v.abs() < 1e-12));

        let c = CircleFit.eval(&[2.0, -1.0, 3.0]);
        assert!(c.iter().all(|v: &f64| v.abs() < 1e-12));

        let g = GaussianPeaks.eval(&GAUSS_TRUTH);
        assert!(g.iter().all(|v: &f64| v.abs() < 1e-12));
    }
}
