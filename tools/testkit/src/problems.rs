//! Named-problem registry.
//!
//! Quadrature integrands and least-squares residuals are functions, so they
//! cannot live in a JSON fixture. Instead a fixture names a problem by a stable
//! string key, and both this module and the Python generator implement that key
//! with the identical formula. Adding a problem means adding it on both sides
//! under the same key.

use multicalc::scalar::{Numeric, ScalarFn, ScalarFnN, VectorFn};

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
        "exp_neg" => |x| Numeric::exp(-x),
        "exp_neg_sq" => |x| Numeric::exp(-(x * x)),
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
        "exp_neg" => |x| Numeric::exp(-x),
        "exp_neg_sq" => |x| Numeric::exp(-(x * x)),
        other => unreachable!("unknown integrand key {other:?}"),
    }
}

/// Transcendental `g(x, y, z) = y·sin x + x·cos y + x·y·eᶻ`.
pub struct G;

impl ScalarFnN<3> for G {
    fn eval<S: Numeric>(&self, v: &[S; 3]) -> S {
        v[1] * v[0].sin() + v[0] * v[1].cos() + v[0] * v[1] * v[2].exp()
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
    let angle = core::f64::consts::TAU * i as f64 / CIRCLE_POINTS as f64;
    2.0 + 3.0 * angle.cos()
}

fn circle_py(i: usize) -> f64 {
    let angle = core::f64::consts::TAU * i as f64 / CIRCLE_POINTS as f64;
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
        y += a * Numeric::exp(-(z * z));
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

/// Hessian target `f(x, y, z) = y·sin x + 2·x·eʸ + z²`.
pub struct HessianTarget;

impl ScalarFnN<3> for HessianTarget {
    fn eval<S: Numeric>(&self, v: &[S; 3]) -> S {
        v[1] * v[0].sin() + S::from_f64(2.0) * v[0] * v[1].exp() + v[2] * v[2]
    }
}

/// Jacobian target `[x·y·z, x² + y²]`, 3 inputs and 2 outputs.
pub struct Jac23;

impl VectorFn<3, 2> for Jac23 {
    fn eval<S: Numeric>(&self, v: &[S; 3]) -> [S; 2] {
        [v[0] * v[1] * v[2], v[0] * v[0] + v[1] * v[1]]
    }
}

/// Jacobian target with cyclic coupling `aᵢ·aᵢ₊₁ + aᵢ₊₂`, 6 inputs and 6 outputs.
pub struct Jac66;

impl VectorFn<6, 6> for Jac66 {
    fn eval<S: Numeric>(&self, a: &[S; 6]) -> [S; 6] {
        [
            a[0] * a[1] + a[2],
            a[1] * a[2] + a[3],
            a[2] * a[3] + a[4],
            a[3] * a[4] + a[5],
            a[4] * a[5] + a[0],
            a[5] * a[0] + a[1],
        ]
    }
}

/// Vector field `[y, -x, 2z]`; curl is `[0, 0, -2]` and divergence is `2`.
pub struct VField3d;

impl VectorFn<3, 3> for VField3d {
    fn eval<S: Numeric>(&self, v: &[S; 3]) -> [S; 3] {
        [v[1], -v[0], S::from_f64(2.0) * v[2]]
    }
}

/// Approximation target `f(x, y, z) = x + y² + z³`.
pub struct ApproxTarget;

impl ScalarFnN<3> for ApproxTarget {
    fn eval<S: Numeric>(&self, v: &[S; 3]) -> S {
        v[0] + v[1] * v[1] + v[2] * v[2] * v[2]
    }
}

/// Wien's displacement equation `-5 + x + 5·e^{-x}`; the nonzero root is near 4.965.
pub struct Wien;

impl ScalarFn for Wien {
    fn eval<S: Numeric>(&self, x: S) -> S {
        S::from_f64(-5.0) + x + S::from_f64(5.0) * (-x).exp()
    }
}

/// Kepler's equation `E - e·sin E - M`, relating the mean anomaly `M` to the
/// eccentric anomaly `E` of an orbit with eccentricity `e`.
pub struct Kepler {
    pub e: f64,
    pub m: f64,
}

impl ScalarFn for Kepler {
    fn eval<S: Numeric>(&self, big_e: S) -> S {
        big_e - S::from_f64(self.e) * big_e.sin() - S::from_f64(self.m)
    }
}

/// Colebrook-White equation for the Darcy friction factor `f` of turbulent pipe
/// flow: `1/√f + 2·log₁₀(rel_roughness/3.7 + 2.51/(Re·√f))`.
pub struct Colebrook {
    pub reynolds: f64,
    pub rel_roughness: f64,
}

impl ScalarFn for Colebrook {
    fn eval<S: Numeric>(&self, f: S) -> S {
        let re = S::from_f64(self.reynolds);
        let eps = S::from_f64(self.rel_roughness);
        let root_f = f.sqrt();
        let inner = eps / S::from_f64(3.7) + S::from_f64(2.51) / (re * root_f);
        let log10 = inner.ln() / S::from_f64(10.0).ln();
        S::ONE / root_f + S::TWO * log10
    }
}

/// Sigmoid `x / √(1 + x²)`; the only root is `x = 0`.
pub struct Sigmoid;

impl ScalarFn for Sigmoid {
    fn eval<S: Numeric>(&self, x: S) -> S {
        x / (S::ONE + x * x).sqrt()
    }
}

/// Two-link planar arm forward kinematics; the root recovers the joint angles
/// that place the tip at the target `(px, py)`.
pub struct TwoLinkArm {
    pub l1: f64,
    pub l2: f64,
    pub px: f64,
    pub py: f64,
}

impl VectorFn<2, 2> for TwoLinkArm {
    fn eval<S: Numeric>(&self, v: &[S; 2]) -> [S; 2] {
        let l1 = S::from_f64(self.l1);
        let l2 = S::from_f64(self.l2);
        let px = S::from_f64(self.px);
        let py = S::from_f64(self.py);
        [
            l1 * v[0].cos() + l2 * (v[0] + v[1]).cos() - px,
            l1 * v[0].sin() + l2 * (v[0] + v[1]).sin() - py,
        ]
    }
}

/// Circle `x² + y² = 4` intersected with hyperbola `xy = 1`, as a 2×2 system.
pub struct CircleHyperbola;

impl VectorFn<2, 2> for CircleHyperbola {
    fn eval<S: Numeric>(&self, v: &[S; 2]) -> [S; 2] {
        [
            v[0] * v[0] + v[1] * v[1] - S::from_f64(4.0),
            v[0] * v[1] - S::ONE,
        ]
    }
}

/// Chemical equilibrium mass balance, a 3×3 system:
/// `[x + y + z - 1, y - 1.25·x², z - 5·x·y]`.
pub struct Equilibrium;

impl VectorFn<3, 3> for Equilibrium {
    fn eval<S: Numeric>(&self, v: &[S; 3]) -> [S; 3] {
        [
            v[0] + v[1] + v[2] - S::ONE,
            v[1] - S::from_f64(1.25) * v[0] * v[0],
            v[2] - S::from_f64(5.0) * v[0] * v[1],
        ]
    }
}

#[cfg(test)]
mod tests {
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
