//! Named-problem registry.
//!
//! Quadrature integrands and least-squares residuals are functions, so they
//! cannot live in a JSON fixture. Instead a fixture names a problem by a stable
//! string key, and both this module and the Python generator implement that key
//! with the identical formula. Adding a problem means adding it on both sides
//! under the same key.

use multicalc::scalar::{Numeric, ScalarFn, ScalarFnN, VectorFn};

/// Returns the integrand for a quadrature key, at whichever scalar the caller
/// asks for. Panics on an unknown key.
///
/// Gauss-Hermite folds an `e^{-x^2}` weight and Gauss-Laguerre an `e^{-x}` weight
/// around this integrand; Legendre and the iterative rules integrate it directly.
#[must_use]
pub fn integrand<S: Numeric>(key: &str) -> fn(S) -> S {
    match key {
        "two_x" => |x| S::from_f64(2.0) * x,
        "quartic" => |x| S::from_f64(4.0) * x * x * x - S::from_f64(3.0) * x * x,
        "cube" => |x| x * x * x,
        "x_squared" => |x| x * x,
        "inv_1px2" => |x| S::ONE / (S::ONE + x * x),
        "exp_neg_sq" => |x| Numeric::exp(-(x * x)),
        other => unreachable!("unknown integrand key {other:?}"),
    }
}

/// Transcendental `g(x, y, z) = y·sin x + x·cos y + x·y·eᶻ`.
pub struct Transcendental;

impl ScalarFnN<3> for Transcendental {
    fn eval<S: Numeric>(&self, point: &[S; 3]) -> S {
        point[1] * point[0].sin() + point[0] * point[1].cos() + point[0] * point[1] * point[2].exp()
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
        for &component in x {
            cos_sum += component.cos();
        }
        core::array::from_fn(|i| {
            n - cos_sum + S::from_f64((i + 1) as f64) * (S::ONE - x[i].cos()) - x[i].sin()
        })
    }
}

// Circle-fit target: 40 points sampled exactly on the circle of center (2, -1),
// radius 3. The same formula is mirrored in the Python generator.
const CIRCLE_POINTS: usize = 40;

#[must_use]
fn circle_px(i: usize) -> f64 {
    let angle = core::f64::consts::TAU * i as f64 / CIRCLE_POINTS as f64;
    2.0 + 3.0 * angle.cos()
}

#[must_use]
fn circle_py(i: usize) -> f64 {
    let angle = core::f64::consts::TAU * i as f64 / CIRCLE_POINTS as f64;
    -1.0 + 3.0 * angle.sin()
}

/// Fit a circle `[cx, cy, r]` to 40 fixed points, minimizing the geometric
/// distance residual `sqrt((x-cx)^2 + (y-cy)^2) - r`. The recovered geometry is
/// center `(2, -1)`, radius `3`.
pub struct CircleFit;

impl VectorFn<3, CIRCLE_POINTS> for CircleFit {
    fn eval<S: Numeric>(&self, params: &[S; 3]) -> [S; CIRCLE_POINTS] {
        let (center_x, center_y, radius) = (params[0], params[1], params[2]);
        core::array::from_fn(|i| {
            let dx = S::from_f64(circle_px(i)) - center_x;
            let delta_y = S::from_f64(circle_py(i)) - center_y;
            (dx * dx + delta_y * delta_y).sqrt() - radius
        })
    }
}

// Gaussian-peaks target: two Gaussians [a, mu, sigma] sampled at 50 points.
const GAUSS_POINTS: usize = 50;
const GAUSS_TRUTH: [f64; 6] = [2.0, 3.0, 0.8, 1.5, 7.0, 1.2];

#[must_use]
fn gauss_t(i: usize) -> f64 {
    i as f64 * 10.0 / (GAUSS_POINTS as f64 - 1.0)
}

#[must_use]
fn gauss_y(i: usize) -> f64 {
    let sample = gauss_t(i);
    let mut y = 0.0;
    for k in 0..2 {
        let a = GAUSS_TRUTH[3 * k];
        let mean = GAUSS_TRUTH[3 * k + 1];
        let sigma = GAUSS_TRUTH[3 * k + 2];
        let z = (sample - mean) / sigma;
        y += a * Numeric::exp(-(z * z));
    }
    y
}

/// Fit two Gaussian peaks `[a, mu, sigma]` to a spectrum sampled at 50 points.
/// The residual is `model(p) - y`, with `y` the two-peak signal at the true
/// parameters `[2, 3, 0.8, 1.5, 7, 1.2]`.
pub struct GaussianPeaks;

impl VectorFn<6, GAUSS_POINTS> for GaussianPeaks {
    fn eval<S: Numeric>(&self, params: &[S; 6]) -> [S; GAUSS_POINTS] {
        core::array::from_fn(|i| {
            let sample = S::from_f64(gauss_t(i));
            let mut model = S::ZERO;
            for k in 0..2 {
                let a = params[3 * k];
                let mean = params[3 * k + 1];
                let sigma = params[3 * k + 2];
                let z = (sample - mean) / sigma;
                model += a * (-(z * z)).exp();
            }
            model - S::from_f64(gauss_y(i))
        })
    }
}

/// Hessian target `f(x, y, z) = y·sin x + 2·x·eʸ + z²`.
pub struct HessianTarget;

impl ScalarFnN<3> for HessianTarget {
    fn eval<S: Numeric>(&self, point: &[S; 3]) -> S {
        point[1] * point[0].sin()
            + S::from_f64(2.0) * point[0] * point[1].exp()
            + point[2] * point[2]
    }
}

/// Jacobian target `[x·y·z, x² + y²]`, 3 inputs and 2 outputs.
pub struct Jac23;

impl VectorFn<3, 2> for Jac23 {
    fn eval<S: Numeric>(&self, point: &[S; 3]) -> [S; 2] {
        [
            point[0] * point[1] * point[2],
            point[0] * point[0] + point[1] * point[1],
        ]
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
    fn eval<S: Numeric>(&self, point: &[S; 3]) -> [S; 3] {
        [point[1], -point[0], S::from_f64(2.0) * point[2]]
    }
}

/// Approximation target `f(x, y, z) = x + y² + z³`.
pub struct ApproxTarget;

impl ScalarFnN<3> for ApproxTarget {
    fn eval<S: Numeric>(&self, point: &[S; 3]) -> S {
        point[0] + point[1] * point[1] + point[2] * point[2] * point[2]
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
    pub eccentricity: f64,
    pub mean_anomaly: f64,
}

impl ScalarFn for Kepler {
    fn eval<S: Numeric>(&self, eccentric_anomaly: S) -> S {
        eccentric_anomaly
            - S::from_f64(self.eccentricity) * eccentric_anomaly.sin()
            - S::from_f64(self.mean_anomaly)
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
        let reynolds = S::from_f64(self.reynolds);
        let eps = S::from_f64(self.rel_roughness);
        let root_f = f.sqrt();
        let inner = eps / S::from_f64(3.7) + S::from_f64(2.51) / (reynolds * root_f);
        let log10 = inner.log() / S::from_f64(10.0).log();
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
    pub first_link: f64,
    pub second_link: f64,
    pub target_x: f64,
    pub target_y: f64,
}

impl VectorFn<2, 2> for TwoLinkArm {
    fn eval<S: Numeric>(&self, angles: &[S; 2]) -> [S; 2] {
        let first_link = S::from_f64(self.first_link);
        let second_link = S::from_f64(self.second_link);
        let target_x = S::from_f64(self.target_x);
        let target_y = S::from_f64(self.target_y);
        [
            first_link * angles[0].cos() + second_link * (angles[0] + angles[1]).cos() - target_x,
            first_link * angles[0].sin() + second_link * (angles[0] + angles[1]).sin() - target_y,
        ]
    }
}

/// Circle `x² + y² = 4` intersected with hyperbola `xy = 1`, as a 2×2 system.
pub struct CircleHyperbola;

impl VectorFn<2, 2> for CircleHyperbola {
    fn eval<S: Numeric>(&self, point: &[S; 2]) -> [S; 2] {
        [
            point[0] * point[0] + point[1] * point[1] - S::from_f64(4.0),
            point[0] * point[1] - S::ONE,
        ]
    }
}

/// Chemical equilibrium mass balance, a 3×3 system:
/// `[x + y + z - 1, y - 1.25·x², z - 5·x·y]`.
pub struct Equilibrium;

impl VectorFn<3, 3> for Equilibrium {
    fn eval<S: Numeric>(&self, point: &[S; 3]) -> [S; 3] {
        [
            point[0] + point[1] + point[2] - S::ONE,
            point[1] - S::from_f64(1.25) * point[0] * point[0],
            point[2] - S::from_f64(5.0) * point[0] * point[1],
        ]
    }
}

/// The turning-arc process model, taken straight from the crate so the fixtures check what ships.
/// Mirrored by the model in `tools/qa/gen/generators/estimation.py`; the two must stay in step.
pub use multicalc::estimation::ConstantTurnAndSpeed;

/// A position fix: the sensor sees the first two state components.
pub struct GlobalPosition;
impl VectorFn<5, 2> for GlobalPosition {
    fn eval<S: Numeric>(&self, state: &[S; 5]) -> [S; 2] {
        [state[0], state[1]]
    }
}

/// A pose `[x, y, heading]` that does not move.
pub struct StationaryPose;
impl VectorFn<3, 3> for StationaryPose {
    fn eval<S: Numeric>(&self, pose: &[S; 3]) -> [S; 3] {
        [pose[0], pose[1], pose[2]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integrands_evaluate() {
        assert_eq!(integrand::<f64>("two_x")(3.0), 6.0);
        assert_eq!(integrand::<f64>("x_squared")(4.0), 16.0);
        assert_eq!(integrand::<f32>("cube")(2.0), 8.0);
    }

    #[test]
    fn residuals_vanish_at_the_solution() {
        // Each problem is a zero-residual fit at its true parameters.
        let rosenbrock = Rosenbrock.eval(&[1.0, 1.0]);
        assert!(
            rosenbrock
                .iter()
                .all(|residual: &f64| residual.abs() < 1e-12)
        );

        let circle = CircleFit.eval(&[2.0, -1.0, 3.0]);
        assert!(circle.iter().all(|residual: &f64| residual.abs() < 1e-12));

        let peaks = GaussianPeaks.eval(&GAUSS_TRUTH);
        assert!(peaks.iter().all(|residual: &f64| residual.abs() < 1e-12));
    }
}
