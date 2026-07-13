//! Tiny on-target math checks. Each asserts a known answer to a tolerance.
//!
//! Golden checks assert against values taken from the host oracle fixtures (see
//! `fixtures.rs`), so the target and the host share one source of truth. Identity
//! checks assert a mathematical identity that needs no fixture. Every assertion is
//! a hard failure: a wrong answer panics, which the runner turns into a non-zero
//! QEMU exit.

use multicalc::LevenbergMarquardt;
use multicalc::error::LinalgError;
use multicalc::linear_algebra::Matrix;
use multicalc::numerical_derivative::autodiff::{AutoDiffMulti, AutoDiffSingle};
use multicalc::numerical_derivative::derivator::DerivatorSingleVariable;
use multicalc::scalar::{Numeric, VectorFn};
use multicalc::scalar_fn;

use crate::fixtures;

/// Combined absolute/relative closeness, matching `close` in
/// tools/oracle/src/load.rs: `|got - want| <= abs + rel * max(|got|, |want|)`.
fn close(got: f64, want: f64, abs: f64, rel: f64) -> bool {
    (got - want).abs() <= abs + rel * got.abs().max(want.abs())
}

/// Golden: the Rosenbrock least-squares minimizer must match the host oracle
/// golden (optimization/rosenbrock). Residuals `[10 (y - x^2), 1 - x]` are zero
/// at the minimum `(1, 1)`. Part of the full set only (thumbv7em).
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn lm_fit() {
    struct Rosenbrock;
    impl VectorFn<2, 2> for Rosenbrock {
        fn eval<S: Numeric>(&self, p: &[S; 2]) -> [S; 2] {
            let (x, y) = (p[0], p[1]);
            [S::from_f64(10.0) * (y - x * x), S::from_f64(1.0) - x]
        }
    }
    let solver = LevenbergMarquardt::<AutoDiffMulti>::default().with_patience(100);
    let report = solver
        .minimize(&Rosenbrock, &fixtures::ROSENBROCK_X0)
        .expect("fit converges");
    for i in 0..2 {
        assert!(close(
            report.solution[i],
            fixtures::ROSENBROCK_SOLUTION[i],
            fixtures::ROSENBROCK_ABS,
            fixtures::ROSENBROCK_REL,
        ));
    }
}

/// Identity: differentiate x^3 at x = 2 by autodiff. Exact derivative is 12.
/// Part of the full set only (thumbv7em).
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn autodiff_derivative() {
    let f = scalar_fn!(|x| x * x * x);
    let d = AutoDiffSingle::default();
    let value = d.get(1, &f, 2.0_f64).expect("derivative");
    assert!((value - 12.0).abs() < 1e-12);
}

/// Identity: a sum over the portable (no-atomics) path, for the Cortex-M0 target.
/// Σ 1..=4 = 10.
pub fn portable_path() {
    let v = [1.0_f64, 2.0, 3.0, 4.0];
    let sum: f64 = v.iter().copied().fold(0.0, |a, b| a + b);
    assert!((sum - 10.0).abs() < 1e-12);
}

/// No-panic negative path: a fallible decomposition returns a typed `Err` on bad
/// input instead of crashing. A singular (all-zero) matrix has no LU
/// factorization; an indefinite matrix has no Cholesky factorization.
pub fn error_path_returns_err() {
    assert!(matches!(
        Matrix::<3, 3>::zeros().lu(),
        Err(LinalgError::Singular)
    ));
    let indefinite = Matrix::<2, 2>::new([[1.0, 2.0], [2.0, 1.0]]);
    assert!(matches!(
        indefinite.cholesky(),
        Err(LinalgError::NotPositiveDefinite)
    ));
}

/// Golden: singular values of a fixture matrix must match the host oracle golden
/// (linalg/svd_3x3). Returns the values so the caller can emit them for the
/// cross-ABI divergence guard.
pub fn svd_golden() -> [f64; 3] {
    let a: Matrix<3, 3> = Matrix::new(fixtures::SVD_3X3_INPUT);
    let sv = a.svd().expect("svd").singular_values();
    for i in 0..3 {
        assert!(close(
            sv[i],
            fixtures::SVD_3X3_SINGULAR_VALUES[i],
            fixtures::SVD_3X3_ABS,
            fixtures::SVD_3X3_REL,
        ));
    }
    [sv[0], sv[1], sv[2]]
}

/// Identity: SO(3)/SE(3) exp/log round trips and one known rotation. No fixture — the answers are
/// exact or self-inverse. Part of the full set only (thumbv7em).
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn lie_group_identity() {
    use multicalc::linear_algebra::Vector;
    use multicalc::spatial::{SE3, SO3};

    // A 90° rotation about z maps x -> y.
    let rz = SO3::<f64>::exp(Vector::new([0.0, 0.0, core::f64::consts::FRAC_PI_2]));
    let p = rz.act(Vector::new([1.0, 0.0, 0.0]));
    assert!(p[0].abs() < 1e-12);
    assert!((p[1] - 1.0).abs() < 1e-12);
    assert!(p[2].abs() < 1e-12);

    // SO(3) exp/log round trip.
    let phi = Vector::new([0.3, -0.6, 0.2]);
    let back = SO3::exp(phi).log();
    for i in 0..3 {
        assert!((back[i] - phi[i]).abs() < 1e-9);
    }

    // SE(3) exp/log round trip (exercises the left Jacobian and its inverse).
    let xi = Vector::new([0.5, -0.2, 0.1, 0.3, -0.6, 0.2]);
    let back6 = SE3::exp(xi).log();
    for i in 0..6 {
        assert!((back6[i] - xi[i]).abs() < 1e-9);
    }
}

/// Identity: the core ODE solvers hit known answers. RK4 integrates the harmonic
/// oscillator over one period back to its start; RK45 integrates y' = -y to e^{-1}.
/// Heavier than the canary checks, so it is part of the full set only (thumbv7em).
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn ode_identity() {
    use multicalc::linear_algebra::Vector;
    use multicalc::ode::{Rk4, Rk45};

    // RK4: harmonic oscillator y'' = -y over one period returns to its start.
    let f = |_t: f64, y: &Vector<2, f64>| Vector::new([y[1], -y[0]]);
    let steps = 2000;
    let dt = core::f64::consts::TAU / steps as f64;
    let yf = Rk4::integrate(&f, 0.0, &Vector::new([1.0, 0.0]), dt, steps, |_, _| {});
    assert!((yf[0] - 1.0).abs() < 1e-4);
    assert!(yf[1].abs() < 1e-4);

    // RK45: y' = -y to t = 1 matches e^{-1} (libm exp — no std float methods on target).
    let g = |_t: f64, y: &Vector<1, f64>| -*y;
    let e = Rk45::default()
        .solve(&g, 0.0, &Vector::new([1.0]), 1.0)
        .expect("rk45 solve");
    assert!((e[0] - multicalc::libm::exp(-1.0)).abs() < 1e-6);
}
