//! Tiny on-target math checks. Each asserts a known answer to a tolerance.
//!
//! Golden checks assert against values taken from the host QA fixtures (see
//! `fixtures.rs`), so the target and the host share one source of truth. Identity
//! checks assert a mathematical identity that needs no fixture. Every assertion is
//! a hard failure: a wrong answer panics, which the runner turns into a non-zero
//! QEMU exit.
//!
//! Every check passes its inputs and its result through `core::hint::black_box` so
//! the compiler cannot const-fold the work away — the point is to run instructions on
//! target, not to prove a constant at build time.

use core::hint::black_box;

use multicalc::LevenbergMarquardt;
use multicalc::error::LinalgError;
use multicalc::linear_algebra::Matrix;
use multicalc::numerical_derivative::autodiff::{AutoDiffMulti, AutoDiffSingle};
use multicalc::numerical_derivative::derivator::DerivatorSingleVariable;
use multicalc::numerical_derivative::jacobian::Jacobian;
use multicalc::numerical_integration::gaussian_integration::GaussianSingle;
use multicalc::numerical_integration::integrator::IntegratorSingleVariable;
use multicalc::numerical_integration::mode::GaussianQuadratureMethod;
use multicalc::root_finding::Newton;
use multicalc::scalar_fn;
use multicalc::vector_field::{curl, divergence};
use multicalc_testkit::problems::{Jac23, Rosenbrock, VField3d, Wien};

use crate::fixtures;

/// Assert `got` is within `abs + rel * max(|got|, |want|)` of `want`, printing both
/// values and the tolerance over semihosting before panicking on failure. Use this for
/// identity checks so a QEMU-only failure shows the diverging number, not just a line.
macro_rules! assert_close {
    ($name:expr, $got:expr, $want:expr, $abs:expr, $rel:expr) => {{
        let got: f64 = $got;
        let want: f64 = $want;
        let ok = (got - want).abs() <= $abs + $rel * got.abs().max(want.abs());
        if !ok {
            let _ = crate::hprintln!(
                "CHECK {} FAIL got={:e} want={:e} abs={:e} rel={:e}",
                $name,
                got,
                want,
                $abs as f64,
                $rel as f64
            );
        }
        assert!(ok, "{}", $name);
    }};
}

/// Golden: the Rosenbrock least-squares minimizer must match the host QA golden
/// (optimization/rosenbrock). Returns `solution[0]` for the cross-ABI guard. Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn lm_fit() -> f64 {
    let x0 = black_box(fixtures::ROSENBROCK_X0);
    let solver = LevenbergMarquardt::<AutoDiffMulti>::default().with_patience(100);
    let report = solver.minimize(&Rosenbrock, &x0).expect("fit converges");
    for i in 0..2 {
        assert_close!(
            "lm_fit",
            black_box(report.solution[i]),
            fixtures::ROSENBROCK_SOLUTION[i],
            fixtures::ROSENBROCK_ABS,
            fixtures::ROSENBROCK_REL
        );
    }
    black_box(report.solution[0])
}

/// Identity: differentiate x^3 at x = 2 by autodiff. Exact derivative is 12. Returns the
/// value for the cross-ABI guard. Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn autodiff_derivative() -> f64 {
    let f = scalar_fn!(|x| x * x * x);
    let d = AutoDiffSingle::default();
    let value = d.get(1, &f, black_box(2.0_f64)).expect("derivative");
    assert_close!("autodiff_derivative", black_box(value), 12.0, 1e-12, 0.0);
    black_box(value)
}

/// Identity in f32: differentiate x^3 at x = 2, exact derivative 12. f32 arithmetic is where
/// soft-float (eabi) and the hardware FPU (eabihf) diverge. Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn autodiff_derivative_f32() -> f32 {
    let f = scalar_fn!(|x| x * x * x);
    let d = AutoDiffSingle::default();
    let value: f32 = d.get(1, &f, black_box(2.0_f32)).expect("derivative f32");
    // f32 tolerance is looser than f64; assert in f32 space directly.
    let ok = (value - 12.0_f32).abs() <= 1e-4;
    if !ok {
        let _ = crate::hprintln!("CHECK autodiff_f32 FAIL got={:e}", value);
    }
    assert!(ok, "autodiff_derivative_f32");
    black_box(value)
}

/// Real portable-path library call for the Cortex-M0 canary: a vector dot product through
/// `multicalc`. `[1,2,3,4] · [4,3,2,1] = 20`. Exercises a no-atomics, no-alloc library symbol
/// (unlike a plain array fold, which touches no `multicalc` code).
pub fn portable_path() {
    use multicalc::linear_algebra::Vector;
    let a = black_box(Vector::new([1.0_f64, 2.0, 3.0, 4.0]));
    let b = black_box(Vector::new([4.0_f64, 3.0, 2.0, 1.0]));
    let dot = black_box(a.dot(b));
    assert_close!("portable_path", dot, 20.0, 1e-12, 0.0);
}

/// No-panic negative path: a fallible decomposition returns a typed `Err` on bad input
/// instead of crashing.
pub fn error_path_returns_err() {
    let singular = black_box(Matrix::<3, 3>::zeros());
    assert!(matches!(singular.lu(), Err(LinalgError::Singular)));
    let indefinite = black_box(Matrix::<2, 2>::new([[1.0, 2.0], [2.0, 1.0]]));
    assert!(matches!(
        indefinite.cholesky(),
        Err(LinalgError::NotPositiveDefinite)
    ));
}

/// Golden: singular values of a fixture matrix must match the host QA golden (linalg/svd_3x3).
/// Returns the values for the cross-ABI guard (emitted on every target).
pub fn svd_golden() -> [f64; 3] {
    let a: Matrix<3, 3> = black_box(Matrix::new(fixtures::SVD_3X3_INPUT));
    let sv = *a.svd().expect("svd").singular_values().as_array();
    for i in 0..3 {
        assert_close!(
            "svd_golden",
            black_box(sv[i]),
            fixtures::SVD_3X3_SINGULAR_VALUES[i],
            fixtures::SVD_3X3_ABS,
            fixtures::SVD_3X3_REL
        );
    }
    [black_box(sv[0]), black_box(sv[1]), black_box(sv[2])]
}

/// Identity: SO(3)/SE(3) exp/log round trips and one known rotation. Returns the first SE(3)
/// log component for the cross-ABI guard. Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn lie_group_identity() -> f64 {
    use multicalc::linear_algebra::Vector;
    use multicalc::spatial::{SE3, SO3};

    // A 90° rotation about z maps x -> y.
    let rz = SO3::<f64>::exp(black_box(Vector::new([
        0.0,
        0.0,
        core::f64::consts::FRAC_PI_2,
    ])));
    let p = *rz.act(black_box(Vector::new([1.0, 0.0, 0.0]))).as_array();
    assert_close!("lie_rot_x", black_box(p[0]), 0.0, 1e-12, 0.0);
    assert_close!("lie_rot_y", black_box(p[1]), 1.0, 1e-12, 0.0);
    assert_close!("lie_rot_z", black_box(p[2]), 0.0, 1e-12, 0.0);

    // SO(3) exp/log round trip.
    let phi = black_box(Vector::new([0.3, -0.6, 0.2]));
    let back = *SO3::exp(phi).log().as_array();
    let phi = *phi.as_array();
    for i in 0..3 {
        assert_close!("lie_so3", black_box(back[i]), phi[i], 1e-9, 0.0);
    }

    // SE(3) exp/log round trip (exercises the left Jacobian and its inverse).
    let xi = black_box(Vector::new([0.5, -0.2, 0.1, 0.3, -0.6, 0.2]));
    let back6 = *SE3::exp(xi).log().as_array();
    let xi = *xi.as_array();
    for i in 0..6 {
        assert_close!("lie_se3", black_box(back6[i]), xi[i], 1e-9, 0.0);
    }
    black_box(back6[0])
}

/// Identity: RK4 integrates the harmonic oscillator over one period back to its start; RK45
/// integrates y' = -y to e^{-1}. Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn ode_identity() {
    use multicalc::linear_algebra::Vector;
    use multicalc::ode::{Rk4, Rk45};

    let f = |_t: f64, y: &Vector<2, f64>| {
        let [y0, y1] = *y.as_array();
        Vector::new([y1, -y0])
    };
    let steps = 2000;
    let dt = core::f64::consts::TAU / steps as f64;
    let yf = Rk4::integrate(
        &f,
        0.0,
        &black_box(Vector::new([1.0, 0.0])),
        dt,
        steps,
        |_, _| {},
    );
    assert_close!("ode_rk4_x", black_box(yf.as_array()[0]), 1.0, 1e-4, 0.0);
    assert_close!("ode_rk4_v", black_box(yf.as_array()[1]), 0.0, 1e-4, 0.0);

    let g = |_t: f64, y: &Vector<1, f64>| -*y;
    let e = Rk45::default()
        .solve(&g, 0.0, &black_box(Vector::new([1.0])), 1.0)
        .expect("rk45 solve");
    assert_close!(
        "ode_rk45",
        black_box(e.as_array()[0]),
        multicalc::libm::exp(-1.0),
        1e-6,
        0.0
    );
}

/// Identity: Gauss-Legendre order 4 integrates `2x` on `[0, 2]` to `4`. Returns the value for
/// the cross-ABI guard. Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn quadrature_identity() -> f64 {
    let f = |x: f64| 2.0 * x;
    let quad = GaussianSingle::<f64>::from_parameters(4, GaussianQuadratureMethod::GaussLegendre);
    let value = quad
        .get_single(&f, &black_box([0.0, 2.0]))
        .expect("quadrature");
    assert_close!("quadrature", black_box(value), 4.0, 1e-12, 0.0);
    black_box(value)
}

/// Identity: the Jacobian of `[x*y*z, x^2 + y^2]` at `(1, 2, 3)` is `[[6,3,2],[2,4,0]]`. Returns
/// the `(0, 0)` entry for the cross-ABI guard. Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn jacobian_identity() -> f64 {
    let j = Jacobian::<AutoDiffMulti>::default()
        .get(&Jac23, &black_box([1.0, 2.0, 3.0]))
        .expect("jacobian");
    let expected = [[6.0, 3.0, 2.0], [2.0, 4.0, 0.0]];
    let j = *j.as_slice_rows();
    for r in 0..2 {
        for c in 0..3 {
            assert_close!("jacobian", black_box(j[r][c]), expected[r][c], 1e-12, 0.0);
        }
    }
    black_box(j[0][0])
}

/// Identity: the field `[y, -x, 2z]` at `(1, 2, 3)` has curl `[0, 0, -2]` and divergence `2`.
/// Returns the divergence for the cross-ABI guard. Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn vector_field_identity() -> f64 {
    let point = black_box([1.0, 2.0, 3.0]);
    let c = curl::get_3d(AutoDiffMulti::default(), &VField3d, &point).expect("curl");
    let expected_curl = [0.0, 0.0, -2.0];
    for i in 0..3 {
        assert_close!("vfield_curl", black_box(c[i]), expected_curl[i], 1e-12, 0.0);
    }
    let d = divergence::get_3d(AutoDiffMulti::default(), &VField3d, &point).expect("divergence");
    assert_close!("vfield_div", black_box(d), 2.0, 1e-12, 0.0);
    black_box(d)
}

/// Golden: Newton on Wien's displacement equation must match the host QA golden root
/// (root_finding/wien_newton). Returns the root for the cross-ABI guard. Full set only.
#[cfg_attr(not(feature = "full-smoke"), allow(dead_code))]
pub fn root_finding_golden() -> f64 {
    let report = Newton::<AutoDiffSingle>::default()
        .solve(&Wien, black_box(fixtures::ROOT_WIEN_X0))
        .expect("newton solve");
    assert_close!(
        "root_finding",
        black_box(report.root),
        fixtures::ROOT_WIEN_ROOT,
        fixtures::ROOT_WIEN_ABS,
        fixtures::ROOT_WIEN_REL
    );
    black_box(report.root)
}
