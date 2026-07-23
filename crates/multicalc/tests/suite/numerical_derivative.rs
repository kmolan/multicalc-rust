#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::error::DiffError;
use multicalc::numerical_derivative::autodiff::{AutoDiffMulti, AutoDiffSingle};
use multicalc::numerical_derivative::derivator::*;
use multicalc::numerical_derivative::finite_difference::*;
use multicalc::numerical_derivative::hessian::Hessian;
use multicalc::numerical_derivative::jacobian::Jacobian;
use multicalc::numerical_derivative::mode::*;
use multicalc::scalar::{Numeric, ScalarFn, ScalarFnN, VectorFn, c};
use multicalc::{scalar_fn, scalar_fn_vec};
use multicalc_testkit::problems::G;
use proptest::prelude::*;
use proptest::test_runner::{RngAlgorithm, TestRng, TestRunner};
use std::cell::Cell;

// ----- autodiff (the default backend): exact derivatives -----

#[test]
fn ad_single_derivative() {
    // f(x) = x^3 -> f' = 3x^2, f'' = 6x, f''' = 6
    let func = scalar_fn!(|x| x * x * x);
    let d = AutoDiffSingle::default();

    assert!(f64::abs(d.get(1, &func, 2.0).unwrap() - 12.0) < 1e-12);
    assert!(f64::abs(d.get(2, &func, 2.0).unwrap() - 12.0) < 1e-12);
    assert!(f64::abs(d.get(3, &func, 2.0).unwrap() - 6.0) < 1e-12);
}

#[test]
fn ad_first_partials() {
    // f(x, y) = 3x^2 + 2xy -> df/dx = 6x + 2y, df/dy = 2x
    let func = scalar_fn!(|v: &[f64; 2]| c(3.0) * v[0] * v[0] + c(2.0) * v[0] * v[1]);
    let d = AutoDiffMulti::default();
    let point = [1.0, 3.0];

    assert!(f64::abs(d.get_single_partial(&func, 0, &point).unwrap() - 12.0) < 1e-12);
    assert!(f64::abs(d.get_single_partial(&func, 1, &point).unwrap() - 2.0) < 1e-12);
}

#[test]
fn ad_first_partials_transcendental() {
    // f(x, y, z) = y*sin(x) + x*cos(y) + x*y*e^z
    let func = G;
    let d = AutoDiffMulti::default();
    let point = [1.0, 2.0, 3.0];

    // df/dx = y*cos(x) + cos(y) + y*e^z
    let dx = 2.0 * f64::cos(1.0) + f64::cos(2.0) + 2.0 * f64::exp(3.0);
    assert!(f64::abs(d.get_single_partial(&func, 0, &point).unwrap() - dx) < 1e-12);

    // df/dz = x*y*e^z
    let dz = 1.0 * 2.0 * f64::exp(3.0);
    assert!(f64::abs(d.get_single_partial(&func, 2, &point).unwrap() - dz) < 1e-12);
}

#[test]
fn ad_second_partials() {
    // f(x, y, z) = y*sin(x) + x*cos(y) + x*y*e^z
    let func = G;
    let d = AutoDiffMulti::default();
    let point = [1.0, 2.0, 3.0];

    // d2f/dx2 = -y*sin(x)
    let dxx = -2.0 * f64::sin(1.0);
    assert!(f64::abs(d.get_double_partial(&func, &[0, 0], &point).unwrap() - dxx) < 1e-12);

    // mixed d2f/dx dy = cos(x) - sin(y) + e^z
    let dxy = f64::cos(1.0) - f64::sin(2.0) + f64::exp(3.0);
    assert!(f64::abs(d.get_double_partial(&func, &[0, 1], &point).unwrap() - dxy) < 1e-12);
}

#[test]
fn ad_third_partials() {
    // f = x^3 y^3 z^3:  d3/dx dy dz = 27 x^2 y^2 z^2 = 972;  d3/dx2 dy = 18 x y^2 z^3 = 1944
    let func = scalar_fn!(|v: &[f64; 3]| v[0].powi(3) * v[1].powi(3) * v[2].powi(3));
    let d = AutoDiffMulti::default();
    let point = [1.0, 2.0, 3.0];

    assert!(f64::abs(d.get(&func, &[0, 1, 2], &point).unwrap() - 972.0) < 1e-9);
    assert!(f64::abs(d.get(&func, &[0, 0, 1], &point).unwrap() - 1944.0) < 1e-9);
}

#[test]
fn ad_jacobian() {
    // (x*y*z, x^2 + y^2)
    let f = scalar_fn_vec!(|v: &[f64; 3]| [v[0] * v[1] * v[2], v[0] * v[0] + v[1] * v[1]]);
    let jacobian: Jacobian = Jacobian::default();
    let result = jacobian.get(&f, &[1.0, 2.0, 3.0]).unwrap();

    let expected = [[6.0, 3.0, 2.0], [2.0, 4.0, 0.0]];
    for (i, row) in expected.iter().enumerate() {
        for (j, &want) in row.iter().enumerate() {
            assert!(f64::abs(result.get(i, j).copied().unwrap() - want) < 1e-12);
        }
    }
}

#[test]
#[cfg(feature = "alloc")]
fn ad_jacobian_on_heap() {
    let f = scalar_fn_vec!(|v: &[f64; 3]| [v[0] * v[1] * v[2], v[0] * v[0] + v[1] * v[1]]);
    let jacobian: Jacobian = Jacobian::default();
    let result = jacobian.get_on_heap(&f, &[1.0, 2.0, 3.0]).unwrap();

    let expected = [[6.0, 3.0, 2.0], [2.0, 4.0, 0.0]];
    for (i, row) in expected.iter().enumerate() {
        for (j, &want) in row.iter().enumerate() {
            assert!(f64::abs(result[i][j] - want) < 1e-12);
        }
    }
}

#[test]
fn ad_hessian() {
    // f(x, y) = y*sin(x) + 2*x*e^y
    let func = scalar_fn!(|v: &[f64; 2]| v[1] * v[0].sin() + c(2.0) * v[0] * v[1].exp());
    let hessian: Hessian = Hessian::default();
    let result = hessian.get(&func, &[1.0, 2.0]).unwrap();

    let expected = [
        [-2.0 * f64::sin(1.0), f64::cos(1.0) + 2.0 * f64::exp(2.0)],
        [f64::cos(1.0) + 2.0 * f64::exp(2.0), 2.0 * f64::exp(2.0)],
    ];
    for (i, row) in expected.iter().enumerate() {
        for (j, &want) in row.iter().enumerate() {
            assert!(f64::abs(result.get(i, j).copied().unwrap() - want) < 1e-12);
        }
    }
}

#[test]
fn ad_f32() {
    // x*x at 0.5; first derivative 2x = 1.0, exact under autodiff
    let func = scalar_fn!(|x| x * x);
    let d = AutoDiffSingle::<f32>::default();
    assert!(f32::abs(d.get(1, &func, 0.5_f32).unwrap() - 1.0) < 1e-6);
}

// ----- autodiff error handling -----

#[test]
fn ad_error_index_out_of_range() {
    let func = scalar_fn!(|v: &[f64; 3]| v[0] + v[1] + v[2]);
    let d = AutoDiffMulti::default();
    let result = d.get_single_partial(&func, 5, &[1.0, 2.0, 3.0]);
    assert_eq!(result.unwrap_err(), DiffError::IndexOutOfRange);
}

#[test]
fn ad_error_order_zero() {
    let func = scalar_fn!(|v: &[f64; 3]| v[0] + v[1] + v[2]);
    let d = AutoDiffMulti::default();
    let idx: [usize; 0] = [];
    assert_eq!(
        d.get(&func, &idx, &[1.0, 2.0, 3.0]).unwrap_err(),
        DiffError::OrderZero
    );
}

#[test]
fn ad_error_order_unsupported() {
    // autodiff multi caps at third order; a fourth-order partial is rejected
    let func = scalar_fn!(|v: &[f64; 3]| v[0] + v[1] + v[2]);
    let d = AutoDiffMulti::default();
    assert_eq!(
        d.get(&func, &[0, 1, 2, 0], &[1.0, 2.0, 3.0]).unwrap_err(),
        DiffError::OrderUnsupported
    );
}

#[test]
fn ad_jacobian_empty_error() {
    // a function with no outputs is an empty function set
    struct EmptyVectorFn;
    impl VectorFn<3, 0> for EmptyVectorFn {
        fn eval<S: Numeric>(&self, _point: &[S; 3]) -> [S; 0] {
            []
        }
    }

    let jacobian: Jacobian = Jacobian::default();
    let result = jacobian.get(&EmptyVectorFn, &[1.0, 2.0, 3.0]);
    assert_eq!(result.unwrap_err(), DiffError::EmptyFunctionSet);
}

// ----- column-seeded Jacobian -----

// A VectorFn that counts how many times it is evaluated, to prove the column-seeded harness
// runs one pass per input column (N) rather than one per matrix cell (M*N).
struct CountingVectorFn {
    calls: Cell<usize>,
}

impl VectorFn<3, 2> for CountingVectorFn {
    fn eval<S: Numeric>(&self, p: &[S; 3]) -> [S; 2] {
        self.calls.set(self.calls.get() + 1);
        // (x*y*z, x^2 + y^2)
        [p[0] * p[1] * p[2], p[0] * p[0] + p[1] * p[1]]
    }
}

#[test]
fn ad_jacobian_is_column_seeded() {
    let f = CountingVectorFn {
        calls: Cell::new(0),
    };
    let jacobian: Jacobian = Jacobian::default();
    let result = jacobian.get(&f, &[1.0, 2.0, 3.0]).unwrap();

    // values are unchanged from the old harness
    let expected = [[6.0, 3.0, 2.0], [2.0, 4.0, 0.0]];
    for (i, row) in expected.iter().enumerate() {
        for (j, &want) in row.iter().enumerate() {
            assert!(f64::abs(result.get(i, j).copied().unwrap() - want) < 1e-12);
        }
    }

    // one evaluation per input column (3), not per cell (2*3 = 6)
    assert_eq!(f.calls.get(), 3);
}

#[test]
fn ad_jacobian_column_reads_all_outputs() {
    // one seeded pass on input 0 gives d/dx of both outputs: [y*z, 2x] = [6, 2]
    let f = scalar_fn_vec!(|v: &[f64; 3]| [v[0] * v[1] * v[2], v[0] * v[0] + v[1] * v[1]]);
    let d = AutoDiffMulti::default();
    let column = d.jacobian_column(&f, 0, &[1.0, 2.0, 3.0]).unwrap();
    assert!(f64::abs(column[0] - 6.0) < 1e-12);
    assert!(f64::abs(column[1] - 2.0) < 1e-12);
}

#[test]
fn ad_jacobian_column_index_out_of_range() {
    let f = scalar_fn_vec!(|v: &[f64; 3]| [v[0] * v[1] * v[2], v[0] * v[0] + v[1] * v[1]]);
    let d = AutoDiffMulti::default();
    let result = d.jacobian_column(&f, 5, &[1.0, 2.0, 3.0]);
    assert_eq!(result.unwrap_err(), DiffError::IndexOutOfRange);
}

#[test]
fn fd_jacobian_column_matches() {
    // the finite-difference implementation produces the right matrix, matching the analytic
    // values to finite-difference tolerance (unchanged from the per-Component path it replaces)
    let f = scalar_fn_vec!(|v: &[f64; 3]| [v[0] * v[1] * v[2], v[0] * v[0] + v[1] * v[1]]);
    let jacobian = Jacobian::from_derivator(FiniteDifferenceMulti::default());
    let result = jacobian.get(&f, &[1.0, 2.0, 3.0]).unwrap();

    let expected = [[6.0, 3.0, 2.0], [2.0, 4.0, 0.0]];
    for (i, row) in expected.iter().enumerate() {
        for (j, &want) in row.iter().enumerate() {
            assert!(f64::abs(result.get(i, j).copied().unwrap() - want) < 1e-5);
        }
    }
}

#[test]
fn fd_jacobian_is_column_seeded() {
    // central difference evaluates the full function twice per input column (2*3 = 6), not twice
    // per matrix cell (2*M*N = 12)
    let f = CountingVectorFn {
        calls: Cell::new(0),
    };
    let jacobian = Jacobian::from_derivator(FiniteDifferenceMulti::default());
    let _ = jacobian.get(&f, &[1.0, 2.0, 3.0]).unwrap();
    assert_eq!(f.calls.get(), 6);
}

// ----- finite differences: kept as a sparse fallback for the engine and the cases autodiff
//       does not cover (high-order mixed partials, zero-step-size error) -----

#[test]
fn fd_single_derivative_modes() {
    // x^2/2, derivative x; check all three finite-difference modes still work
    let func = scalar_fn!(|x| c(0.5) * x * x);
    for mode in [
        FiniteDifferenceMode::Forward,
        FiniteDifferenceMode::Backward,
        FiniteDifferenceMode::Central,
    ] {
        let mut d = FiniteDifferenceSingle::default();
        d.config.method = mode;
        assert!(f64::abs(d.get(1, &func, 2.0).unwrap() - 2.0) < 0.001);
    }
}

#[test]
fn fd_step_size_zero_error() {
    let func = scalar_fn!(|v: &[f64; 3]| v[1] * v[0].sin());
    let d = FiniteDifferenceMulti::from_parameters(0.0, FiniteDifferenceMode::Central, 1.0);
    assert_eq!(
        d.get(&func, &[0], &[1.0, 2.0, 3.0]).unwrap_err(),
        DiffError::StepSizeZero
    );
}

fn eval_poly<S: Numeric>(coeffs: &[f64], x: S) -> S {
    let mut acc = S::from_f64(0.0);
    let mut x_pow = S::from_f64(1.0);
    for &a in coeffs {
        acc += S::from_f64(a) * x_pow;
        x_pow *= x;
    }
    acc
}

struct PolyComp {
    inner: Vec<f64>,
    outer: Vec<f64>,
}

impl ScalarFn for PolyComp {
    fn eval<S: Numeric>(&self, x: S) -> S {
        eval_poly(&self.outer, eval_poly(&self.inner, x))
    }
}

struct BivariatePoly {
    coeffs: Vec<f64>,
}

impl ScalarFnN<2> for BivariatePoly {
    fn eval<S: Numeric>(&self, p: &[S; 2]) -> S {
        let (x, y) = (p[0], p[1]);
        let c = |i| S::from_f64(self.coeffs[i]);
        c(0) + c(1) * x + c(2) * y + c(3) * x * x + c(4) * x * y + c(5) * y * y
    }
}

fn coeff_l1(coeffs: &[f64]) -> f64 {
    coeffs.iter().map(|c| c.abs()).sum()
}

fn ad_fd_tol(ad: f64, h: f64, order: i32, c: f64, coeff_scale: f64) -> f64 {
    let scale = ad.abs().max(1.0) * coeff_scale.max(1.0);
    c * h.powi(order) * scale
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn proptest_ad_fd_single_first(
        inner in prop::collection::vec(-5.0f64..5.0, 3..=5),
        outer in prop::collection::vec(-5.0f64..5.0, 3..=5),
        x in -2.0f64..2.0,
    ) {
        let scale = 1.0 + coeff_l1(&inner) + coeff_l1(&outer);
        let f = PolyComp { inner, outer };
        let h = DEFAULT_STEP_SIZE;
        let ad = AutoDiffSingle::default().get(1, &f, x).unwrap();
        let fd = FiniteDifferenceSingle::default();
        let fd_val = fd.get(1, &f, x).unwrap();
        let tol = ad_fd_tol(ad, h, 2, 1e3, scale);
        prop_assert!(
            (fd_val - ad).abs() < tol,
            "fd={fd_val} ad={ad} tol={tol} x={x}"
        );
    }

    #[test]
    fn proptest_ad_fd_multi_first_partial(
        coeffs in prop::collection::vec(-5.0f64..5.0, 6),
        x in -2.0f64..2.0,
        y in -2.0f64..2.0,
    ) {
        let scale = 1.0 + coeff_l1(&coeffs);
        let f = BivariatePoly { coeffs };
        let point = [x, y];
        let h = DEFAULT_STEP_SIZE;
        let ad_d = AutoDiffMulti::default();
        let fd_d = FiniteDifferenceMulti::default();
        for idx in [0usize, 1] {
            let ad = ad_d.get_single_partial(&f, idx, &point).unwrap();
            let fd_val = fd_d.get_single_partial(&f, idx, &point).unwrap();
            let tol = ad_fd_tol(ad, h, 2, 1e3, scale);
            prop_assert!(
                (fd_val -ad).abs() < tol,
                "idx={idx} fd={fd_val} ad={ad} tol={tol}"
            );
        }
    }
}

// Nested FD is noisier than first-deriv; use a milder domain and larger step. A fixed RNG seed
// keeps the sampled cases identical across runs, so a pass or failure is reproducible rather than
// dependent on the run's entropy.
#[test]
fn proptest_ad_fd_single_second() {
    let strategy = (
        prop::collection::vec(-2.0f64..2.0, 3..=4),
        prop::collection::vec(-2.0f64..2.0, 3..=4),
        -2.0f64..2.0,
    );
    let mut runner = TestRunner::new_with_rng(
        ProptestConfig::with_cases(256),
        TestRng::deterministic_rng(RngAlgorithm::default()),
    );
    runner
        .run(&strategy, |(inner, outer, x)| {
            let scale = 1.0 + coeff_l1(&inner) + coeff_l1(&outer);
            let f = PolyComp { inner, outer };
            let h = 1e-4;
            let ad = AutoDiffSingle::default().get(2, &f, x).unwrap();
            let fd = FiniteDifferenceSingle::from_parameters(
                h,
                FiniteDifferenceMode::Central,
                DEFAULT_STEP_SIZE_MULTIPLIER,
            );
            let fd_val = fd.get(2, &f, x).unwrap();
            let tol = ad_fd_tol(ad, h, 2, 1e4, scale);
            prop_assert!((fd_val - ad).abs() < tol, "fd={fd_val} ad={ad} tol={tol}");
            Ok(())
        })
        .unwrap();
}
