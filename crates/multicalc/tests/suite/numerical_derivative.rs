#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::error::DiffError;
use multicalc::numerical_derivative::*;
use multicalc::scalar::{Numeric, ScalarFn, ScalarFnN, VectorFn, constant};
use multicalc::{scalar_fn, scalar_fn_vec};
use multicalc_testkit::problems::Transcendental;
use proptest::prelude::*;
use proptest::test_runner::{RngAlgorithm, TestRng, TestRunner};
use std::cell::Cell;

// ----- autodiff (the default backend): exact derivatives -----

#[test]
fn ad_single_derivative() {
    // f(x) = x^3 -> f' = 3x^2, f'' = 6x, f''' = 6
    let function = scalar_fn!(|x| x * x * x);
    let derivator = AutoDiffSingle::default();

    assert!(f64::abs(derivator.differentiate(1, &function, 2.0).unwrap() - 12.0) < 1e-12);
    assert!(f64::abs(derivator.differentiate(2, &function, 2.0).unwrap() - 12.0) < 1e-12);
    assert!(f64::abs(derivator.differentiate(3, &function, 2.0).unwrap() - 6.0) < 1e-12);
}

#[test]
fn ad_first_partials() {
    // f(x, y) = 3x^2 + 2xy -> df/dx = 6x + 2y, df/dy = 2x
    let function = scalar_fn!(|point: &[f64; 2]| constant(3.0) * point[0] * point[0]
        + constant(2.0) * point[0] * point[1]);
    let derivator = AutoDiffMulti::default();
    let point = [1.0, 3.0];

    let x = 0;
    let partial_x = derivator
        .first_partial_derivative(&function, x, &point)
        .unwrap();
    assert!(f64::abs(partial_x - 12.0) < 1e-12);

    let y = 1;
    let partial_y = derivator
        .first_partial_derivative(&function, y, &point)
        .unwrap();
    assert!(f64::abs(partial_y - 2.0) < 1e-12);
}

#[test]
fn ad_first_partials_transcendental() {
    // f(x, y, z) = y*sin(x) + x*cos(y) + x*y*e^z
    let function = Transcendental;
    let derivator = AutoDiffMulti::default();
    let point = [1.0, 2.0, 3.0];

    // df/dx = y*cos(x) + cos(y) + y*e^z
    let x = 0;
    let expected_dx = 2.0 * f64::cos(1.0) + f64::cos(2.0) + 2.0 * f64::exp(3.0);
    let partial_x = derivator
        .first_partial_derivative(&function, x, &point)
        .unwrap();
    assert!(f64::abs(partial_x - expected_dx) < 1e-12);

    // df/dz = x*y*e^z
    let z = 2;
    let expected_dz = 1.0 * 2.0 * f64::exp(3.0);
    let partial_z = derivator
        .first_partial_derivative(&function, z, &point)
        .unwrap();
    assert!(f64::abs(partial_z - expected_dz) < 1e-12);
}

#[test]
fn ad_second_partials() {
    // f(x, y, z) = y*sin(x) + x*cos(y) + x*y*e^z
    let function = Transcendental;
    let derivator = AutoDiffMulti::default();
    let point = [1.0, 2.0, 3.0];

    let x = 0;
    let y = 1;

    // d2f/dx2 = -y*sin(x)
    let expected_dxx = -2.0 * f64::sin(1.0);
    let partial_xx = derivator
        .second_partial_derivative(&function, &[x, x], &point)
        .unwrap();
    assert!(f64::abs(partial_xx - expected_dxx) < 1e-12);

    // mixed d2f/dx dy = cos(x) - sin(y) + e^z
    let expected_dxy = f64::cos(1.0) - f64::sin(2.0) + f64::exp(3.0);
    let partial_xy = derivator
        .second_partial_derivative(&function, &[x, y], &point)
        .unwrap();
    assert!(f64::abs(partial_xy - expected_dxy) < 1e-12);
}

#[test]
fn ad_third_partials() {
    // f = x^3 y^3 z^3:  d3/dx dy dz = 27 x^2 y^2 z^2 = 972;  d3/dx2 dy = 18 x y^2 z^3 = 1944
    let function =
        scalar_fn!(|point: &[f64; 3]| point[0].powi(3) * point[1].powi(3) * point[2].powi(3));
    let derivator = AutoDiffMulti::default();
    let point = [1.0, 2.0, 3.0];

    let x = 0;
    let y = 1;
    let z = 2;

    let mixed_xyz = derivator
        .differentiate(&function, &[x, y, z], &point)
        .unwrap();
    assert!(f64::abs(mixed_xyz - 972.0) < 1e-9);

    let mixed_xxy = derivator
        .differentiate(&function, &[x, x, y], &point)
        .unwrap();
    assert!(f64::abs(mixed_xxy - 1944.0) < 1e-9);
}

#[test]
fn ad_jacobian() {
    // (x*y*z, x^2 + y^2)
    let function = scalar_fn_vec!(|point: &[f64; 3]| [
        point[0] * point[1] * point[2],
        point[0] * point[0] + point[1] * point[1]
    ]);
    let jacobian: Jacobian = Jacobian::default();
    let result = jacobian.evaluate(&function, &[1.0, 2.0, 3.0]).unwrap();

    let expected = [[6.0, 3.0, 2.0], [2.0, 4.0, 0.0]];
    for (row, expected_row) in expected.iter().enumerate() {
        for (column, &expected_entry) in expected_row.iter().enumerate() {
            assert!(f64::abs(result[(row, column)] - expected_entry) < 1e-12);
        }
    }
}

#[test]
#[cfg(feature = "alloc")]
fn ad_jacobian_on_heap() {
    let function = scalar_fn_vec!(|point: &[f64; 3]| [
        point[0] * point[1] * point[2],
        point[0] * point[0] + point[1] * point[1]
    ]);
    let jacobian: Jacobian = Jacobian::default();
    let result = jacobian
        .evaluate_on_heap(&function, &[1.0, 2.0, 3.0])
        .unwrap();

    let expected = [[6.0, 3.0, 2.0], [2.0, 4.0, 0.0]];
    for (row, expected_row) in expected.iter().enumerate() {
        for (column, &expected_entry) in expected_row.iter().enumerate() {
            assert!(f64::abs(result[row][column] - expected_entry) < 1e-12);
        }
    }
}

#[test]
fn ad_hessian() {
    // f(x, y) = y*sin(x) + 2*x*e^y
    let function =
        scalar_fn!(|point: &[f64; 2]| point[1] * point[0].sin()
            + constant(2.0) * point[0] * point[1].exp());
    let hessian: Hessian = Hessian::default();
    let result = hessian.evaluate(&function, &[1.0, 2.0]).unwrap();

    let expected = [
        [-2.0 * f64::sin(1.0), f64::cos(1.0) + 2.0 * f64::exp(2.0)],
        [f64::cos(1.0) + 2.0 * f64::exp(2.0), 2.0 * f64::exp(2.0)],
    ];
    for (row, expected_row) in expected.iter().enumerate() {
        for (column, &expected_entry) in expected_row.iter().enumerate() {
            assert!(f64::abs(result[(row, column)] - expected_entry) < 1e-12);
        }
    }
}

#[test]
fn ad_f32() {
    // x*x at 0.5; first derivative 2x = 1.0, exact under autodiff
    let function = scalar_fn!(|x| x * x);
    let derivator = AutoDiffSingle::<f32>::default();
    assert!(f32::abs(derivator.differentiate(1, &function, 0.5_f32).unwrap() - 1.0) < 1e-6);
}

// ----- autodiff error handling -----

#[test]
fn ad_error_index_out_of_range() {
    let function = scalar_fn!(|point: &[f64; 3]| point[0] + point[1] + point[2]);
    let derivator = AutoDiffMulti::default();
    let result = derivator.first_partial_derivative(&function, 5, &[1.0, 2.0, 3.0]);
    assert_eq!(result.unwrap_err(), DiffError::IndexOutOfRange);
}

#[test]
fn ad_error_order_zero() {
    let function = scalar_fn!(|point: &[f64; 3]| point[0] + point[1] + point[2]);
    let derivator = AutoDiffMulti::default();
    let variable_indices: [usize; 0] = [];
    assert_eq!(
        derivator
            .differentiate(&function, &variable_indices, &[1.0, 2.0, 3.0])
            .unwrap_err(),
        DiffError::OrderZero
    );
}

#[test]
fn ad_error_order_unsupported() {
    // autodiff multi caps at third order; a fourth-order partial is rejected
    let function = scalar_fn!(|point: &[f64; 3]| point[0] + point[1] + point[2]);
    let derivator = AutoDiffMulti::default();
    assert_eq!(
        derivator
            .differentiate(&function, &[0, 1, 2, 0], &[1.0, 2.0, 3.0])
            .unwrap_err(),
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
    let result = jacobian.evaluate(&EmptyVectorFn, &[1.0, 2.0, 3.0]);
    assert_eq!(result.unwrap_err(), DiffError::EmptyFunctionSet);
}

// ----- column-seeded Jacobian -----

// A VectorFn that counts how many times it is evaluated, to prove the column-seeded harness
// runs one pass per input column (N) rather than one per matrix cell (M*N).
struct CountingVectorFn {
    calls: Cell<usize>,
}

impl VectorFn<3, 2> for CountingVectorFn {
    fn eval<S: Numeric>(&self, point: &[S; 3]) -> [S; 2] {
        self.calls.set(self.calls.get() + 1);
        // (x*y*z, x^2 + y^2)
        [
            point[0] * point[1] * point[2],
            point[0] * point[0] + point[1] * point[1],
        ]
    }
}

#[test]
fn ad_jacobian_is_column_seeded() {
    let counter = CountingVectorFn {
        calls: Cell::new(0),
    };
    let jacobian: Jacobian = Jacobian::default();
    let result = jacobian.evaluate(&counter, &[1.0, 2.0, 3.0]).unwrap();

    // values are unchanged from the old harness
    let expected = [[6.0, 3.0, 2.0], [2.0, 4.0, 0.0]];
    for (row, expected_row) in expected.iter().enumerate() {
        for (column, &expected_entry) in expected_row.iter().enumerate() {
            assert!(f64::abs(result[(row, column)] - expected_entry) < 1e-12);
        }
    }

    // one evaluation per input column (3), not per cell (2*3 = 6)
    assert_eq!(counter.calls.get(), 3);
}

#[test]
fn ad_jacobian_column_reads_all_outputs() {
    // one seeded pass on input 0 gives d/dx of both outputs: [y*z, 2x] = [6, 2]
    let function = scalar_fn_vec!(|point: &[f64; 3]| [
        point[0] * point[1] * point[2],
        point[0] * point[0] + point[1] * point[1]
    ]);
    let derivator = AutoDiffMulti::default();
    let column = derivator
        .jacobian_column(&function, 0, &[1.0, 2.0, 3.0])
        .unwrap();
    assert!(f64::abs(column[0] - 6.0) < 1e-12);
    assert!(f64::abs(column[1] - 2.0) < 1e-12);
}

#[test]
fn ad_jacobian_column_index_out_of_range() {
    let function = scalar_fn_vec!(|point: &[f64; 3]| [
        point[0] * point[1] * point[2],
        point[0] * point[0] + point[1] * point[1]
    ]);
    let derivator = AutoDiffMulti::default();
    let result = derivator.jacobian_column(&function, 5, &[1.0, 2.0, 3.0]);
    assert_eq!(result.unwrap_err(), DiffError::IndexOutOfRange);
}

#[test]
fn fd_jacobian_column_matches() {
    // the finite-difference implementation produces the right matrix, matching the analytic
    // values to finite-difference tolerance (unchanged from the per-Component path it replaces)
    let function = scalar_fn_vec!(|point: &[f64; 3]| [
        point[0] * point[1] * point[2],
        point[0] * point[0] + point[1] * point[1]
    ]);
    let jacobian = Jacobian::from_derivator(FiniteDifferenceMulti::default());
    let result = jacobian.evaluate(&function, &[1.0, 2.0, 3.0]).unwrap();

    let expected = [[6.0, 3.0, 2.0], [2.0, 4.0, 0.0]];
    for (row, expected_row) in expected.iter().enumerate() {
        for (column, &expected_entry) in expected_row.iter().enumerate() {
            assert!(f64::abs(result[(row, column)] - expected_entry) < 1e-5);
        }
    }
}

#[test]
fn fd_jacobian_is_column_seeded() {
    // central difference evaluates the full function twice per input column (2*3 = 6), not twice
    // per matrix cell (2*M*N = 12)
    let counter = CountingVectorFn {
        calls: Cell::new(0),
    };
    let jacobian = Jacobian::from_derivator(FiniteDifferenceMulti::default());
    let _ = jacobian.evaluate(&counter, &[1.0, 2.0, 3.0]).unwrap();
    assert_eq!(counter.calls.get(), 6);
}

// ----- finite differences: kept as a sparse fallback for the engine and the cases autodiff
//       does not cover (high-order mixed partials, zero-step-size error) -----

#[test]
fn fd_single_derivative_modes() {
    // x^2/2, derivative x; check every mode with its type-scaled default step.
    let function = scalar_fn!(|x| constant(0.5) * x * x);
    for mode in [
        FiniteDifferenceMode::Forward,
        FiniteDifferenceMode::Backward,
        FiniteDifferenceMode::Central,
    ] {
        let derivator = FiniteDifferenceSingle::from_parameters(
            mode.default_step_size::<f64>(),
            mode,
            DEFAULT_STEP_SIZE_MULTIPLIER,
        );
        assert!(f64::abs(derivator.differentiate(1, &function, 2.0).unwrap() - 2.0) < 0.001);
    }
}

#[test]
fn fd_default_step_scales_with_scalar_type() {
    assert_eq!(
        FiniteDifferenceSingle::<f32>::default().config.step_size,
        f32::EPSILON.cbrt()
    );
    assert_eq!(
        FiniteDifferenceSingle::<f64>::default().config.step_size,
        f64::EPSILON.cbrt()
    );
    for mode in [
        FiniteDifferenceMode::Forward,
        FiniteDifferenceMode::Backward,
    ] {
        assert_eq!(mode.default_step_size::<f32>(), f32::EPSILON.sqrt());
        assert_eq!(mode.default_step_size::<f64>(), f64::EPSILON.sqrt());
    }
}

#[test]
fn fd_default_is_accurate_at_f32() {
    // f(x) = x²/2, so f'(2) = 2. The error bar follows the expected rounding scale for a
    // central difference whose step is cbrt(epsilon), rather than a fixed decimal tolerance.
    let function = scalar_fn!(|x| constant(0.5) * x * x);
    let derivator = FiniteDifferenceSingle::<f32>::default();
    let derivative = derivator.differentiate(1, &function, 2.0_f32).unwrap();
    let step = f32::EPSILON.cbrt();

    assert!((derivative - 2.0).abs() < 4.0 * step * step);
}

#[test]
fn fd_explicit_positive_step_is_preserved() {
    // A forward difference of x²/2 at x=2 is exactly 2 + step/2.
    let function = scalar_fn!(|x| constant(0.5) * x * x);
    let step = 0.25_f64;
    let derivator = FiniteDifferenceSingle::from_parameters(
        step,
        FiniteDifferenceMode::Forward,
        DEFAULT_STEP_SIZE_MULTIPLIER,
    );

    assert_eq!(derivator.config.step_size, step);
    assert_eq!(
        derivator.differentiate(1, &function, 2.0).unwrap(),
        2.0 + step / 2.0
    );
}

#[test]
fn fd_invalid_step_error() {
    let function = scalar_fn!(|x| constant(0.5) * x * x);
    for step in [-1.0_f64, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let derivator = FiniteDifferenceSingle::from_parameters(
            step,
            FiniteDifferenceMode::Central,
            DEFAULT_STEP_SIZE_MULTIPLIER,
        );
        assert_eq!(
            derivator.differentiate(1, &function, 2.0).unwrap_err(),
            DiffError::InvalidStepSize
        );
    }
}

#[test]
fn fd_step_size_zero_error() {
    let function = scalar_fn!(|point: &[f64; 3]| point[1] * point[0].sin());
    let derivator = FiniteDifferenceMulti::from_parameters(0.0, FiniteDifferenceMode::Central, 1.0);
    assert_eq!(
        derivator
            .differentiate(&function, &[0], &[1.0, 2.0, 3.0])
            .unwrap_err(),
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
    fn eval<S: Numeric>(&self, point: &[S; 2]) -> S {
        let x = point[0];
        let y = point[1];
        let coefficient = |index| S::from_f64(self.coeffs[index]);
        coefficient(0)
            + coefficient(1) * x
            + coefficient(2) * y
            + coefficient(3) * x * x
            + coefficient(4) * x * y
            + coefficient(5) * y * y
    }
}

fn coeff_l1(coeffs: &[f64]) -> f64 {
    coeffs.iter().map(|coeff| coeff.abs()).sum()
}

fn ad_fd_tol(autodiff: f64, step: f64, order: i32, coeff: f64, coeff_scale: f64) -> f64 {
    let scale = autodiff.abs().max(1.0) * coeff_scale.max(1.0);
    coeff * step.powi(order) * scale
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
        let composed_polynomial = PolyComp { inner, outer };
        let step = DEFAULT_STEP_SIZE;
        let autodiff = AutoDiffSingle::default().differentiate(1, &composed_polynomial, x).unwrap();
        let finite_difference_derivator = FiniteDifferenceSingle::default();
        let finite_difference =
            finite_difference_derivator.differentiate(1, &composed_polynomial, x).unwrap();
        let tolerance = ad_fd_tol(autodiff, step, 2, 1e3, scale);
        prop_assert!(
            (finite_difference - autodiff).abs() < tolerance,
            "fd={finite_difference} ad={autodiff} tol={tolerance} x={x}"
        );
    }

    #[test]
    fn proptest_ad_fd_multi_first_partial(
        coeffs in prop::collection::vec(-5.0f64..5.0, 6),
        x in -2.0f64..2.0,
        y in -2.0f64..2.0,
    ) {
        let scale = 1.0 + coeff_l1(&coeffs);
        let bivariate = BivariatePoly { coeffs };
        let point = [x, y];
        let step = DEFAULT_STEP_SIZE;
        let autodiff_derivator = AutoDiffMulti::default();
        let finite_difference_derivator = FiniteDifferenceMulti::default();
        for variable_index in [0usize, 1] {
            let autodiff =
                autodiff_derivator.first_partial_derivative(&bivariate, variable_index, &point).unwrap();
            let finite_difference =
                finite_difference_derivator.first_partial_derivative(&bivariate, variable_index, &point).unwrap();
            let tolerance = ad_fd_tol(autodiff, step, 2, 1e3, scale);
            prop_assert!(
                (finite_difference - autodiff).abs() < tolerance,
                "idx={variable_index} fd={finite_difference} ad={autodiff} tol={tolerance}"
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
            let step = 1e-4;
            let autodiff = AutoDiffSingle::default().differentiate(2, &f, x).unwrap();
            let finite_diff = FiniteDifferenceSingle::from_parameters(
                step,
                FiniteDifferenceMode::Central,
                DEFAULT_STEP_SIZE_MULTIPLIER,
            );
            let finite_diff_value = finite_diff.differentiate(2, &f, x).unwrap();
            let tol = ad_fd_tol(autodiff, step, 2, 1e4, scale);
            prop_assert!(
                (finite_diff_value - autodiff).abs() < tol,
                "fd={finite_diff_value} ad={autodiff} tol={tol}"
            );
            Ok(())
        })
        .unwrap();
}
