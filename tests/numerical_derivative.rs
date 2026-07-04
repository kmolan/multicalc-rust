use multicalc::numerical_derivative::autodiff::{AutoDiffMulti, AutoDiffSingle};
use multicalc::numerical_derivative::derivator::*;
use multicalc::numerical_derivative::finite_difference::*;
use multicalc::numerical_derivative::hessian::Hessian;
use multicalc::numerical_derivative::jacobian::Jacobian;
use multicalc::numerical_derivative::mode::*;
use multicalc::scalar::{Numeric, VectorFn, c};
use multicalc::utils::error_codes::*;
use multicalc::{scalar_fn, scalar_fn_vec};

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
    let func = scalar_fn!(|v: &[f64; 3]| {
        v[1] * v[0].sin() + v[0] * v[1].cos() + v[0] * v[1] * v[2].exp()
    });
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
    let func = scalar_fn!(|v: &[f64; 3]| {
        v[1] * v[0].sin() + v[0] * v[1].cos() + v[0] * v[1] * v[2].exp()
    });
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
    for i in 0..2 {
        for j in 0..3 {
            assert!(f64::abs(result[i][j] - expected[i][j]) < 1e-12);
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
    for i in 0..2 {
        for j in 0..3 {
            assert!(f64::abs(result[i][j] - expected[i][j]) < 1e-12);
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
    for i in 0..2 {
        for j in 0..2 {
            assert!(f64::abs(result[i][j] - expected[i][j]) < 1e-12);
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
    assert_eq!(result.unwrap_err(), CalcError::IndexOutOfRange);
}

#[test]
fn ad_error_order_zero() {
    let func = scalar_fn!(|v: &[f64; 3]| v[0] + v[1] + v[2]);
    let d = AutoDiffMulti::default();
    let idx: [usize; 0] = [];
    assert_eq!(
        d.get(&func, &idx, &[1.0, 2.0, 3.0]).unwrap_err(),
        CalcError::DerivativeOrderZero
    );
}

#[test]
fn ad_error_order_unsupported() {
    // autodiff multi caps at third order; a fourth-order partial is rejected
    let func = scalar_fn!(|v: &[f64; 3]| v[0] + v[1] + v[2]);
    let d = AutoDiffMulti::default();
    assert_eq!(
        d.get(&func, &[0, 1, 2, 0], &[1.0, 2.0, 3.0]).unwrap_err(),
        CalcError::DerivativeOrderUnsupported
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
    assert_eq!(result.unwrap_err(), CalcError::EmptyFunctionSet);
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
        CalcError::StepSizeZero
    );
}
