use multicalc::numerical_integration::mode::*;

use multicalc::numerical_integration::gaussian_integration;
use multicalc::numerical_integration::integrator::*;
use multicalc::numerical_integration::iterative_integration;
use multicalc::utils::error_codes::*;

#[test]
fn test_booles_integration_1() {
    //equation is 2.0*x
    let func = |args: f64| -> f64 { 2.0 * args };

    let integration_limit = [0.0, 2.0];

    let integrator =
        iterative_integration::IterativeSingle::from_parameters(100, IterativeMethod::Booles);

    //simple integration for x, known to be x*x, expect a value of ~4.00
    let val = integrator.get_single(&func, &integration_limit).unwrap();
    assert!(f64::abs(val - 4.0) < 1e-14);
}

#[test]
fn test_booles_integration_2() {
    //equation is 2.0*x + y*z
    let func = |args: &[f64; 3]| -> f64 { 2.0 * args[0] + args[1] * args[2] };

    let integration_limit = [0.0, 1.0];
    let point = [1.0, 2.0, 3.0];

    let integrator =
        iterative_integration::IterativeMulti::from_parameters(100, IterativeMethod::Booles);

    //partial integration for x, known to be x*x + x*y*z, expect a value of ~7.00
    let val = integrator
        .get_single_partial(&func, 0, &integration_limit, &point)
        .unwrap();
    assert!(f64::abs(val - 7.0) < 1e-25);

    let integration_limit = [0.0, 2.0];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of ~10.00
    let val = integrator
        .get_single_partial(&func, 1, &integration_limit, &point)
        .unwrap();
    assert!(f64::abs(val - 10.0) < 0.00001);

    let integration_limit = [0.0, 3.0];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of ~15.0
    let val = integrator
        .get_single_partial(&func, 2, &integration_limit, &point)
        .unwrap();
    assert!(f64::abs(val - 15.0) < 0.00001);
}

#[test]
fn test_booles_integration_3() {
    //equation is 6.0*x
    let func = |args: f64| -> f64 { 6.0 * args };

    let integration_limits = [[0.0, 2.0], [0.0, 2.0]];

    let integrator =
        iterative_integration::IterativeSingle::from_parameters(20, IterativeMethod::Booles);

    //simple double integration for 6*x, expect a value of ~24.00
    let val = integrator.get_double(&func, &integration_limits).unwrap();
    assert!(f64::abs(val - 24.0) < 0.00001);
}

#[test]
fn test_gauss_legendre_quadrature_integration_1() {
    //equation is 4.0*x*x*x - 3.0*x*x
    let func = |args: f64| -> f64 { 4.0 * args * args * args - 3.0 * args * args };

    let integration_limit = [0.0, 2.0];

    let integrator = gaussian_integration::GaussianSingle::from_parameters(
        4,
        GaussianQuadratureMethod::GaussLegendre,
    );

    //simple integration for x, known to be x^4 - x^3, expect a value of ~8.00
    let val = integrator.get_single(&func, &integration_limit).unwrap();
    assert!(f64::abs(val - 8.0) < 1e-14);
}

#[test]
fn test_gauss_legendre_quadrature_integration_2() {
    //equation is 2.0*x + y*z
    let func = |args: &[f64; 3]| -> f64 { 2.0 * args[0] + args[1] * args[2] };

    let integration_limit = [0.0, 1.0];
    let point = [1.0, 2.0, 3.0];

    let integrator = gaussian_integration::GaussianMulti::from_parameters(
        2,
        GaussianQuadratureMethod::GaussLegendre,
    );

    //partial integration for x, known to be x*x + x*y*z, expect a value of ~7.00
    let val = integrator
        .get_single_partial(&func, 0, &integration_limit, &point)
        .unwrap();
    assert!(f64::abs(val - 7.0) < 1e-14);

    let integration_limit = [0.0, 2.0];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of ~10.00
    let val = integrator
        .get_single_partial(&func, 1, &integration_limit, &point)
        .unwrap();
    assert!(f64::abs(val - 10.0) < 1e-14);

    let integration_limit = [0.0, 3.0];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of ~15.0
    let val = integrator
        .get_single_partial(&func, 2, &integration_limit, &point)
        .unwrap();
    assert!(f64::abs(val - 15.0) < 1e-14);
}

#[test]
fn test_gauss_legendre_quadrature_integration_3() {
    //equation is 6.0*x
    let func = |args: f64| -> f64 { 6.0 * args };

    let integration_limits = [[0.0, 2.0], [0.0, 2.0]];
    let integrator = gaussian_integration::GaussianSingle::from_parameters(
        2,
        GaussianQuadratureMethod::GaussLegendre,
    );

    //simple double integration for 6*x, expect a value of ~24.00
    let val = integrator.get_double(&func, &integration_limits).unwrap();
    assert!(f64::abs(val - 24.0) < 1e-14);
}

#[test]
fn test_simpsons_integration_1() {
    //equation is 2.0*x
    let func = |args: f64| -> f64 { 2.0 * args };

    let integration_limit = [0.0, 2.0];

    let integrator =
        iterative_integration::IterativeSingle::from_parameters(200, IterativeMethod::Simpsons);

    //simple integration for x, known to be x*x, expect a value of ~4.00
    let val = integrator.get_single(&func, &integration_limit).unwrap();
    assert!(f64::abs(val - 4.0) < 0.05);
}

#[test]
fn test_simpsons_integration_2() {
    //equation is 2.0*x + y*z
    let func = |args: &[f64; 3]| -> f64 { 2.0 * args[0] + args[1] * args[2] };

    let integration_limit = [0.0, 1.0];
    let point = [1.0, 2.0, 3.0];

    let integrator =
        iterative_integration::IterativeMulti::from_parameters(200, IterativeMethod::Simpsons);

    //partial integration for x, known to be x*x + x*y*z, expect a value of ~7.00
    let val = integrator
        .get_single_partial(&func, 0, &integration_limit, &point)
        .unwrap();
    assert!(f64::abs(val - 7.0) < 0.05);

    let integration_limit = [0.0, 2.0];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of ~10.00
    let val = integrator
        .get_single_partial(&func, 1, &integration_limit, &point)
        .unwrap();
    assert!(f64::abs(val - 10.0) < 0.05);

    let integration_limit = [0.0, 3.0];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of ~15.0
    let val = integrator
        .get_single_partial(&func, 2, &integration_limit, &point)
        .unwrap();
    assert!(f64::abs(val - 15.0) < 0.05);
}

#[test]
fn test_simpsons_integration_3() {
    //equation is 6.0*x
    let func = |args: f64| -> f64 { 6.0 * args };

    let integration_limits = [[0.0, 2.0], [0.0, 2.0]];

    let integrator =
        iterative_integration::IterativeSingle::from_parameters(200, IterativeMethod::Simpsons);

    //simple double integration for 6*x, expect a value of ~24.00
    let val = integrator.get_double(&func, &integration_limits).unwrap();
    assert!(f64::abs(val - 24.0) < 0.05);
}

#[test]
fn test_simpsons_integration_4() {
    //equation is 2.0*x + y*z
    let func = |args: &[f64; 3]| -> f64 { 2.0 * args[0] + args[1] * args[2] };

    let integration_limits = [[0.0, 1.0], [0.0, 1.0]];
    let point = [1.0, 1.0, 1.0];

    let integrator =
        iterative_integration::IterativeMulti::from_parameters(200, IterativeMethod::Simpsons);

    //double partial integration for first x then y, expect a value of ~1.50
    let val = integrator
        .get_double_partial(&func, [0, 1], &integration_limits, &point)
        .unwrap();
    assert!(f64::abs(val - 1.50) < 0.05);
}

#[test]
fn test_trapezoidal_integration_1() {
    //equation is 2.0*x
    let func = |args: f64| -> f64 { 2.0 * args };

    let integration_limit = [0.0, 2.0];

    let iterator =
        iterative_integration::IterativeSingle::from_parameters(100, IterativeMethod::Trapezoidal);
    let val = iterator.get_single(&func, &integration_limit).unwrap();

    assert!(f64::abs(val - 4.0) < 0.00001);
}

#[test]
fn test_trapezoidal_integration_2() {
    //equation is 2.0*x + y*z
    let func = |args: &[f64; 3]| -> f64 { 2.0 * args[0] + args[1] * args[2] };

    let integration_limit = [0.0, 1.0];
    let point = [1.0, 2.0, 3.0];

    let iterator =
        iterative_integration::IterativeMulti::from_parameters(100, IterativeMethod::Trapezoidal);

    //partial integration for x, known to be x*x + x*y*z, expect a value of ~7.00
    let val = iterator
        .get_single_partial(&func, 0, &integration_limit, &point)
        .unwrap();
    assert!(f64::abs(val - 7.0) < 0.00001);

    let integration_limit = [0.0, 2.0];

    //partial integration for y, known to be 2.0*x*y + y*y*z/2.0, expect a value of ~10.00
    let val = iterator
        .get_single_partial(&func, 1, &integration_limit, &point)
        .unwrap();
    assert!(f64::abs(val - 10.0) < 0.00001);

    let integration_limit = [0.0, 3.0];

    //partial integration for z, known to be 2.0*x*z + y*z*z/2.0, expect a value of ~15.0
    let val = iterator
        .get_single_partial(&func, 2, &integration_limit, &point)
        .unwrap();
    assert!(f64::abs(val - 15.0) < 0.00001);
}

#[test]
fn test_trapezoidal_integration_3() {
    //equation is 6.0*x
    let func = |args: f64| -> f64 { 6.0 * args };

    let integration_limits = [[0.0, 2.0], [0.0, 2.0]];

    let integrator =
        iterative_integration::IterativeSingle::from_parameters(10, IterativeMethod::Trapezoidal);

    //simple double integration for 6*x, expect a value of ~24.00
    let val = integrator.get_double(&func, &integration_limits).unwrap();
    assert!(f64::abs(val - 24.0) < 0.00001);
}

#[test]
fn test_trapezoidal_integration_4() {
    //equation is 2.0*x + y*z
    let func = |args: &[f64; 3]| -> f64 { 2.0 * args[0] + args[1] * args[2] };

    let integration_limits = [[0.0, 1.0], [0.0, 2.0]];
    let point = [1.0, 2.0, 3.0];

    let integrator =
        iterative_integration::IterativeMulti::from_parameters(10, IterativeMethod::Trapezoidal);

    //double partial integration for first x then y, expect a value of ~2.50
    let val = integrator
        .get_double_partial(&func, [0, 1], &integration_limits, &point)
        .unwrap();
    assert!(f64::abs(val - 8.0) < 0.00001);
}

#[test]
fn test_error_checking_1() {
    //equation is 2.0*x
    let func = |args: f64| -> f64 { 2.0 * args };

    let integration_limit = [10.0, 1.0];

    let integrator = iterative_integration::IterativeSingle::default();

    //expect failure because integration interval is ill-defined (lower limit is higher than the upper limit)
    let result = integrator.get_single(&func, &integration_limit);
    assert!(result.is_err());
    assert!(result.unwrap_err() == CalcError::IntegrationLimitsIllDefined);
}

#[test]
fn test_error_checking_2() {
    //equation is 2.0*x
    let func = |args: f64| -> f64 { 2.0 * args };

    let integration_limit = [0.0, 1.0];

    let integrator =
        iterative_integration::IterativeSingle::from_parameters(0, IterativeMethod::Booles);

    //expect failure because number of steps is 0
    let result = integrator.get_single(&func, &integration_limit);
    assert!(result.is_err());
    assert!(result.unwrap_err() == CalcError::IterationsZero);
}

//TODO: add more tests

#[test]
fn test_error_checking_3() {
    //equation is 4.0*x*x*x - 3.0*x*x
    let func = |args: f64| -> f64 { 4.0 * args * args * args - 3.0 * args * args };

    let integration_limit = [0.0, 2.0];

    //Gauss Legendre not valid for n < 1
    let integrator = gaussian_integration::GaussianSingle::from_parameters(
        0,
        GaussianQuadratureMethod::GaussLegendre,
    );
    let result = integrator.get_single(&func, &integration_limit);
    assert!(result.is_err());
    assert!(result.unwrap_err() == CalcError::QuadratureOrderOutOfRange);
}

#[test]
fn test_error_checking_4() {
    //equation is 4.0*x*x*x - 3.0*x*x
    let func = |args: f64| -> f64 { 4.0 * args * args * args - 3.0 * args * args };

    let integration_limit = [0.0, 2.0];

    //Gauss Legendre not valid for n > 30
    let integrator = gaussian_integration::GaussianSingle::from_parameters(
        31,
        GaussianQuadratureMethod::GaussLegendre,
    );
    let result = integrator.get_single(&func, &integration_limit);
    assert!(result.is_err());
    assert!(result.unwrap_err() == CalcError::QuadratureOrderOutOfRange);
}

#[test]
fn test_gauss_hermite_single() {
    //integrand is x*x; weights carry the e^{-x*x} kernel
    //∫_{-∞}^∞ x² e^{-x²} dx = √π / 2
    let func = |x: f64| -> f64 { x * x };

    let integrator = gaussian_integration::GaussianSingle::from_parameters(
        5,
        GaussianQuadratureMethod::GaussHermite,
    );

    let integration_limit = [f64::NEG_INFINITY, f64::INFINITY];
    let val = integrator.get_single(&func, &integration_limit).unwrap();

    let expected = core::f64::consts::PI.sqrt() / 2.0;
    assert!(f64::abs(val - expected) < 1e-10);
}

#[test]
fn test_gauss_laguerre_single() {
    //integrand is x*x; weights carry the e^{-x} kernel
    //∫_0^∞ x² e^{-x} dx = 2
    let func = |x: f64| -> f64 { x * x };

    let integrator = gaussian_integration::GaussianSingle::from_parameters(
        5,
        GaussianQuadratureMethod::GaussLaguerre,
    );

    let integration_limit = [0.0, f64::INFINITY];
    let val = integrator.get_single(&func, &integration_limit).unwrap();

    assert!(f64::abs(val - 2.0) < 1e-9);
}

#[test]
fn test_gauss_hermite_multivariable() {
    //∫∫ x² y² e^{-x²} e^{-y²} dx dy = (√π/2)²
    let func = |args: &[f64; 2]| -> f64 { args[0] * args[0] * args[1] * args[1] };

    let integrator = gaussian_integration::GaussianMulti::from_parameters(
        5,
        GaussianQuadratureMethod::GaussHermite,
    );

    let integration_limits = [
        [f64::NEG_INFINITY, f64::INFINITY],
        [f64::NEG_INFINITY, f64::INFINITY],
    ];
    let point = [0.0, 0.0];
    let val = integrator
        .get([0, 1], &func, &integration_limits, &point)
        .unwrap();

    let sqrt_pi_half = core::f64::consts::PI.sqrt() / 2.0;
    assert!(f64::abs(val - sqrt_pi_half * sqrt_pi_half) < 1e-10);
}

#[test]
fn test_gauss_laguerre_multivariable() {
    //∫∫ x² y² e^{-x} e^{-y} dx dy = 2 * 2 = 4
    let func = |args: &[f64; 2]| -> f64 { args[0] * args[0] * args[1] * args[1] };

    let integrator = gaussian_integration::GaussianMulti::from_parameters(
        5,
        GaussianQuadratureMethod::GaussLaguerre,
    );

    let integration_limits = [[0.0, f64::INFINITY], [0.0, f64::INFINITY]];
    let point = [0.0, 0.0];
    let val = integrator
        .get([0, 1], &func, &integration_limits, &point)
        .unwrap();

    assert!(f64::abs(val - 4.0) < 1e-8);
}

#[test]
fn test_iterative_infinite_gaussian() {
    //∫_{-∞}^∞ e^{-x²} dx = √π
    let func = |x: f64| -> f64 { f64::exp(-x * x) };

    let integrator = iterative_integration::IterativeSingle::default();

    let integration_limit = [f64::NEG_INFINITY, f64::INFINITY];
    let val = integrator.get_single(&func, &integration_limit).unwrap();

    assert!(f64::abs(val - core::f64::consts::PI.sqrt()) < 1e-3);
}

#[test]
fn test_iterative_semi_infinite_exp() {
    //∫_0^∞ e^{-x} dx = 1
    let func = |x: f64| -> f64 { f64::exp(-x) };

    let integrator = iterative_integration::IterativeSingle::default();

    let integration_limit = [0.0, f64::INFINITY];
    let val = integrator.get_single(&func, &integration_limit).unwrap();

    assert!(f64::abs(val - 1.0) < 1e-3);
}

#[test]
fn test_iterative_semi_infinite_inverse_square() {
    //∫_1^∞ x^{-2} dx = 1
    let func = |x: f64| -> f64 { 1.0 / (x * x) };

    let integrator = iterative_integration::IterativeSingle::default();

    let integration_limit = [1.0, f64::INFINITY];
    let val = integrator.get_single(&func, &integration_limit).unwrap();

    assert!(f64::abs(val - 1.0) < 1e-3);
}

#[test]
fn test_iterative_negative_limits() {
    //∫_{-2}^{1} 2x dx = x² evaluated from -2 to 1 = 1 - 4 = -3
    let func = |x: f64| -> f64 { 2.0 * x };

    let integrator = iterative_integration::IterativeSingle::default();

    let integration_limit = [-2.0, 1.0];
    let val = integrator.get_single(&func, &integration_limit).unwrap();

    assert!(f64::abs(val - (-3.0)) < 1e-9);
}

#[test]
fn test_composite_rule_degree_3_polynomial() {
    //a degree-3 integrand exposes composite-rule divisibility (a linear one would be exact
    //under every rule and hide it). ∫_0^2 x³ dx = 4
    let func = |x: f64| -> f64 { x * x * x };

    let integration_limit = [0.0, 2.0];

    //120 is a multiple of 3, so Simpson's 3/8 is exact for cubics here
    let simpson =
        iterative_integration::IterativeSingle::from_parameters(120, IterativeMethod::Simpsons);
    let val = simpson.get_single(&func, &integration_limit).unwrap();
    assert!(f64::abs(val - 4.0) < 1e-9);

    //120 is a multiple of 4, so Boole's rule is exact too
    let boole =
        iterative_integration::IterativeSingle::from_parameters(120, IterativeMethod::Booles);
    let val = boole.get_single(&func, &integration_limit).unwrap();
    assert!(f64::abs(val - 4.0) < 1e-9);
}

//naive left-to-right trapezoidal accumulation, matching the library's point stepping and
//weights so the only difference from the pairwise version is the summation order
fn naive_trapezoidal<F: Fn(f64) -> f64>(iterations: u64, lo: f64, hi: f64, f: F) -> f64 {
    let delta = (hi - lo) / iterations as f64;
    let mut point = lo;
    let mut ans = f(point);
    for _ in 0..iterations - 1 {
        point += delta;
        ans += 2.0 * f(point);
    }
    ans += f(hi);
    0.5 * delta * ans
}

#[test]
fn pairwise_integration_is_accurate_on_long_sum() {
    //1/(1+x^2) over [0, 1] is exactly pi/4. With 2^23 intervals the trapezoidal truncation
    //error sits at machine epsilon, so naive accumulation (error ~ n*eps) becomes the limiting
    //factor; pairwise keeps the result truncation-limited and lands far closer to the exact value
    let func = |x: f64| -> f64 { 1.0 / (1.0 + x * x) };
    let exact = core::f64::consts::PI / 4.0;
    let iterations: u64 = 1 << 23;

    let integrator = iterative_integration::IterativeSingle::from_parameters(
        iterations,
        IterativeMethod::Trapezoidal,
    );
    let pairwise = integrator.get_single(&func, &[0.0, 1.0]).unwrap();
    let naive = naive_trapezoidal(iterations, 0.0, 1.0, func);

    let pairwise_err = f64::abs(pairwise - exact);
    let naive_err = f64::abs(naive - exact);

    //tighter than the ~n*eps a naive sum could reach at this term count
    assert!(pairwise_err < 1e-12, "pairwise error {pairwise_err:e} too large");
    //and strictly closer to the exact value than the naive accumulation
    assert!(
        pairwise_err < naive_err,
        "pairwise ({pairwise_err:e}) should be closer than naive ({naive_err:e})"
    );
}

#[test]
fn test_booles_integration_f32() {
    //2x integrated over [0, 2] is 4
    let func = |x: f32| -> f32 { 2.0 * x };

    let integrator = iterative_integration::IterativeSingle::<f32>::from_parameters(
        100,
        IterativeMethod::Booles,
    );

    let val = integrator.get_single(&func, &[0.0, 2.0]).unwrap();
    assert!(f32::abs(val - 4.0) < 1e-3, "got {val}");
}

#[test]
fn test_gauss_legendre_integration_f32() {
    //4x^3 - 3x^2 integrated over [0, 2] is 8
    let func = |x: f32| -> f32 { 4.0 * x * x * x - 3.0 * x * x };

    let integrator = gaussian_integration::GaussianSingle::<f32>::from_parameters(
        4,
        GaussianQuadratureMethod::GaussLegendre,
    );

    let val = integrator.get_single(&func, &[0.0, 2.0]).unwrap();
    assert!(f32::abs(val - 8.0) < 1e-2, "got {val}");
}
