#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::error::IntegrateError;
use multicalc::numerical_integration::*;

use proptest::prelude::*;

#[test]
fn booles_rule_integrates_a_line_to_its_closed_form() {
    // closed form of 2x is x*x
    let line = |x: f64| -> f64 { 2.0 * x };
    let limits = [0.0, 2.0];
    let interval_count = 100;
    let integrator = IterativeSingle::from_parameters(interval_count, IterativeMethod::Booles);

    let expected = 4.0;
    let area = integrator.single_integral(&line, &limits).unwrap();
    assert!(f64::abs(area - expected) < 1e-14);
}

#[test]
fn booles_rule_takes_partial_integrals_in_each_variable() {
    let function = |point: &[f64; 3]| -> f64 { 2.0 * point[0] + point[1] * point[2] };
    let point = [1.0, 2.0, 3.0];
    let interval_count = 100;
    let integrator = IterativeMulti::from_parameters(interval_count, IterativeMethod::Booles);

    // closed form in x is x*x + x*y*z
    let x = 0;
    let limits = [0.0, 1.0];
    let expected = 7.0;
    let partial = integrator
        .single_partial_integral(&function, x, &limits, &point)
        .unwrap();
    assert!(f64::abs(partial - expected) < 1e-25);

    // closed form in y is 2.0*x*y + y*y*z/2.0
    let y = 1;
    let limits = [0.0, 2.0];
    let expected = 10.0;
    let partial = integrator
        .single_partial_integral(&function, y, &limits, &point)
        .unwrap();
    assert!(f64::abs(partial - expected) < 0.00001);

    // closed form in z is 2.0*x*z + y*z*z/2.0
    let z = 2;
    let limits = [0.0, 3.0];
    let expected = 15.0;
    let partial = integrator
        .single_partial_integral(&function, z, &limits, &point)
        .unwrap();
    assert!(f64::abs(partial - expected) < 0.00001);
}

#[test]
fn booles_rule_double_integrates_a_line() {
    let line = |x: f64| -> f64 { 6.0 * x };
    let limits = [[0.0, 2.0], [0.0, 2.0]];
    let interval_count = 20;
    let integrator = IterativeSingle::from_parameters(interval_count, IterativeMethod::Booles);

    let expected = 24.0;
    let volume = integrator.double_integral(&line, &limits).unwrap();
    assert!(f64::abs(volume - expected) < 0.00001);
}

#[test]
fn simpsons_rule_integrates_a_line_to_its_closed_form() {
    // closed form of 2x is x*x
    let line = |x: f64| -> f64 { 2.0 * x };
    let limits = [0.0, 2.0];
    let interval_count = 200;
    let integrator = IterativeSingle::from_parameters(interval_count, IterativeMethod::Simpsons);

    let expected = 4.0;
    let area = integrator.single_integral(&line, &limits).unwrap();
    assert!(f64::abs(area - expected) < 0.05);
}

#[test]
fn simpsons_rule_takes_partial_integrals_in_each_variable() {
    let function = |point: &[f64; 3]| -> f64 { 2.0 * point[0] + point[1] * point[2] };
    let point = [1.0, 2.0, 3.0];
    let interval_count = 200;
    let integrator = IterativeMulti::from_parameters(interval_count, IterativeMethod::Simpsons);

    // closed form in x is x*x + x*y*z
    let x = 0;
    let limits = [0.0, 1.0];
    let expected = 7.0;
    let partial = integrator
        .single_partial_integral(&function, x, &limits, &point)
        .unwrap();
    assert!(f64::abs(partial - expected) < 0.05);

    // closed form in y is 2.0*x*y + y*y*z/2.0
    let y = 1;
    let limits = [0.0, 2.0];
    let expected = 10.0;
    let partial = integrator
        .single_partial_integral(&function, y, &limits, &point)
        .unwrap();
    assert!(f64::abs(partial - expected) < 0.05);

    // closed form in z is 2.0*x*z + y*z*z/2.0
    let z = 2;
    let limits = [0.0, 3.0];
    let expected = 15.0;
    let partial = integrator
        .single_partial_integral(&function, z, &limits, &point)
        .unwrap();
    assert!(f64::abs(partial - expected) < 0.05);
}

#[test]
fn simpsons_rule_double_integrates_a_line() {
    let line = |x: f64| -> f64 { 6.0 * x };
    let limits = [[0.0, 2.0], [0.0, 2.0]];
    let interval_count = 200;
    let integrator = IterativeSingle::from_parameters(interval_count, IterativeMethod::Simpsons);

    let expected = 24.0;
    let volume = integrator.double_integral(&line, &limits).unwrap();
    assert!(f64::abs(volume - expected) < 0.05);
}

#[test]
fn simpsons_rule_takes_a_double_partial_integral() {
    let function = |point: &[f64; 3]| -> f64 { 2.0 * point[0] + point[1] * point[2] };
    let point = [1.0, 1.0, 1.0];
    let interval_count = 200;
    let integrator = IterativeMulti::from_parameters(interval_count, IterativeMethod::Simpsons);

    // first in x, then in y
    let x = 0;
    let y = 1;
    let limits = [[0.0, 1.0], [0.0, 1.0]];
    let expected = 1.50;
    let partial = integrator
        .double_partial_integral(&function, [x, y], &limits, &point)
        .unwrap();
    assert!(f64::abs(partial - expected) < 0.05);
}

#[test]
fn trapezoidal_rule_integrates_a_line_to_its_closed_form() {
    // closed form of 2x is x*x
    let line = |x: f64| -> f64 { 2.0 * x };
    let limits = [0.0, 2.0];
    let interval_count = 100;
    let integrator = IterativeSingle::from_parameters(interval_count, IterativeMethod::Trapezoidal);

    let expected = 4.0;
    let area = integrator.single_integral(&line, &limits).unwrap();
    assert!(f64::abs(area - expected) < 0.00001);
}

#[test]
fn trapezoidal_rule_takes_partial_integrals_in_each_variable() {
    let function = |point: &[f64; 3]| -> f64 { 2.0 * point[0] + point[1] * point[2] };
    let point = [1.0, 2.0, 3.0];
    let interval_count = 100;
    let integrator = IterativeMulti::from_parameters(interval_count, IterativeMethod::Trapezoidal);

    // closed form in x is x*x + x*y*z
    let x = 0;
    let limits = [0.0, 1.0];
    let expected = 7.0;
    let partial = integrator
        .single_partial_integral(&function, x, &limits, &point)
        .unwrap();
    assert!(f64::abs(partial - expected) < 0.00001);

    // closed form in y is 2.0*x*y + y*y*z/2.0
    let y = 1;
    let limits = [0.0, 2.0];
    let expected = 10.0;
    let partial = integrator
        .single_partial_integral(&function, y, &limits, &point)
        .unwrap();
    assert!(f64::abs(partial - expected) < 0.00001);

    // closed form in z is 2.0*x*z + y*z*z/2.0
    let z = 2;
    let limits = [0.0, 3.0];
    let expected = 15.0;
    let partial = integrator
        .single_partial_integral(&function, z, &limits, &point)
        .unwrap();
    assert!(f64::abs(partial - expected) < 0.00001);
}

#[test]
fn trapezoidal_rule_double_integrates_a_line() {
    let line = |x: f64| -> f64 { 6.0 * x };
    let limits = [[0.0, 2.0], [0.0, 2.0]];
    let interval_count = 10;
    let integrator = IterativeSingle::from_parameters(interval_count, IterativeMethod::Trapezoidal);

    let expected = 24.0;
    let volume = integrator.double_integral(&line, &limits).unwrap();
    assert!(f64::abs(volume - expected) < 0.00001);
}

#[test]
fn trapezoidal_rule_takes_a_double_partial_integral() {
    let function = |point: &[f64; 3]| -> f64 { 2.0 * point[0] + point[1] * point[2] };
    let point = [1.0, 2.0, 3.0];
    let interval_count = 10;
    let integrator = IterativeMulti::from_parameters(interval_count, IterativeMethod::Trapezoidal);

    // first in x, then in y
    let x = 0;
    let y = 1;
    let limits = [[0.0, 1.0], [0.0, 2.0]];
    let expected = 8.0;
    let partial = integrator
        .double_partial_integral(&function, [x, y], &limits, &point)
        .unwrap();
    assert!(f64::abs(partial - expected) < 0.00001);
}

#[test]
fn reversed_limits_are_rejected() {
    let line = |x: f64| -> f64 { 2.0 * x };

    //lower limit is higher than the upper limit
    let limits = [10.0, 1.0];

    let integrator = IterativeSingle::default();
    let result = integrator.single_integral(&line, &limits);
    assert!(result.is_err());
    assert!(result.unwrap_err() == IntegrateError::LimitsIllDefined);
}

#[test]
fn zero_interval_count_is_rejected() {
    let line = |x: f64| -> f64 { 2.0 * x };
    let limits = [0.0, 1.0];

    let integrator = IterativeSingle::from_parameters(0, IterativeMethod::Booles);
    let result = integrator.single_integral(&line, &limits);
    assert!(result.is_err());
    assert!(result.unwrap_err() == IntegrateError::IterationsZero);
}

//TODO: add more tests

#[test]
fn iterative_rule_surfaces_a_nan_integrand() {
    //the rule must surface the NaN instead of a garbage number
    let nan_integrand = |_x: f64| -> f64 { f64::NAN };
    let limits = [0.0, 1.0];

    let integrator = IterativeSingle::default();
    let result = integrator.single_integral(&nan_integrand, &limits);
    assert!(result.is_err());
    assert!(result.unwrap_err() == IntegrateError::NonFinite);
}

#[test]
fn iterative_rule_integrates_a_bell_curve_over_the_real_line() {
    //∫_{-∞}^∞ e^{-x²} dx = √π
    let bell_curve = |x: f64| -> f64 { f64::exp(-x * x) };
    let integrator = IterativeSingle::default();

    let real_line = [f64::NEG_INFINITY, f64::INFINITY];
    let area = integrator.single_integral(&bell_curve, &real_line).unwrap();

    let expected = core::f64::consts::PI.sqrt();
    assert!(f64::abs(area - expected) < 1e-3);
}

#[test]
fn iterative_rule_integrates_a_decaying_exponential_over_the_half_line() {
    //∫_0^∞ e^{-x} dx = 1
    let decay = |x: f64| -> f64 { f64::exp(-x) };
    let integrator = IterativeSingle::default();

    let half_line = [0.0, f64::INFINITY];
    let area = integrator.single_integral(&decay, &half_line).unwrap();

    let expected = 1.0;
    assert!(f64::abs(area - expected) < 1e-3);
}

#[test]
fn iterative_rule_integrates_an_inverse_square_over_the_half_line() {
    //∫_1^∞ x^{-2} dx = 1
    let inverse_square = |x: f64| -> f64 { 1.0 / (x * x) };
    let integrator = IterativeSingle::default();

    let half_line = [1.0, f64::INFINITY];
    let area = integrator
        .single_integral(&inverse_square, &half_line)
        .unwrap();

    let expected = 1.0;
    assert!(f64::abs(area - expected) < 1e-3);
}

#[test]
fn iterative_rule_accepts_a_negative_lower_limit() {
    //∫_{-2}^{1} 2x dx = x² evaluated from -2 to 1 = 1 - 4 = -3
    let line = |x: f64| -> f64 { 2.0 * x };
    let integrator = IterativeSingle::default();

    let limits = [-2.0, 1.0];
    let area = integrator.single_integral(&line, &limits).unwrap();

    let expected = -3.0;
    assert!(f64::abs(area - expected) < 1e-9);
}

#[test]
fn composite_rules_are_exact_on_a_cubic() {
    //a degree-3 integrand exposes composite-rule divisibility (a linear one would be exact
    //under every rule and hide it). ∫_0^2 x³ dx = 4
    let cubic = |x: f64| -> f64 { x * x * x };
    let limits = [0.0, 2.0];
    let interval_count = 120;
    let expected = 4.0;

    //120 is a multiple of 3, so Simpson's 3/8 is exact for cubics here
    let simpson = IterativeSingle::from_parameters(interval_count, IterativeMethod::Simpsons);
    let area = simpson.single_integral(&cubic, &limits).unwrap();
    assert!(f64::abs(area - expected) < 1e-9);

    //120 is a multiple of 4, so Boole's rule is exact too
    let boole = IterativeSingle::from_parameters(interval_count, IterativeMethod::Booles);
    let area = boole.single_integral(&cubic, &limits).unwrap();
    assert!(f64::abs(area - expected) < 1e-9);
}

//naive left-to-right trapezoidal accumulation, matching the library's point stepping and
//weights so the only difference from the pairwise version is the summation order
fn naive_trapezoidal<F: Fn(f64) -> f64>(
    interval_count: u64,
    lower_limit: f64,
    upper_limit: f64,
    function: F,
) -> f64 {
    let width = (upper_limit - lower_limit) / interval_count as f64;
    let mut point = lower_limit;
    let mut sum = function(point);
    for _ in 0..interval_count - 1 {
        point += width;
        sum += 2.0 * function(point);
    }
    sum += function(upper_limit);
    0.5 * width * sum
}

#[test]
fn pairwise_integration_is_accurate_on_long_sum() {
    //1/(1+x^2) over [0, 1] is exactly pi/4. With 2^23 intervals the trapezoidal truncation
    //error sits at machine epsilon, so naive accumulation (error ~ n*eps) becomes the limiting
    //factor; pairwise keeps the result truncation-limited and lands far closer to the exact value
    let function = |x: f64| -> f64 { 1.0 / (1.0 + x * x) };
    let exact = core::f64::consts::PI / 4.0;
    let interval_count: u64 = 1 << 23;

    let integrator = IterativeSingle::from_parameters(interval_count, IterativeMethod::Trapezoidal);
    let pairwise = integrator.single_integral(&function, &[0.0, 1.0]).unwrap();
    let naive = naive_trapezoidal(interval_count, 0.0, 1.0, function);

    let pairwise_error = f64::abs(pairwise - exact);
    let naive_error = f64::abs(naive - exact);

    //tighter than the ~n*eps a naive sum could reach at this term count
    assert!(
        pairwise_error < 1e-12,
        "pairwise error {pairwise_error:e} too large"
    );
    //and strictly closer to the exact value than the naive accumulation
    assert!(
        pairwise_error < naive_error,
        "pairwise ({pairwise_error:e}) should be closer than naive ({naive_error:e})"
    );
}

#[test]
fn booles_rule_integrates_a_line_at_f32() {
    //2x integrated over [0, 2] is 4
    let line = |x: f32| -> f32 { 2.0 * x };
    let interval_count = 100;
    let integrator =
        IterativeSingle::<f32>::from_parameters(interval_count, IterativeMethod::Booles);

    let expected = 4.0;
    let area = integrator.single_integral(&line, &[0.0, 2.0]).unwrap();
    assert!(f32::abs(area - expected) < 1e-3, "got {area}");
}

#[cfg(feature = "gauss-legendre")]
mod gauss_legendre {
    use super::*;

    #[test]
    fn gaussian_rule_surfaces_an_infinite_integrand() {
        let infinite_integrand = |_x: f64| -> f64 { f64::INFINITY };
        let limits = [0.0, 2.0];

        let integrator = GaussianSingle::default();
        let result = integrator.single_integral(&infinite_integrand, &limits);
        assert!(result.is_err());
        assert!(result.unwrap_err() == IntegrateError::NonFinite);
    }

    #[test]
    fn gaussian_multi_rejects_out_of_range_index() {
        let function = |point: &[f64; 2]| point[0] + point[1];
        let point = [1.0, 2.0];
        let integrator = GaussianMulti::default();
        let error = integrator
            .integrate([2; 1], &function, &[[0.0, 1.0]; 1], &point)
            .unwrap_err();
        assert_eq!(error, IntegrateError::IndexOutOfRange);
    }

    #[test]
    fn gauss_legendre_integrates_a_cubic_exactly() {
        // closed form of 4x³ - 3x² is x^4 - x^3
        let cubic = |x: f64| -> f64 { 4.0 * x * x * x - 3.0 * x * x };
        let limits = [0.0, 2.0];
        let order = 4;
        let integrator =
            GaussianSingle::from_parameters(order, GaussianQuadratureMethod::GaussLegendre);

        let expected = 8.0;
        let area = integrator.single_integral(&cubic, &limits).unwrap();
        assert!(f64::abs(area - expected) < 1e-14);
    }

    #[test]
    fn gauss_legendre_takes_partial_integrals_in_each_variable() {
        let function = |point: &[f64; 3]| -> f64 { 2.0 * point[0] + point[1] * point[2] };
        let point = [1.0, 2.0, 3.0];
        let order = 2;
        let integrator =
            GaussianMulti::from_parameters(order, GaussianQuadratureMethod::GaussLegendre);

        // closed form in x is x*x + x*y*z
        let x = 0;
        let limits = [0.0, 1.0];
        let expected = 7.0;
        let partial = integrator
            .single_partial_integral(&function, x, &limits, &point)
            .unwrap();
        assert!(f64::abs(partial - expected) < 1e-14);

        // closed form in y is 2.0*x*y + y*y*z/2.0
        let y = 1;
        let limits = [0.0, 2.0];
        let expected = 10.0;
        let partial = integrator
            .single_partial_integral(&function, y, &limits, &point)
            .unwrap();
        assert!(f64::abs(partial - expected) < 1e-14);

        // closed form in z is 2.0*x*z + y*z*z/2.0
        let z = 2;
        let limits = [0.0, 3.0];
        let expected = 15.0;
        let partial = integrator
            .single_partial_integral(&function, z, &limits, &point)
            .unwrap();
        assert!(f64::abs(partial - expected) < 1e-14);
    }

    #[test]
    fn gauss_legendre_double_integrates_a_line() {
        let line = |x: f64| -> f64 { 6.0 * x };
        let limits = [[0.0, 2.0], [0.0, 2.0]];
        let order = 2;
        let integrator =
            GaussianSingle::from_parameters(order, GaussianQuadratureMethod::GaussLegendre);

        let expected = 24.0;
        let volume = integrator.double_integral(&line, &limits).unwrap();
        assert!(f64::abs(volume - expected) < 1e-14);
    }

    #[test]
    fn gauss_legendre_rejects_an_order_below_one() {
        let cubic = |x: f64| -> f64 { 4.0 * x * x * x - 3.0 * x * x };
        let limits = [0.0, 2.0];

        let integrator =
            GaussianSingle::from_parameters(0, GaussianQuadratureMethod::GaussLegendre);
        let result = integrator.single_integral(&cubic, &limits);
        assert!(result.is_err());
        assert!(result.unwrap_err() == IntegrateError::QuadratureOrderOutOfRange);
    }

    #[test]
    fn gauss_legendre_rejects_an_order_above_thirty() {
        let cubic = |x: f64| -> f64 { 4.0 * x * x * x - 3.0 * x * x };
        let limits = [0.0, 2.0];

        let integrator =
            GaussianSingle::from_parameters(31, GaussianQuadratureMethod::GaussLegendre);
        let result = integrator.single_integral(&cubic, &limits);
        assert!(result.is_err());
        assert!(result.unwrap_err() == IntegrateError::QuadratureOrderOutOfRange);
    }

    #[test]
    fn gauss_legendre_integrates_a_cubic_at_f32() {
        //4x^3 - 3x^2 integrated over [0, 2] is 8
        let cubic = |x: f32| -> f32 { 4.0 * x * x * x - 3.0 * x * x };
        let order = 4;
        let integrator =
            GaussianSingle::<f32>::from_parameters(order, GaussianQuadratureMethod::GaussLegendre);

        let expected = 8.0;
        let area = integrator.single_integral(&cubic, &[0.0, 2.0]).unwrap();
        assert!(f32::abs(area - expected) < 1e-2, "got {area}");
    }
}

#[cfg(feature = "gauss-hermite")]
mod gauss_hermite {
    use super::*;

    #[test]
    fn gauss_hermite_integrates_over_the_real_line() {
        //integrand is x*x; weights carry the e^{-x*x} kernel
        //∫_{-∞}^∞ x² e^{-x²} dx = √π / 2
        let square = |x: f64| -> f64 { x * x };
        let order = 5;
        let integrator =
            GaussianSingle::from_parameters(order, GaussianQuadratureMethod::GaussHermite);

        let real_line = [f64::NEG_INFINITY, f64::INFINITY];
        let area = integrator.single_integral(&square, &real_line).unwrap();

        let expected = core::f64::consts::PI.sqrt() / 2.0;
        assert!(f64::abs(area - expected) < 1e-10);
    }

    #[test]
    fn gauss_hermite_integrates_a_product_over_the_plane() {
        //∫∫ x² y² e^{-x²} e^{-y²} dx dy = (√π/2)²
        let product_of_squares =
            |point: &[f64; 2]| -> f64 { point[0] * point[0] * point[1] * point[1] };
        let order = 5;
        let integrator =
            GaussianMulti::from_parameters(order, GaussianQuadratureMethod::GaussHermite);

        let limits = [
            [f64::NEG_INFINITY, f64::INFINITY],
            [f64::NEG_INFINITY, f64::INFINITY],
        ];
        let point = [0.0, 0.0];
        let x = 0;
        let y = 1;
        let volume = integrator
            .integrate([x, y], &product_of_squares, &limits, &point)
            .unwrap();

        let sqrt_pi_half = core::f64::consts::PI.sqrt() / 2.0;
        let expected = sqrt_pi_half * sqrt_pi_half;
        assert!(f64::abs(volume - expected) < 1e-10);
    }
}

#[cfg(feature = "gauss-laguerre")]
mod gauss_laguerre {
    use super::*;

    #[test]
    fn gauss_laguerre_integrates_over_the_half_line() {
        //integrand is x*x; weights carry the e^{-x} kernel
        //∫_0^∞ x² e^{-x} dx = 2
        let square = |x: f64| -> f64 { x * x };
        let order = 5;
        let integrator =
            GaussianSingle::from_parameters(order, GaussianQuadratureMethod::GaussLaguerre);

        let half_line = [0.0, f64::INFINITY];
        let area = integrator.single_integral(&square, &half_line).unwrap();

        let expected = 2.0;
        assert!(f64::abs(area - expected) < 1e-9);
    }

    #[test]
    fn gauss_laguerre_integrates_a_product_over_the_quadrant() {
        //∫∫ x² y² e^{-x} e^{-y} dx dy = 2 * 2 = 4
        let product_of_squares =
            |point: &[f64; 2]| -> f64 { point[0] * point[0] * point[1] * point[1] };
        let order = 5;
        let integrator =
            GaussianMulti::from_parameters(order, GaussianQuadratureMethod::GaussLaguerre);

        let limits = [[0.0, f64::INFINITY], [0.0, f64::INFINITY]];
        let point = [0.0, 0.0];
        let x = 0;
        let y = 1;
        let volume = integrator
            .integrate([x, y], &product_of_squares, &limits, &point)
            .unwrap();

        let expected = 4.0;
        assert!(f64::abs(volume - expected) < 1e-8);
    }
}

fn polynomial_coeffs(degree: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(-10.0..10.0, degree + 1)
}

fn integration_limit() -> impl Strategy<Value = [f64; 2]> {
    (-10.0..10.0f64, 0.1..5.0f64).prop_map(|(lower_limit, length)| {
        let upper_limit = lower_limit + length;
        [lower_limit, upper_limit]
    })
}

fn func_from_coeffs(coefficients: &[f64]) -> impl Fn(f64) -> f64 {
    move |x| {
        coefficients
            .iter()
            .enumerate()
            .map(|(degree, coefficient)| coefficient * x.powf(degree as f64))
            .sum()
    }
}

fn closed_form_from_coeffs(coefficients: &[f64], interval: [f64; 2]) -> f64 {
    let [lower_limit, upper_limit] = interval;
    coefficients
        .iter()
        .enumerate()
        .map(|(degree, coefficient)| {
            let exponent = (degree + 1) as f64;
            (upper_limit.powf(exponent) - lower_limit.powf(exponent)) * coefficient / exponent
        })
        .sum()
}

fn tolerance_from_coeffs(coefficients: &[f64], interval: [f64; 2]) -> f64 {
    let [lower_limit, upper_limit] = interval;
    let bound: f64 = coefficients
        .iter()
        .enumerate()
        .map(|(degree, coefficient)| {
            let exponent = (degree + 1) as f64;
            lower_limit
                .powf(exponent)
                .abs()
                .max(upper_limit.powf(exponent).abs())
                * coefficient.abs()
                / exponent
        })
        .sum();

    1e-6 * bound.max(1.0)
}

fn iterative_integration_proptest(
    degree: usize,
    integrator: impl IntegratorSingleVariable<Scalar = f64>,
) {
    proptest!(|(
        limit in integration_limit(),
        coeffs in polynomial_coeffs(degree)
        )| {
        let function = func_from_coeffs(&coeffs);
        let closed_form = closed_form_from_coeffs(&coeffs, limit);
        let tolerance = tolerance_from_coeffs(&coeffs, limit);

        let area = integrator.single_integral(&function, &limit).unwrap();
        prop_assert!(f64::abs(area - closed_form) < tolerance);
    });
}

#[test]
fn proptest_trapezoid_integration_f64() {
    let integrator = IterativeSingle::from_parameters(100, IterativeMethod::Trapezoidal);
    iterative_integration_proptest(1, integrator);
}

#[test]
fn proptest_simpsons_integration_f64() {
    let integrator = IterativeSingle::from_parameters(120, IterativeMethod::Simpsons);
    iterative_integration_proptest(3, integrator);
}

#[test]
fn proptest_booles_integration_f64() {
    let integrator = IterativeSingle::from_parameters(100, IterativeMethod::Booles);
    iterative_integration_proptest(5, integrator);
}

#[cfg(any(
    feature = "gauss-legendre",
    feature = "gauss-hermite",
    feature = "gauss-laguerre"
))]
mod gauss_proptests {
    use super::*;

    fn gauss_coeffs() -> impl Strategy<Value = (usize, Vec<f64>)> {
        (1..10usize).prop_flat_map(|order| (Just(order), polynomial_coeffs(2 * order - 1)))
    }

    #[cfg(feature = "gauss-legendre")]
    fn legendre_moment(degree: usize, [lower_limit, upper_limit]: [f64; 2]) -> f64 {
        (upper_limit.powi(degree as i32 + 1) - lower_limit.powi(degree as i32 + 1))
            / (degree as f64 + 1.0)
    }

    #[cfg(feature = "gauss-hermite")]
    fn double_factorial(value: u64) -> f64 {
        let mut product = 1.0;
        let mut term = value;
        while term > 1 {
            product *= term as f64;
            term -= 2;
        }
        product
    }

    #[cfg(feature = "gauss-hermite")]
    fn hermite_moment(degree: usize) -> f64 {
        if degree % 2 == 1 {
            0.0
        } else {
            let half_degree = degree / 2;
            let double_factorial_term = if half_degree == 0 {
                1.0
            } else {
                double_factorial(2 * half_degree as u64 - 1)
            };
            double_factorial_term * std::f64::consts::PI.sqrt() / 2f64.powi(half_degree as i32)
        }
    }

    #[cfg(feature = "gauss-laguerre")]
    fn laguerre_moment(degree: usize) -> f64 {
        (1..=degree as u64)
            .map(|factor| factor as f64)
            .product::<f64>()
            .max(1.0)
    }

    #[allow(unused)]
    fn gauss_closed_form(
        quadrature: GaussianQuadratureMethod,
        coeffs: &[f64],
        limit: [f64; 2],
    ) -> f64 {
        coeffs
            .iter()
            .enumerate()
            .map(|(degree, coefficient)| {
                coefficient
                    * match quadrature {
                        #[cfg(feature = "gauss-legendre")]
                        GaussianQuadratureMethod::GaussLegendre => legendre_moment(degree, limit),
                        #[cfg(feature = "gauss-hermite")]
                        GaussianQuadratureMethod::GaussHermite => hermite_moment(degree),
                        #[cfg(feature = "gauss-laguerre")]
                        GaussianQuadratureMethod::GaussLaguerre => laguerre_moment(degree),
                        _ => unreachable!(),
                    }
            })
            .sum()
    }

    #[allow(unused, dead_code)]
    fn gauss_tolerance(
        quadrature: GaussianQuadratureMethod,
        coeffs: &[f64],
        limit: [f64; 2],
    ) -> f64 {
        let moment_fn: fn(usize) -> f64 = match quadrature {
            #[cfg(feature = "gauss-legendre")]
            GaussianQuadratureMethod::GaussLegendre => return tolerance_from_coeffs(coeffs, limit),
            #[cfg(feature = "gauss-hermite")]
            GaussianQuadratureMethod::GaussHermite => hermite_moment,
            #[cfg(feature = "gauss-laguerre")]
            GaussianQuadratureMethod::GaussLaguerre => laguerre_moment,
            _ => unreachable!(),
        };

        let term_sum_abs: f64 = coeffs
            .iter()
            .enumerate()
            .map(|(degree, &coefficient)| (coefficient * moment_fn(degree)).abs())
            .sum();

        (term_sum_abs).max(1.0) * 1e-9
    }

    fn gauss_integration_proptest(
        quadrature: GaussianQuadratureMethod,
        limit_strat: impl Strategy<Value = [f64; 2]>,
    ) {
        proptest!(|(
            limit in limit_strat,
            (n, coeffs) in gauss_coeffs(),
            )| {

            let function = func_from_coeffs(&coeffs);
            let closed_form = gauss_closed_form(quadrature, &coeffs, limit);
            let tolerance = gauss_tolerance(quadrature, &coeffs, limit);

            let integrator = GaussianSingle::<f64>::from_parameters(
            n,
            quadrature,
            );

            let area = integrator.single_integral(&function, &limit).unwrap();
            prop_assert!(f64::abs(area - closed_form) < tolerance);
        });
    }

    #[test]
    #[cfg(feature = "gauss-legendre")]
    fn proptest_gauss_legendre_integration_f64() {
        gauss_integration_proptest(GaussianQuadratureMethod::GaussLegendre, integration_limit());
    }

    #[test]
    #[cfg(feature = "gauss-hermite")]
    fn proptest_gauss_hermite_integration_f64() {
        gauss_integration_proptest(
            GaussianQuadratureMethod::GaussHermite,
            Just([-f64::INFINITY, f64::INFINITY]),
        );
    }

    #[test]
    #[cfg(feature = "gauss-laguerre")]
    fn proptest_gauss_laguerre_integration_f64() {
        gauss_integration_proptest(
            GaussianQuadratureMethod::GaussLaguerre,
            Just([0.0, f64::INFINITY]),
        );
    }
}

#[test]
fn kahan_integration_beats_naive_on_long_sum() {
    let function = |x: f64| -> f64 { 1.0 / (1.0 + x * x) };
    let exact = core::f64::consts::PI / 4.0;
    let interval_count: u64 = 1 << 23;

    let integrator = IterativeSingle::from_parameters(interval_count, IterativeMethod::Trapezoidal)
        .with_kahan_summation();

    let kahan = integrator.single_integral(&function, &[0.0, 1.0]).unwrap();
    let naive = naive_trapezoidal(interval_count, 0.0, 1.0, function);

    let kahan_error = f64::abs(kahan - exact);
    let naive_error = f64::abs(naive - exact);
    assert!(kahan_error < 1e-12, "kahan error {kahan_error:e} too large");
    assert!(
        kahan_error < naive_error,
        "kahan ({kahan_error:e}) should be closer than naive ({naive_error:e})"
    );
}

#[test]
fn iterative_multi_rejects_out_of_range_index() {
    let function = |point: &[f64; 2]| point[0] + point[1];
    let point = [1.0, 2.0];
    let integrator = IterativeMulti::default();
    let error = integrator
        .integrate([2; 1], &function, &[[0.0, 1.0]; 1], &point)
        .unwrap_err();
    assert_eq!(error, IntegrateError::IndexOutOfRange);
}
