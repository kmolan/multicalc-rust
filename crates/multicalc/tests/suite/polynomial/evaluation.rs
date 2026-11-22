//! Polynomial tests: evaluation and its derivatives, the arithmetic that grows or shrinks a
//! polynomial, and the calculus, at f32 and f64 and through dual numbers.

use multicalc::error::PolynomialError;
use multicalc::polynomial::Polynomial;
use multicalc::scalar::Dual;

/// A degree-5 polynomial with mixed signs, used wherever a test wants something without symmetry.
const MIXED: [f64; 6] = [2.0, -3.5, 0.75, 4.0, -1.25, 0.5];

// ---- evaluation -------------------------------------------------------------

#[test]
fn evaluate_matches_direct_expansion() {
    let poly = Polynomial::new(MIXED);
    let x = 1.7_f64;

    let mut expected = 0.0;
    for (power, coefficient) in MIXED.iter().enumerate() {
        expected += coefficient * x.powi(power as i32);
    }
    assert!((poly.evaluate(x) - expected).abs() < 1e-12);
}

#[test]
fn evaluate_with_derivatives_matches_repeated_derivative() {
    let poly = Polynomial::new(MIXED);
    let x = 1.7_f64;
    let [value, first, second, third] = poly.evaluate_with_derivatives(x);

    let first_derivative = poly.derivative();
    let second_derivative = first_derivative.derivative();
    let third_derivative = second_derivative.derivative();

    assert!((value - poly.evaluate(x)).abs() < 1e-12);
    assert!((first - first_derivative.evaluate(x)).abs() < 1e-12);
    assert!((second - second_derivative.evaluate(x)).abs() < 1e-12);
    assert!((third - third_derivative.evaluate(x)).abs() < 1e-12);
}

#[test]
fn evaluate_is_accurate_for_f32() {
    // 1 - 2x + 3x², which is 9 at x = 2.
    let poly: Polynomial<3, f32> = Polynomial::new([1.0, -2.0, 3.0]);
    assert!((poly.evaluate(2.0) - 9.0).abs() < 1e-5);
}

#[test]
fn differentiates_through_autodiff() {
    let plain: Polynomial<4> = Polynomial::new([2.0, -3.5, 0.75, 4.0]);
    let dual: Polynomial<4, Dual<f64>> = Polynomial::new([
        Dual::constant(2.0),
        Dual::constant(-3.5),
        Dual::constant(0.75),
        Dual::constant(4.0),
    ]);
    let x = 1.7_f64;

    let evaluated = dual.evaluate(Dual::variable(x));
    assert!((evaluated.value - plain.evaluate(x)).abs() < 1e-12);
    assert!((evaluated.deriv - plain.derivative().evaluate(x)).abs() < 1e-12);
}

// ---- introspection and resizing ---------------------------------------------

#[test]
fn degree_ignores_trailing_zeros() {
    assert_eq!(Polynomial::<4>::new([1.0, 2.0, 0.0, 0.0]).degree(), Some(1));
    assert_eq!(Polynomial::<4>::new([0.0, 0.0, 0.0, 7.0]).degree(), Some(3));
    assert_eq!(Polynomial::<4>::zeros().degree(), None);
}

#[test]
fn is_zero_and_leading_coefficient() {
    let poly: Polynomial<4> = Polynomial::new([1.0, 2.0, 0.0, 0.0]);
    assert!(!poly.is_zero());
    assert_eq!(poly.leading_coefficient(), Some(2.0));

    let zero = Polynomial::<4>::zeros();
    assert!(zero.is_zero());
    assert_eq!(zero.leading_coefficient(), None);
}

#[test]
fn try_resize_grows_and_refuses_to_lose_a_term() {
    // 1 + 2x, with room for two more coefficients.
    let poly: Polynomial<4> = Polynomial::new([1.0, 2.0, 0.0, 0.0]);

    let grown: Polynomial<7> = poly.try_resize().unwrap();
    assert_eq!(grown.coefficients(), &[1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    // Two coefficients still hold every term; one would drop the 2x.
    let shrunk: Polynomial<2> = poly.try_resize().unwrap();
    assert_eq!(shrunk.coefficients(), &[1.0, 2.0]);
    assert!(poly.try_resize::<1>().is_none());
}

// ---- arithmetic -------------------------------------------------------------

#[test]
fn multiply_into_matches_hand_expansion() {
    // (1 + 2x - x²)(3 - x + 4x²) = 3 + 5x - x² + 9x³ - 4x⁴
    let left: Polynomial<3> = Polynomial::new([1.0, 2.0, -1.0]);
    let right: Polynomial<3> = Polynomial::new([3.0, -1.0, 4.0]);

    let product = left.multiply_into::<3, 5>(&right).unwrap();
    assert_eq!(product.coefficients(), &[3.0, 5.0, -1.0, 9.0, -4.0]);

    // Four coefficients cannot hold the x⁴ term.
    assert_eq!(
        left.multiply_into::<3, 4>(&right),
        Err(PolynomialError::DegreeOverflow)
    );
}

#[test]
fn divide_then_multiply_returns_the_original() {
    // x⁴ - 2x³ + 3x - 5, over x² + 1.
    let poly: Polynomial<5> = Polynomial::new([-5.0, 3.0, 0.0, -2.0, 1.0]);
    let divisor: Polynomial<3> = Polynomial::new([1.0, 0.0, 1.0]);

    let (quotient, remainder) = poly.divide::<3, 5, 3>(&divisor).unwrap();
    let rebuilt =
        quotient.multiply_into::<3, 5>(&divisor).unwrap() + remainder.try_resize::<5>().unwrap();

    for (returned, original) in rebuilt
        .coefficients()
        .iter()
        .zip(poly.coefficients().iter())
    {
        assert!((returned - original).abs() < 1e-12);
    }
}

#[test]
fn compose_into_matches_evaluating_twice() {
    let outer: Polynomial<3> = Polynomial::new([2.0, -1.0, 0.5]);
    let inner: Polynomial<2> = Polynomial::new([1.0, 3.0]);

    let composed = outer.compose_into::<2, 3>(&inner).unwrap();
    for x in [-1.4, 0.0, 0.75, 2.6] {
        let expected = outer.evaluate(inner.evaluate(x));
        assert!((composed.evaluate(x) - expected).abs() < 1e-12);
    }
}

#[test]
fn shift_argument_matches_evaluating_shifted() {
    let poly: Polynomial<4> = Polynomial::new([2.0, -3.5, 0.75, 4.0]);
    let offset = 1.3;

    let shifted = poly.shift_argument(offset);
    for x in [-2.0, 0.0, 0.4, 3.1] {
        assert!((shifted.evaluate(x) - poly.evaluate(x + offset)).abs() < 1e-12);
    }
}

#[test]
fn scale_argument_matches_evaluating_scaled() {
    let poly: Polynomial<4> = Polynomial::new([2.0, -3.5, 0.75, 4.0]);
    let factor = -0.8;

    let scaled = poly.scale_argument(factor);
    for x in [-2.0, 0.0, 0.4, 3.1] {
        assert!((scaled.evaluate(x) - poly.evaluate(factor * x)).abs() < 1e-12);
    }
}

#[test]
fn reverse_turns_roots_into_reciprocals() {
    // 2 - 3x + x², whose roots are 1 and 2.
    let poly: Polynomial<3> = Polynomial::new([2.0, -3.0, 1.0]);

    let reversed = poly.reverse();
    assert!(reversed.evaluate(1.0).abs() < 1e-10);
    assert!(reversed.evaluate(0.5).abs() < 1e-10);
}

// ---- calculus ---------------------------------------------------------------

#[test]
fn derivative_matches_finite_difference() {
    let poly = Polynomial::new(MIXED);
    let x = 1.7_f64;
    let step = 1e-5;

    let approximate = (poly.evaluate(x + step) - poly.evaluate(x - step)) / (2.0 * step);
    assert!((poly.derivative().evaluate(x) - approximate).abs() < 1e-6);
}

#[test]
fn derivative_leaves_top_coefficient_zero() {
    let poly: Polynomial<4> = Polynomial::new([1.0, 2.0, 3.0, 4.0]);
    assert_eq!(poly.derivative().coefficient(3), Some(0.0));
}

#[test]
fn nth_derivative_past_the_degree_is_zero() {
    // 1 + 2x + 3x² + 4x³, whose third derivative is 24.
    let poly: Polynomial<4> = Polynomial::new([1.0, 2.0, 3.0, 4.0]);

    assert_eq!(poly.nth_derivative(0), poly);
    assert_eq!(
        poly.nth_derivative(3).coefficients(),
        &[24.0, 0.0, 0.0, 0.0]
    );
    assert!(poly.nth_derivative(4).is_zero());
    assert!(poly.nth_derivative(100).is_zero());
}

#[test]
fn definite_integral_matches_known_areas() {
    // 2 - 3.5x + 0.75x² + 4x³, whose area from 0 to 1 is 2 - 1.75 + 0.25 + 1.
    let poly: Polynomial<5> = Polynomial::new([2.0, -3.5, 0.75, 4.0, 0.0]);
    assert!((poly.definite_integral(0.0, 1.0) - 1.5).abs() < 1e-12);

    // Covering the range in two pieces gives the same area as covering it in one.
    let (lower, split, upper) = (-0.6, 1.0, 2.3);
    let whole = poly.definite_integral(lower, upper);
    let pieces = poly.definite_integral(lower, split) + poly.definite_integral(split, upper);
    assert!((whole - pieces).abs() < 1e-12);

    // Swapping the ends flips the sign.
    assert!((poly.definite_integral(upper, lower) + whole).abs() < 1e-12);

    // 3x² from 0 to 2 covers 8.
    let three_x_squared: Polynomial<3> = Polynomial::new([0.0, 0.0, 3.0]);
    assert!((three_x_squared.definite_integral(0.0, 2.0) - 8.0).abs() < 1e-12);
}
