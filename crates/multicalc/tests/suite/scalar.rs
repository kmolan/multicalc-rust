// Tests for the scalar number system: Dual, HyperDual, Jet, and the function abstraction.

mod numeric_methods {
    use multicalc::scalar::Numeric;

    const TOL: f64 = 1e-12;

    #[test]
    fn f64_value_goldens() {
        // atan2 across quadrants and the axes, vs std.
        for &(y, x) in &[
            (-2.0_f64, 3.0),
            (1.0, -1.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (-1.0, 0.0),
        ] {
            assert!((Numeric::atan2(y, x) - y.atan2(x)).abs() < TOL);
        }
        assert_eq!(Numeric::copysign(3.0_f64, -1.0), -3.0);
        assert_eq!(Numeric::copysign(-3.0_f64, 1.0), 3.0);
        assert_eq!(Numeric::floor(2.7_f64), 2.0);
        assert_eq!(Numeric::floor(-2.1_f64), -3.0);
        assert!((Numeric::asin(0.5_f64) - 0.5_f64.asin()).abs() < TOL);
        assert!((Numeric::acos(0.5_f64) - 0.5_f64.acos()).abs() < TOL);
        assert!((Numeric::atan(0.7_f64) - 0.7_f64.atan()).abs() < TOL);
        assert!((Numeric::sinh(1.1_f64) - 1.1_f64.sinh()).abs() < TOL);
        assert!((Numeric::cosh(1.1_f64) - 1.1_f64.cosh()).abs() < TOL);
        assert!((Numeric::tanh(1.1_f64) - 1.1_f64.tanh()).abs() < TOL);
        assert!((Numeric::hypot(3.0_f64, 4.0) - 5.0).abs() < TOL);
        assert!((Numeric::powf(2.0_f64, 3.5) - 2.0_f64.powf(3.5)).abs() < TOL);
        assert!((Numeric::exp(3.0_f64) - 3.0_f64.exp()).abs() < TOL);
        assert!((Numeric::expm1(0.03_f64) - f64::exp_m1(0.03)).abs() < TOL);
        assert!((Numeric::ln(7.0_f64) - 7.0_f64.ln()).abs() < TOL);
        assert!((Numeric::ln_1p(0.05_f64) - 0.05_f64.ln_1p()).abs() < TOL);
        assert!((Numeric::log2(7.0_f64) - 7.0_f64.log2()).abs() < TOL);
        assert!((Numeric::log10(7.0_f64) - 7.0_f64.log10()).abs() < TOL);
        assert_eq!(Numeric::mul_add(2.0_f64, 3.0, 1.0), 7.0);
        assert_eq!(Numeric::recip(4.0_f64), 0.25);
        assert_eq!(Numeric::signum(-3.0_f64), -1.0);
        assert!(Numeric::signum(f64::NAN).is_nan());
        assert_eq!(Numeric::max(2.0_f64, 5.0), 5.0);
        assert_eq!(Numeric::min(2.0_f64, 5.0), 2.0);
        // max/min return the non-NaN operand, matching the primitive f64 methods.
        assert_eq!(Numeric::max(f64::NAN, 5.0), 5.0);
        assert_eq!(Numeric::min(5.0, f64::NAN), 5.0);
    }

    #[test]
    fn hypot_no_overflow() {
        // naive sqrt(a²+b²) would overflow; the scaled form must stay finite and correct.
        let big = 1e200_f64;
        let hypotenuse = Numeric::hypot(big, big);
        assert!(hypotenuse.is_finite());
        assert!((hypotenuse - big * 2.0_f64.sqrt()).abs() / hypotenuse < 1e-12);
    }

    #[test]
    fn f32_identities() {
        // f32 checked via identities, not value goldens.
        let x = 0.6_f32;
        assert!(
            (Numeric::cosh(x) * Numeric::cosh(x) - Numeric::sinh(x) * Numeric::sinh(x) - 1.0).abs()
                < 1e-4
        ); // cosh² − sinh² = 1
        assert!((Numeric::tanh(x) - Numeric::sinh(x) / Numeric::cosh(x)).abs() < 1e-5);
        assert!((Numeric::powf(3.0_f32, 2.0) - 9.0).abs() < 1e-3);
        // cos(atan2(y, x))·hypot(y, x) = x
        let opposite = 0.7_f32;
        let adjacent = 1.3_f32;
        assert!(
            (Numeric::atan2(opposite, adjacent).cos() * Numeric::hypot(opposite, adjacent)
                - adjacent)
                .abs()
                < 1e-4
        );
    }

    #[test]
    fn edge_cases() {
        // atan2 at the origin matches float atan2; no panic.
        assert_eq!(Numeric::atan2(0.0_f64, 0.0), 0.0_f64.atan2(0.0));
        // asin/acos outside [-1, 1] yield NaN rather than panicking.
        assert!(Numeric::asin(2.0_f64).is_nan());
        assert!(Numeric::acos(-2.0_f64).is_nan());
        // degenerate hypot and zero-exponent powf.
        assert_eq!(Numeric::hypot(0.0_f64, 0.0), 0.0);
        assert_eq!(Numeric::powf(5.0_f64, 0.0), 1.0);
        // copysign carries a signed zero.
        assert!(Numeric::copysign(0.0_f64, -1.0).is_sign_negative());
        // floor on an exact integer and a negative.
        assert_eq!(Numeric::floor(-3.0_f64), -3.0);
        assert_eq!(Numeric::floor(-2.1_f64), -3.0);
    }
}

mod primal {
    use multicalc::scalar::Primal;
    use multicalc::{Dual, HyperDual, Jet};

    #[test]
    fn projects_floats_and_autodiff_primals() {
        assert_eq!(2.0_f64.to_f64(), 2.0);
        assert_eq!(2.0_f32.to_f64(), 2.0);

        assert_eq!(Dual::new(2.0, 99.0).to_f64(), 2.0);
        assert_eq!(HyperDual::new(2.0, 1.0, 1.0, 1.0).to_f64(), 2.0);
        assert_eq!(Jet::<f64, 2>::variable(2.0).to_f64(), 2.0);

        let nested = Dual::new(
            HyperDual::new(2.0, 1.0, 1.0, 1.0),
            HyperDual::new(99.0, 3.0, 3.0, 3.0),
        );
        assert_eq!(nested.to_f64(), 2.0);
        assert_eq!(nested.to_f32(), 2.0);
    }
}

mod dual {
    use multicalc::scalar::Dual;
    use multicalc::scalar::Numeric;

    // Dual results are exact to rounding, so the tolerances are tight.
    const TOL: f64 = 1e-12;
    const TOL_F32: f32 = 1e-5;

    #[test]
    fn powi_carries_the_power_rule() {
        // f(x) = x^3, f'(x) = 3x^2; at x = 2 -> 8 and 12
        let cubed = Dual::variable(2.0_f64).powi(3);
        assert!(f64::abs(cubed.value - 8.0) < TOL);
        assert!(f64::abs(cubed.deriv - 12.0) < TOL);
    }

    #[test]
    fn a_sum_of_terms_differentiates_termwise() {
        // f(x) = 3x^2 + 2x, f'(x) = 6x + 2; at x = 2 -> 16 and 14
        let x = Dual::variable(2.0_f64);
        let sum = Dual::constant(3.0) * x * x + Dual::constant(2.0) * x;
        assert!(f64::abs(sum.value - 16.0) < TOL);
        assert!(f64::abs(sum.deriv - 14.0) < TOL);
    }

    #[test]
    fn a_negative_power_differentiates_correctly() {
        // f(x) = x^-2, f'(x) = -2 x^-3; at x = 2 -> 0.25 and -0.25
        let inverse_square = Dual::variable(2.0_f64).powi(-2);
        assert!(f64::abs(inverse_square.value - 0.25) < TOL);
        assert!(f64::abs(inverse_square.deriv - (-0.25)) < TOL);
    }

    #[test]
    fn sqrt_value_and_derivative() {
        // f(x) = sqrt(x), f'(x) = 1/(2 sqrt(x)); at x = 4 -> 2 and 0.25
        let root = Dual::variable(4.0_f64).sqrt();
        assert!(f64::abs(root.value - 2.0) < TOL);
        assert!(f64::abs(root.deriv - 0.25) < TOL);
    }

    #[test]
    fn sin_cos_and_tan_derivatives() {
        let x = 0.7_f64;
        let sine = Dual::variable(x).sin();
        assert!(f64::abs(sine.value - f64::sin(x)) < TOL);
        assert!(f64::abs(sine.deriv - f64::cos(x)) < TOL);

        let cosine = Dual::variable(x).cos();
        assert!(f64::abs(cosine.value - f64::cos(x)) < TOL);
        assert!(f64::abs(cosine.deriv - (-f64::sin(x))) < TOL);

        let tangent = Dual::variable(x).tan();
        assert!(f64::abs(tangent.value - f64::tan(x)) < TOL);
        assert!(f64::abs(tangent.deriv - (1.0 + f64::tan(x) * f64::tan(x))) < TOL);
    }

    #[test]
    fn exp_and_ln_derivatives() {
        let x = 1.3_f64;
        let exponential = Dual::variable(x).exp();
        assert!(f64::abs(exponential.value - f64::exp(x)) < TOL);
        assert!(f64::abs(exponential.deriv - f64::exp(x)) < TOL);

        let x = 0.03_f64;
        let em1 = Dual::variable(x).expm1();
        assert!(f64::abs(em1.value - f64::exp_m1(x)) < TOL);
        assert!(f64::abs(em1.deriv - f64::exp(x)) < TOL);

        // f(x) = ln(x), f'(x) = 1/x; at x = 2 -> ln 2 and 0.5
        let logarithm = Dual::variable(2.0_f64).ln();
        assert!(f64::abs(logarithm.value - f64::ln(2.0)) < TOL);
        assert!(f64::abs(logarithm.deriv - 0.5) < TOL);

        // f(x) = ln(1 + x), f'(x) = 1/(1 + x); at x = -0.2 -> ln 0.8 and 1.25
        let logarithm = Dual::variable(-0.2_f64).ln_1p();
        assert!(f64::abs(logarithm.value - f64::ln(0.8)) < TOL);
        assert!(f64::abs(logarithm.deriv - 1.25) < TOL);

        // f(x) = log10(x), f'(x) = 1/(ln(10) * x); at x = 7 -> log10 7 and 1/(7 * ln(10))
        let log10 = Dual::variable(7.0).log10();
        assert!(f64::abs(log10.value - f64::log10(7.0)) < TOL);
        assert!(f64::abs(log10.deriv - (7.0 * 10.0.ln()).recip()) < TOL);

        // f(x) = log2(x), f'(x) = 1/(ln(2) * x); at x = 7 -> log2 7 and 1/(7 * ln(2))
        let log2 = Dual::variable(7.0).log2();
        assert!(f64::abs(log2.value - f64::log2(7.0)) < TOL);
        assert!(f64::abs(log2.deriv - (7.0 * 2.0.ln()).recip()) < TOL);
    }

    #[test]
    fn the_chain_rule_holds_through_exp_of_sin() {
        // f(x) = exp(sin(x)), f'(x) = cos(x) exp(sin(x))
        let x = 0.6_f64;
        let chained = Dual::variable(x).sin().exp();
        assert!(f64::abs(chained.value - f64::exp(f64::sin(x))) < TOL);
        assert!(f64::abs(chained.deriv - f64::cos(x) * f64::exp(f64::sin(x))) < TOL);
    }

    #[test]
    fn a_rational_function_uses_the_quotient_rule() {
        // f(x) = x / (1 + x^2), f'(x) = (1 - x^2) / (1 + x^2)^2
        let x = 1.5_f64;
        let variable = Dual::variable(x);
        let rational = variable / (Dual::constant(1.0) + variable * variable);
        let denominator = (1.0 + x * x) * (1.0 + x * x);
        assert!(f64::abs(rational.value - x / (1.0 + x * x)) < TOL);
        assert!(f64::abs(rational.deriv - (1.0 - x * x) / denominator) < TOL);
    }

    #[test]
    fn a_product_uses_the_product_rule() {
        // f(x) = sin(x) cos(x), f'(x) = cos^2(x) - sin^2(x)
        let x = 0.9_f64;
        let variable = Dual::variable(x);
        let product = variable.sin() * variable.cos();
        assert!(f64::abs(product.value - f64::sin(x) * f64::cos(x)) < TOL);
        let expected = f64::cos(x) * f64::cos(x) - f64::sin(x) * f64::sin(x);
        assert!(f64::abs(product.deriv - expected) < TOL);
    }

    #[test]
    fn abs_differentiates_to_the_sign() {
        // derivative of |x| is +1 for x > 0 and -1 for x < 0
        let positive = Dual::variable(2.0_f64).abs();
        assert!(f64::abs(positive.value - 2.0) < TOL);
        assert!(f64::abs(positive.deriv - 1.0) < TOL);

        let negative = Dual::variable(-2.0_f64).abs();
        assert!(f64::abs(negative.value - 2.0) < TOL);
        assert!(f64::abs(negative.deriv - (-1.0)) < TOL);
    }

    #[test]
    fn max_and_min_carry_the_selected_branch_derivative() {
        // max/min pick a branch by value and carry that branch's derivative.
        let variable = Dual::variable(3.0_f64); // value 3, deriv 1
        let constant = Dual::constant(5.0_f64); // value 5, deriv 0
        let higher = variable.max(constant);
        assert!(f64::abs(higher.value - 5.0) < TOL && f64::abs(higher.deriv) < TOL);
        let lower = variable.min(constant);
        assert!(f64::abs(lower.value - 3.0) < TOL && f64::abs(lower.deriv - 1.0) < TOL);
    }

    #[test]
    fn generic_over_numeric() {
        // The same function runs with a plain float or with a Dual.
        fn poly<T: Numeric>(t: T) -> T {
            t.powi(3) + T::from_f64(2.0) * t
        }

        let x = 1.7_f64;
        let plain = poly(x);
        let dual = poly(Dual::variable(x));
        assert!(f64::abs(dual.value - plain) < TOL);
        // f'(x) = 3x^2 + 2
        assert!(f64::abs(dual.deriv - (3.0 * x * x + 2.0)) < TOL);
    }

    #[test]
    fn seeding_one_variable_gives_its_partial_derivative() {
        // f(x, y) = x^2 * y + sin(x)
        fn function<T: Numeric>(point: &[T; 2]) -> T {
            point[0] * point[0] * point[1] + point[0].sin()
        }

        let x = 1.0_f64;
        let y = 2.0_f64;

        // seed x: df/dx = 2xy + cos(x)
        let partial_x = function(&[Dual::variable(x), Dual::constant(y)]).deriv;
        assert!(f64::abs(partial_x - (2.0 * x * y + f64::cos(x))) < TOL);

        // seed y: df/dy = x^2
        let partial_y = function(&[Dual::constant(x), Dual::variable(y)]).deriv;
        assert!(f64::abs(partial_y - x * x) < TOL);
    }

    #[test]
    fn generic_over_f32() {
        // Dual is generic over the scalar; here it carries f32.
        let cubed = Dual::variable(2.0_f32).powi(3);
        assert!(f32::abs(cubed.value - 8.0) < TOL_F32);
        assert!(f32::abs(cubed.deriv - 12.0) < TOL_F32);
    }

    #[test]
    fn the_zeroth_power_has_no_derivative() {
        // x^0 = 1, derivative 0
        let zeroth_power = Dual::variable(3.0_f64).powi(0);
        assert!(f64::abs(zeroth_power.value - 1.0) < TOL);
        assert!(f64::abs(zeroth_power.deriv) < TOL);
    }

    #[test]
    fn a_constant_has_zero_derivative() {
        // a constant carries no derivative through any operation
        let constant = Dual::constant(1.3_f64);
        let combined = constant.exp() * constant.sin() + constant.powi(2);
        assert!(f64::abs(combined.deriv) < TOL);
    }

    #[test]
    fn sqrt_at_zero_has_an_infinite_derivative() {
        // the derivative of sqrt at 0 is unbounded, while the value stays finite
        let root = Dual::variable(0.0_f64).sqrt();
        assert!(f64::abs(root.value) < TOL);
        assert!(root.deriv.is_infinite());
        // is_finite reflects the value only, so it still reports finite here
        assert!(root.is_finite());
    }

    #[test]
    fn ln_at_zero_blows_up() {
        // ln(0) = -inf with an unbounded derivative
        let logarithm = Dual::variable(0.0_f64).ln();
        assert!(logarithm.value.is_infinite() && logarithm.value < 0.0);
        assert!(logarithm.deriv.is_infinite() && logarithm.deriv > 0.0);
    }

    #[test]
    fn atan2_derivative_in_its_first_argument() {
        // f(y) = atan2(y, x), x constant: f'(y) = x/(x²+y²)
        let y = 1.0_f64;
        let x = 2.0_f64;
        let angle = Dual::variable(y).atan2(Dual::constant(x));
        assert!(f64::abs(angle.value - y.atan2(x)) < TOL);
        assert!(f64::abs(angle.deriv - x / (x * x + y * y)) < TOL);
    }

    #[test]
    fn inverse_trig_derivatives() {
        let x = 0.3_f64;
        let arcsine = Dual::variable(x).asin();
        assert!(f64::abs(arcsine.value - x.asin()) < TOL);
        assert!(f64::abs(arcsine.deriv - 1.0 / (1.0 - x * x).sqrt()) < TOL); // asin' = 1/√(1−x²)
        let arccosine = Dual::variable(x).acos();
        assert!(f64::abs(arccosine.deriv + 1.0 / (1.0 - x * x).sqrt()) < TOL); // acos' = −asin'
        let arctangent = Dual::variable(x).atan();
        assert!(f64::abs(arctangent.deriv - 1.0 / (1.0 + x * x)) < TOL); // atan' = 1/(1+x²)
    }

    #[test]
    fn hyperbolic_derivatives() {
        let x = 0.8_f64;
        let sinh = Dual::variable(x).sinh();
        assert!(f64::abs(sinh.deriv - x.cosh()) < TOL); // sinh' = cosh
        let cosh = Dual::variable(x).cosh();
        assert!(f64::abs(cosh.deriv - x.sinh()) < TOL); // cosh' = sinh
        let tanh = Dual::variable(x).tanh();
        assert!(f64::abs(tanh.deriv - (1.0 - x.tanh() * x.tanh())) < 1e-10); // tanh' = 1−tanh²
    }

    #[test]
    fn powf_recip_and_hypot_derivatives() {
        let x = 2.0_f64;
        let power = Dual::variable(x).powf(Dual::constant(3.5)); // d/dx x^3.5 = 3.5 x^2.5
        assert!(f64::abs(power.deriv - 3.5 * x.powf(2.5)) < 1e-10);
        let reciprocal = Dual::variable(x).recip(); // d/dx 1/x = −1/x²
        assert!(f64::abs(reciprocal.deriv + 0.25) < TOL);
        // hypot(x, 4) with x variable: d/dx = x/hypot
        let hypotenuse = Dual::variable(3.0_f64).hypot(Dual::constant(4.0));
        assert!(f64::abs(hypotenuse.value - 5.0) < TOL);
        assert!(f64::abs(hypotenuse.deriv - 3.0 / 5.0) < TOL);
    }

    #[test]
    fn floor_has_zero_derivative() {
        let floored = Dual::variable(2.7_f64).floor();
        assert!(f64::abs(floored.value - 2.0) < TOL);
        assert!(f64::abs(floored.deriv) < TOL);
    }

    #[test]
    fn copysign_carries_the_sign_into_the_derivative() {
        let same = Dual::variable(3.0_f64).copysign(Dual::constant(1.0));
        assert!(f64::abs(same.value - 3.0) < TOL && f64::abs(same.deriv - 1.0) < TOL);
        let flip = Dual::variable(3.0_f64).copysign(Dual::constant(-1.0));
        assert!(f64::abs(flip.value + 3.0) < TOL && f64::abs(flip.deriv + 1.0) < TOL);
    }

    #[test]
    fn atan2_derivatives_match_the_closed_form_across_the_plane() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        // d/dy atan2(y, x) = x/(x²+y²); d/dx atan2(y, x) = −y/(x²+y²). Sweep both, all
        // quadrants and near the axes, comparing AD to the closed form.
        let mut rng = StdRng::seed_from_u64(0xA7A2);
        for _ in 0..1000 {
            let y: f64 = rng.gen_range(-5.0..5.0);
            let x: f64 = rng.gen_range(-5.0..5.0);
            let denominator = x * x + y * y;
            if denominator < 1e-6 {
                continue; // origin is the documented singularity
            }
            // seed y (x constant): expect x/denominator
            let seeded_y = Dual::variable(y).atan2(Dual::constant(x));
            assert!(f64::abs(seeded_y.value - y.atan2(x)) < TOL);
            assert!(f64::abs(seeded_y.deriv - x / denominator) < 1e-9);
            // seed x (y constant): expect −y/denominator
            let seeded_x = Dual::constant(y).atan2(Dual::variable(x));
            assert!(f64::abs(seeded_x.deriv + y / denominator) < 1e-9);
        }
    }
}

mod hyper_dual {
    use multicalc::scalar::HyperDual;
    use multicalc::scalar::Numeric;

    // Hyper-dual results are exact to rounding, so the tolerances are tight.
    const TOL: f64 = 1e-12;
    const TOL_F32: f32 = 1e-3;

    #[test]
    fn a_cubic_carries_both_derivative_directions() {
        // f(x) = x^3 -> f'(x) = 3x^2, f''(x) = 6x; at x = 3: 27, 27, 18
        let cubed = HyperDual::variable(3.0_f64).powi(3);
        assert!(f64::abs(cubed.real - 27.0) < TOL);
        assert!(f64::abs(cubed.eps1 - 27.0) < TOL);
        assert!(f64::abs(cubed.eps2 - 27.0) < TOL);
        assert!(f64::abs(cubed.eps1eps2 - 18.0) < TOL);
    }

    #[test]
    fn powi_second_derivative() {
        // f(x) = x^4 -> f'(x) = 4x^3, f''(x) = 12x^2; at x = 2: 16, 32, 48
        let quartic = HyperDual::variable(2.0_f64).powi(4);
        assert!(f64::abs(quartic.real - 16.0) < TOL);
        assert!(f64::abs(quartic.eps1 - 32.0) < TOL);
        assert!(f64::abs(quartic.eps1eps2 - 48.0) < TOL);
    }

    #[test]
    fn sin_second_derivative() {
        // f(x) = sin(x) -> f' = cos(x), f'' = -sin(x)
        let x = 0.7_f64;
        let sine = HyperDual::variable(x).sin();
        assert!(f64::abs(sine.real - f64::sin(x)) < TOL);
        assert!(f64::abs(sine.eps1 - f64::cos(x)) < TOL);
        assert!(f64::abs(sine.eps1eps2 - (-f64::sin(x))) < TOL);
    }

    #[test]
    fn log_second_derivative() {
        // f(x) = ln(x) -> f' = 1/x, f'' = -1/x^2
        let x = 7.0;
        let ln = HyperDual::variable(x).ln();
        assert!(f64::abs(ln.real - f64::ln(x)) < TOL);
        assert!(f64::abs(ln.eps1 - x.recip()) < TOL);
        assert!(f64::abs(ln.eps1eps2 - (-1.0 / (x * x))) < TOL);

        // f(x) = log2(x) -> f' = 1/(ln2 * x), f'' = -1/(ln2 * x^2)
        let ln2 = 2.0.ln();
        let log2 = HyperDual::variable(x).log2();
        assert!(f64::abs(log2.real - f64::log2(x)) < TOL);
        assert!(f64::abs(log2.eps1 - x.recip() / ln2) < TOL);
        assert!(f64::abs(log2.eps1eps2 - (-1.0 / (ln2 * x * x))) < TOL);

        // f(x) = log10(x) -> f' = 1/(ln10 * x), f'' = -1/(ln10 * x^2)
        let ln10 = 10.0.ln();
        let log10 = HyperDual::variable(x).log10();
        assert!(f64::abs(log10.real - f64::log10(x)) < TOL);
        assert!(f64::abs(log10.eps1 - x.recip() / ln10) < TOL);
        assert!(f64::abs(log10.eps1eps2 - (-1.0 / (ln10 * x * x))) < TOL);

        // f(x) = ln(1 + x) -> f' = 1/(1 + x), f'' = -1/(1 + x)^2
        let x = 0.03;
        let xp1 = x + 1.0;
        let ln = HyperDual::variable(x).ln_1p();
        assert!(f64::abs(ln.real - f64::ln_1p(x)) < TOL);
        assert!(f64::abs(ln.eps1 - xp1.recip()) < TOL);
        assert!(f64::abs(ln.eps1eps2 - (-1.0 / (xp1 * xp1))) < TOL);
    }

    #[test]
    fn exp_is_its_own_second_derivative() {
        // f(x) = exp(x) is its own derivative to all orders
        let x = 1.3_f64;
        let exponential = HyperDual::variable(x).exp();
        assert!(f64::abs(exponential.real - f64::exp(x)) < TOL);
        assert!(f64::abs(exponential.eps1 - f64::exp(x)) < TOL);
        assert!(f64::abs(exponential.eps1eps2 - f64::exp(x)) < TOL);

        // f(x) = exp(x) - 1 derivatives match those of exp(x)
        let x = 0.03_f64;
        let exponential = HyperDual::variable(x).expm1();
        assert!(f64::abs(exponential.real - f64::exp_m1(x)) < TOL);
        assert!(f64::abs(exponential.eps1 - f64::exp(x)) < TOL);
        assert!(f64::abs(exponential.eps1eps2 - f64::exp(x)) < TOL);
    }

    #[test]
    fn reciprocal_second_derivative() {
        // f(x) = 1/x -> f' = -1/x^2, f'' = 2/x^3; at x = 2: 0.5, -0.25, 0.25
        let reciprocal = HyperDual::constant(1.0_f64) / HyperDual::variable(2.0_f64);
        assert!(f64::abs(reciprocal.real - 0.5) < TOL);
        assert!(f64::abs(reciprocal.eps1 - (-0.25)) < TOL);
        assert!(f64::abs(reciprocal.eps1eps2 - 0.25) < TOL);
    }

    #[test]
    fn seeding_two_directions_gives_the_full_hessian() {
        // f(x, y) = x^2 * y + sin(x)
        // grad  = [2xy + cos x, x^2]
        // H     = [[2y - sin x, 2x], [2x, 0]]
        fn function<T: Numeric>(point: &[T; 2]) -> T {
            point[0] * point[0] * point[1] + point[0].sin()
        }

        let x = 1.0_f64;
        let y = 2.0_f64;

        // diagonal Hxx and the x-gradient: seed x on both directions, y constant
        let seeded_x = function(&[HyperDual::variable(x), HyperDual::constant(y)]);
        assert!(f64::abs(seeded_x.eps1 - (2.0 * x * y + f64::cos(x))) < TOL); // df/dx
        assert!(f64::abs(seeded_x.eps1eps2 - (2.0 * y - f64::sin(x))) < TOL); // d2f/dx2

        // diagonal Hyy and the y-gradient: seed y on both directions, x constant
        let seeded_y = function(&[HyperDual::constant(x), HyperDual::variable(y)]);
        assert!(f64::abs(seeded_y.eps1 - x * x) < TOL); // df/dy
        assert!(f64::abs(seeded_y.eps1eps2) < TOL); // d2f/dy2 = 0

        // mixed Hxy: seed x on direction 1, y on direction 2
        let mixed = function(&[
            HyperDual::new(x, 1.0, 0.0, 0.0),
            HyperDual::new(y, 0.0, 1.0, 0.0),
        ]);
        assert!(f64::abs(mixed.eps1eps2 - 2.0 * x) < TOL);

        // symmetry: swapping the directions gives the same mixed partial
        let mixed_swapped = function(&[
            HyperDual::new(x, 0.0, 1.0, 0.0),
            HyperDual::new(y, 1.0, 0.0, 0.0),
        ]);
        assert!(f64::abs(mixed.eps1eps2 - mixed_swapped.eps1eps2) < TOL);
    }

    #[test]
    fn generic_over_numeric() {
        // The same function runs with a plain float or with a HyperDual.
        fn polynomial<T: Numeric>(t: T) -> T {
            t.powi(3) + T::from_f64(2.0) * t
        }

        let x = 1.7_f64;
        let plain = polynomial(x);
        let hyper_dual = polynomial(HyperDual::variable(x));
        assert!(f64::abs(hyper_dual.real - plain) < TOL);
        assert!(f64::abs(hyper_dual.eps1 - (3.0 * x * x + 2.0)) < TOL); // f'
        assert!(f64::abs(hyper_dual.eps1eps2 - 6.0 * x) < TOL); // f''
    }

    #[test]
    fn generic_over_f32() {
        // HyperDual is generic over the scalar; here it carries f32.
        let quartic = HyperDual::variable(2.0_f32).powi(4);
        assert!(f32::abs(quartic.real - 16.0) < TOL_F32);
        assert!(f32::abs(quartic.eps1 - 32.0) < TOL_F32);
        assert!(f32::abs(quartic.eps1eps2 - 48.0) < TOL_F32);
    }

    #[test]
    fn the_zeroth_power_has_no_derivatives() {
        // x^0 = 1 with all derivatives 0
        let zeroth_power = HyperDual::variable(3.0_f64).powi(0);
        assert!(f64::abs(zeroth_power.real - 1.0) < TOL);
        assert!(f64::abs(zeroth_power.eps1) < TOL);
        assert!(f64::abs(zeroth_power.eps1eps2) < TOL);
    }

    #[test]
    fn a_constant_has_zero_derivatives() {
        // a constant carries no derivative through any operation
        let constant = HyperDual::constant(1.3_f64);
        let combined = constant.exp() * constant.sin() + constant.powi(2);
        assert!(f64::abs(combined.eps1) < TOL);
        assert!(f64::abs(combined.eps2) < TOL);
        assert!(f64::abs(combined.eps1eps2) < TOL);
    }

    #[test]
    fn sqrt_at_zero_blows_up() {
        // the derivative of sqrt at 0 is unbounded, while the value stays finite
        let root = HyperDual::variable(0.0_f64).sqrt();
        assert!(f64::abs(root.real) < TOL);
        assert!(root.eps1.is_infinite());
        // is_finite reflects the real part only
        assert!(root.is_finite());
    }

    #[test]
    fn ln_at_zero_blows_up() {
        // ln(0) = -inf with an unbounded first derivative
        let logarithm = HyperDual::variable(0.0_f64).ln();
        assert!(logarithm.real.is_infinite() && logarithm.real < 0.0);
        assert!(logarithm.eps1.is_infinite() && logarithm.eps1 > 0.0);
    }

    #[test]
    fn atan2_second_derivative() {
        // f(y) = atan2(y, x), x constant: f' = x/(x²+y²), f'' = −2xy/(x²+y²)²
        let y = 1.0_f64;
        let x = 2.0_f64;
        let angle = HyperDual::variable(y).atan2(HyperDual::constant(x));
        let denominator = x * x + y * y;
        assert!(f64::abs(angle.real - y.atan2(x)) < TOL);
        assert!(f64::abs(angle.eps1 - x / denominator) < TOL);
        assert!(f64::abs(angle.eps1eps2 - (-2.0 * x * y) / (denominator * denominator)) < TOL);
    }

    #[test]
    fn atan2_mixed_partial_derivative() {
        // ∂²/∂y∂x atan2(y, x) = (y²−x²)/(x²+y²)²; seed y on dir 1, x on dir 2.
        let y = 1.0_f64;
        let x = 2.0_f64;
        let angle = HyperDual::new(y, 1.0, 0.0, 0.0).atan2(HyperDual::new(x, 0.0, 1.0, 0.0));
        let denominator = x * x + y * y;
        assert!(f64::abs(angle.eps1eps2 - (y * y - x * x) / (denominator * denominator)) < TOL);
    }
}

mod jet {
    use multicalc::scalar::Jet;
    use multicalc::scalar::{Dual, Numeric};

    const TOL: f64 = 1e-9;
    const TOL_F32: f32 = 1e-3;

    #[test]
    fn every_derivative_of_exp_is_exp() {
        // every derivative of exp(x) is exp(x)
        let x = 0.4_f64;
        let exponential = Jet::<f64, 6>::variable(x).exp();
        for order in 0..6 {
            assert!(f64::abs(exponential.derivative(order) - f64::exp(x)) < TOL);
        }

        // every non-trivial derivative of exp(x) - 1 is exp(x)
        let exponential = Jet::<f64, 6>::variable(x).expm1();
        assert!(f64::abs(exponential.derivative(0) - f64::exp_m1(x)) < TOL);
        for order in 1..6 {
            assert!(f64::abs(exponential.derivative(order) - f64::exp(x)) < TOL);
        }
    }

    #[test]
    fn a_quartic_terminates_after_four_derivatives() {
        // f(x) = x^4: f'=4x^3, f''=12x^2, f'''=24x, f''''=24, f'''''=0
        let x = 2.0_f64;
        let quartic = Jet::<f64, 6>::variable(x).powi(4);
        assert!(f64::abs(quartic.value() - 16.0) < TOL);
        assert!(f64::abs(quartic.derivative(1) - 32.0) < TOL);
        assert!(f64::abs(quartic.derivative(2) - 48.0) < TOL);
        assert!(f64::abs(quartic.derivative(3) - 48.0) < TOL);
        assert!(f64::abs(quartic.derivative(4) - 24.0) < TOL);
        assert!(f64::abs(quartic.derivative(5)) < TOL);
    }

    #[test]
    fn sin_derivatives_cycle_through_four_terms() {
        // derivatives of sin cycle: sin, cos, -sin, -cos, sin
        let x = 0.6_f64;
        let sine = Jet::<f64, 5>::variable(x).sin();
        assert!(f64::abs(sine.derivative(0) - f64::sin(x)) < TOL);
        assert!(f64::abs(sine.derivative(1) - f64::cos(x)) < TOL);
        assert!(f64::abs(sine.derivative(2) - (-f64::sin(x))) < TOL);
        assert!(f64::abs(sine.derivative(3) - (-f64::cos(x))) < TOL);
        assert!(f64::abs(sine.derivative(4) - f64::sin(x)) < TOL);
    }

    #[test]
    fn reciprocal_derivatives_follow_the_factorial_rule() {
        // f(x) = 1/(1+x): f^(k) = (-1)^k k! / (1+x)^(k+1)
        let x = 0.3_f64;
        let denominator = Jet::<f64, 5>::constant(1.0) + Jet::variable(x);
        let reciprocal = Jet::<f64, 5>::constant(1.0) / denominator;
        let mut sign = 1.0;
        let mut factorial = 1.0;
        for order in 0..5 {
            if order >= 1 {
                factorial *= order as f64;
            }
            let expected = sign * factorial / (1.0 + x).powi(order as i32 + 1);
            assert!(f64::abs(reciprocal.derivative(order) - expected) < TOL);
            sign = -sign;
        }
    }

    #[test]
    fn sqrt_derivatives_to_third_order() {
        // f(x) = sqrt(x): f'=1/(2√x), f''=-1/(4 x^{3/2}), f'''=3/(8 x^{5/2})
        let x = 1.7_f64;
        let root = Jet::<f64, 4>::variable(x).sqrt();
        assert!(f64::abs(root.derivative(0) - f64::sqrt(x)) < TOL);
        assert!(f64::abs(root.derivative(1) - 1.0 / (2.0 * f64::sqrt(x))) < TOL);
        assert!(f64::abs(root.derivative(2) - (-1.0 / (4.0 * x * f64::sqrt(x)))) < TOL);
        assert!(f64::abs(root.derivative(3) - 3.0 / (8.0 * x * x * f64::sqrt(x))) < TOL);
    }

    #[test]
    fn ln_derivatives_to_third_order() {
        // f(x) = ln(x): f'=1/x, f''=-1/x^2, f'''=2/x^3
        let x = 2.0_f64;
        let logarithm = Jet::<f64, 4>::variable(x).ln();
        assert!(f64::abs(logarithm.derivative(0) - f64::ln(x)) < TOL);
        assert!(f64::abs(logarithm.derivative(1) - 1.0 / x) < TOL);
        assert!(f64::abs(logarithm.derivative(2) - (-1.0 / (x * x))) < TOL);
        assert!(f64::abs(logarithm.derivative(3) - 2.0 / (x * x * x)) < TOL);

        // f(x) = log2(x): f'=1/(ln2 * x), f''=-1/(ln2 * x^2), f'''=2/(ln2 * x^3)
        let ln2 = 2.0.ln();
        let logarithm = Jet::<f64, 4>::variable(x).log2();
        assert!(f64::abs(logarithm.derivative(0) - f64::log2(x)) < TOL);
        assert!(f64::abs(logarithm.derivative(1) - x.recip() / ln2) < TOL);
        assert!(f64::abs(logarithm.derivative(2) - (-1.0 / (ln2 * x * x))) < TOL);
        assert!(f64::abs(logarithm.derivative(3) - 2.0 / (ln2 * x * x * x)) < TOL);

        // f(x) = log10(x): f'=1/(ln10 * x), f''=-1/(ln10 * x^2), f'''=2/(ln10 * x^3)
        let ln10 = 10.0.ln();
        let logarithm = Jet::<f64, 4>::variable(x).log10();
        assert!(f64::abs(logarithm.derivative(0) - f64::log10(x)) < TOL);
        assert!(f64::abs(logarithm.derivative(1) - x.recip() / ln10) < TOL);
        assert!(f64::abs(logarithm.derivative(2) - (-1.0 / (ln10 * x * x))) < TOL);
        assert!(f64::abs(logarithm.derivative(3) - 2.0 / (ln10 * x * x * x)) < TOL);

        // f(x) = ln(1 + x): f'=1/(1 + x), f''=-1/(1 + x)^2, f'''=2/(1 + x)^3
        let x = 0.123_f64;
        let xp1 = x + 1.0;
        let logarithm = Jet::<f64, 4>::variable(x).ln_1p();
        assert!(f64::abs(logarithm.derivative(0) - f64::ln_1p(x)) < TOL);
        assert!(f64::abs(logarithm.derivative(1) - 1.0 / xp1) < TOL);
        assert!(f64::abs(logarithm.derivative(2) - (-1.0 / (xp1 * xp1))) < TOL);
        assert!(f64::abs(logarithm.derivative(3) - 2.0 / (xp1 * xp1 * xp1)) < TOL);
    }

    #[test]
    fn tan_derivatives_to_second_order() {
        // f(x) = tan(x): f' = 1+tan^2, f'' = 2 tan (1+tan^2)
        let x = 0.5_f64;
        let tangent = f64::tan(x);
        let jet = Jet::<f64, 3>::variable(x).tan();
        assert!(f64::abs(jet.derivative(0) - tangent) < TOL);
        assert!(f64::abs(jet.derivative(1) - (1.0 + tangent * tangent)) < TOL);
        assert!(f64::abs(jet.derivative(2) - 2.0 * tangent * (1.0 + tangent * tangent)) < TOL);
    }

    #[test]
    fn matches_dual_at_first_order() {
        // Jet<T,2> carries the same first derivative as Dual.
        fn function<T: Numeric>(t: T) -> T {
            t.sin() * t.exp() + t.powi(3)
        }
        let x = 0.8_f64;
        let jet = function(Jet::<f64, 2>::variable(x));
        let dual = function(Dual::<f64>::variable(x));
        assert!(f64::abs(jet.value() - dual.value) < TOL);
        assert!(f64::abs(jet.coeffs[1] - dual.deriv) < TOL);
    }

    #[test]
    fn generic_over_numeric() {
        // The same function runs with a plain float or with a Jet.
        fn polynomial<T: Numeric>(t: T) -> T {
            t.powi(3) + T::from_f64(2.0) * t
        }
        let x = 1.5_f64;
        let plain = polynomial(x);
        let jet = polynomial(Jet::<f64, 4>::variable(x));
        assert!(f64::abs(jet.value() - plain) < TOL);
        assert!(f64::abs(jet.derivative(1) - (3.0 * x * x + 2.0)) < TOL); // f'
        assert!(f64::abs(jet.derivative(2) - 6.0 * x) < TOL); // f''
        assert!(f64::abs(jet.derivative(3) - 6.0) < TOL); // f'''
    }

    #[test]
    fn generic_over_f32() {
        // Jet is generic over the scalar; here it carries f32.
        let cubed = Jet::<f32, 4>::variable(2.0).powi(3);
        assert!(f32::abs(cubed.derivative(0) - 8.0) < TOL_F32);
        assert!(f32::abs(cubed.derivative(1) - 12.0) < TOL_F32);
        assert!(f32::abs(cubed.derivative(2) - 12.0) < TOL_F32);
        assert!(f32::abs(cubed.derivative(3) - 6.0) < TOL_F32);
    }

    #[test]
    fn a_single_coefficient_behaves_like_a_bare_scalar() {
        // N = 1 carries only the value and behaves like the bare scalar.
        let product = Jet::<f64, 1>::constant(2.0) * Jet::<f64, 1>::constant(3.0);
        assert!(f64::abs(product.value() - 6.0) < TOL);
        assert!(f64::abs(Jet::<f64, 1>::constant(1.0).exp().value() - f64::exp(1.0)) < TOL);
    }

    #[test]
    fn a_constant_has_zero_higher_coefficients() {
        let constant = Jet::<f64, 4>::constant(2.0);
        for order in 1..4 {
            assert!(f64::abs(constant.coeffs[order]) < TOL);
        }
    }

    #[test]
    fn sqrt_at_zero_blows_up() {
        // the derivative of sqrt at 0 is unbounded, while the value stays finite
        let root = Jet::<f64, 3>::variable(0.0).sqrt();
        assert!(f64::abs(root.value()) < TOL);
        assert!(root.coeffs[1].is_infinite());
        // is_finite reflects the value only
        assert!(root.is_finite());
    }

    #[test]
    fn atan2_derivatives_to_third_order() {
        // atan2(y, 1) = atan(y): f'=1/(1+y²), f''=−2y/(1+y²)², f'''=(6y²−2)/(1+y²)³
        let y = 0.5_f64;
        let jet = Jet::<f64, 4>::variable(y).atan2(Jet::constant(1.0));
        let denominator = 1.0 + y * y;
        assert!(f64::abs(jet.derivative(0) - y.atan()) < TOL);
        assert!(f64::abs(jet.derivative(1) - 1.0 / denominator) < TOL);
        assert!(f64::abs(jet.derivative(2) - (-2.0 * y) / (denominator * denominator)) < TOL);
        assert!(
            f64::abs(
                jet.derivative(3) - (6.0 * y * y - 2.0) / (denominator * denominator * denominator)
            ) < TOL
        );
    }

    #[test]
    fn atan2_matches_dual_at_first_order() {
        let y = 1.3_f64;
        let x = 0.7_f64;
        let jet = Jet::<f64, 2>::variable(y).atan2(Jet::constant(x));
        let dual = Dual::variable(y).atan2(Dual::constant(x));
        assert!(f64::abs(jet.value() - dual.value) < TOL);
        assert!(f64::abs(jet.coeffs[1] - dual.deriv) < TOL);
    }
}

mod function {
    use multicalc::scalar::{Dual, HyperDual, Jet, Numeric};
    use multicalc::scalar::{ScalarFn, ScalarFnN, VectorFn, c};
    use multicalc::{scalar_fn, scalar_fn_vec};

    // f(x) = 4x^3 - 3x^2, hand-written over the scalar.
    struct Cubic;
    impl ScalarFn for Cubic {
        fn eval<S: Numeric>(&self, x: S) -> S {
            S::from_f64(4.0) * x * x * x - S::from_f64(3.0) * x * x
        }
    }

    #[test]
    fn one_function_drives_every_backend() {
        let cubic = Cubic;
        // plain f64 (finite-difference path): 4*8 - 3*4 = 20
        assert!(f64::abs(cubic.eval(2.0_f64) - 20.0) < 1e-12);
        // Dual: f'(x) = 12x^2 - 6x = 36 at x = 2
        assert!(f64::abs(cubic.eval(Dual::variable(2.0_f64)).deriv - 36.0) < 1e-12);
        // HyperDual: f''(x) = 24x - 6 = 42 at x = 2
        assert!(f64::abs(cubic.eval(HyperDual::variable(2.0_f64)).eps1eps2 - 42.0) < 1e-12);
        // Jet: f'''(x) = 24
        assert!(f64::abs(cubic.eval(Jet::<f64, 4>::variable(2.0_f64)).derivative(3) - 24.0) < 1e-9);
    }

    // g(x, y, z) = y*sin(x) + 2*x*e^z, hand-written over the scalar.
    struct Mixed;
    impl ScalarFnN<3> for Mixed {
        fn eval<S: Numeric>(&self, point: &[S; 3]) -> S {
            point[1] * point[0].sin() + S::from_f64(2.0) * point[0] * point[2].exp()
        }
    }

    #[test]
    fn multivariable_partial_via_seeding() {
        let mixed = Mixed;
        let point = [1.0_f64, 2.0, 0.5];
        let expected = 2.0 * f64::sin(1.0) + 2.0 * f64::exp(0.5);
        assert!(f64::abs(mixed.eval(&point) - expected) < 1e-12);

        // partial dg/dx via Dual seeding of index 0:
        // dg/dx = y*cos(x) + 2*e^z = 2*cos(1) + 2*e^0.5
        let seeded = [
            Dual::variable(1.0_f64),
            Dual::constant(2.0),
            Dual::constant(0.5),
        ];
        let expected_dx = 2.0 * f64::cos(1.0) + 2.0 * f64::exp(0.5);
        assert!(f64::abs(mixed.eval(&seeded).deriv - expected_dx) < 1e-12);
    }

    #[test]
    fn macro_single_var() {
        // f(x) = 4x^3 - 3x^2, authored via the macro.
        let cubic = scalar_fn!(|x| c(4.0) * x * x * x - c(3.0) * x * x);
        assert!(f64::abs(cubic.eval(2.0_f64) - 20.0) < 1e-12);
        assert!(f64::abs(cubic.eval(Dual::variable(2.0_f64)).deriv - 36.0) < 1e-12);
        assert!(f64::abs(cubic.eval(HyperDual::variable(2.0_f64)).eps1eps2 - 42.0) < 1e-12);
        assert!(f64::abs(cubic.eval(Jet::<f64, 4>::variable(2.0_f64)).derivative(3) - 24.0) < 1e-9);
    }

    #[test]
    fn macro_single_var_typed_param() {
        let scaled_sine = scalar_fn!(|x: f64| c(2.0) * x.sin());
        assert!(f64::abs(scaled_sine.eval(0.5_f64) - 2.0 * f64::sin(0.5)) < 1e-12);
    }

    #[test]
    fn macro_multivariable() {
        let mixed = scalar_fn!(|v: &[f64; 3]| v[1] * v[0].sin() + c(2.0) * v[0] * v[2].exp());
        let point = [1.0_f64, 2.0, 0.5];
        let expected = 2.0 * f64::sin(1.0) + 2.0 * f64::exp(0.5);
        assert!(f64::abs(mixed.eval(&point) - expected) < 1e-12);
    }

    #[test]
    fn macro_vector_valued() {
        // f(x, y) = [x*y, sin(y)]
        let function = scalar_fn_vec!(|v: &[f64; 2]| [v[0] * v[1], v[1].sin()]);
        let outputs = function.eval(&[3.0_f64, 0.5]);
        assert!(f64::abs(outputs[0] - 1.5) < 1e-12);
        assert!(f64::abs(outputs[1] - f64::sin(0.5)) < 1e-12);

        // a Jacobian column via Dual: d/dx [x*y, sin(y)] = [y, 0] at (3, 0.5)
        let seeded = [Dual::variable(3.0_f64), Dual::constant(0.5)];
        let column = function.eval(&seeded);
        assert!(f64::abs(column[0].deriv - 0.5) < 1e-12);
        assert!(f64::abs(column[1].deriv) < 1e-12);
    }
}
