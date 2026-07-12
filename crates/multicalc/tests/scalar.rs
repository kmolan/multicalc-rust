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
        assert_eq!(Numeric::mul_add(2.0_f64, 3.0, 1.0), 7.0);
        assert_eq!(Numeric::recip(4.0_f64), 0.25);
        assert_eq!(Numeric::signum(-3.0_f64), -1.0);
        assert!(Numeric::signum(f64::NAN).is_nan());
    }

    #[test]
    fn hypot_no_overflow() {
        // naive sqrt(a²+b²) would overflow; the scaled form must stay finite and correct.
        let big = 1e200_f64;
        let h = Numeric::hypot(big, big);
        assert!(h.is_finite());
        assert!((h - big * 2.0_f64.sqrt()).abs() / h < 1e-12);
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
        let (y, xx) = (0.7_f32, 1.3_f32);
        // cos(atan2(y, x))·hypot(y, x) = x
        assert!((Numeric::atan2(y, xx).cos() * Numeric::hypot(y, xx) - xx).abs() < 1e-4);
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
    use multicalc::{Dual, HyperDual, Jet, Numeric, scalar::primal::Primal};

    #[test]
    fn test_f64() {
        let two = f64::TWO;

        assert_eq!(two.to_f64(), two);
        assert_eq!(two.to_f32(), two as f32);
    }

    #[test]
    fn test_f32() {
        let two = f32::TWO;

        assert_eq!(two.to_f64(), two as f64);
        assert_eq!(two.to_f32(), two);
    }

    #[test]
    fn test_dual() {
        let two = Dual::<f64>::TWO;

        assert_eq!(two.to_f64(), f64::TWO);
        assert_eq!(two.to_f32(), f32::TWO);
    }

    #[test]
    fn test_hyperdual() {
        let two = HyperDual::<f64>::TWO;

        assert_eq!(two.to_f64(), f64::TWO);
        assert_eq!(two.to_f32(), f32::TWO);
    }

    #[test]
    fn test_jet() {
        let two = Jet::<f64, 2>::TWO;

        assert_eq!(two.to_f64(), f64::TWO);
        assert_eq!(two.to_f32(), f32::TWO);
    }

    #[test]
    fn test_dual_hyperdual() {
        let two = Dual::<HyperDual<f64>>::TWO;

        assert_eq!(two.to_f64(), f64::TWO);
        assert_eq!(two.to_f32(), f32::TWO);
    }
}

mod dual {
    use multicalc::scalar::Dual;
    use multicalc::scalar::Numeric;

    // Dual results are exact to rounding, so the tolerances are tight.
    const TOL: f64 = 1e-12;
    const TOL_F32: f32 = 1e-5;

    #[test]
    fn test_polynomial_powi() {
        // f(x) = x^3, f'(x) = 3x^2; at x = 2 -> 8 and 12
        let y = Dual::variable(2.0_f64).powi(3);
        assert!(f64::abs(y.value - 8.0) < TOL);
        assert!(f64::abs(y.deriv - 12.0) < TOL);
    }

    #[test]
    fn test_polynomial_sum() {
        // f(x) = 3x^2 + 2x, f'(x) = 6x + 2; at x = 2 -> 16 and 14
        let x = Dual::variable(2.0_f64);
        let y = Dual::constant(3.0) * x * x + Dual::constant(2.0) * x;
        assert!(f64::abs(y.value - 16.0) < TOL);
        assert!(f64::abs(y.deriv - 14.0) < TOL);
    }

    #[test]
    fn test_negative_powi() {
        // f(x) = x^-2, f'(x) = -2 x^-3; at x = 2 -> 0.25 and -0.25
        let y = Dual::variable(2.0_f64).powi(-2);
        assert!(f64::abs(y.value - 0.25) < TOL);
        assert!(f64::abs(y.deriv - (-0.25)) < TOL);
    }

    #[test]
    fn test_sqrt() {
        // f(x) = sqrt(x), f'(x) = 1/(2 sqrt(x)); at x = 4 -> 2 and 0.25
        let y = Dual::variable(4.0_f64).sqrt();
        assert!(f64::abs(y.value - 2.0) < TOL);
        assert!(f64::abs(y.deriv - 0.25) < TOL);
    }

    #[test]
    fn test_sin_cos_tan() {
        let x0 = 0.7_f64;
        let s = Dual::variable(x0).sin();
        assert!(f64::abs(s.value - f64::sin(x0)) < TOL);
        assert!(f64::abs(s.deriv - f64::cos(x0)) < TOL);

        let c = Dual::variable(x0).cos();
        assert!(f64::abs(c.value - f64::cos(x0)) < TOL);
        assert!(f64::abs(c.deriv - (-f64::sin(x0))) < TOL);

        let t = Dual::variable(x0).tan();
        assert!(f64::abs(t.value - f64::tan(x0)) < TOL);
        assert!(f64::abs(t.deriv - (1.0 + f64::tan(x0) * f64::tan(x0))) < TOL);
    }

    #[test]
    fn test_exp_ln() {
        let x0 = 1.3_f64;
        let e = Dual::variable(x0).exp();
        assert!(f64::abs(e.value - f64::exp(x0)) < TOL);
        assert!(f64::abs(e.deriv - f64::exp(x0)) < TOL);

        // f(x) = ln(x), f'(x) = 1/x; at x = 2 -> ln 2 and 0.5
        let l = Dual::variable(2.0_f64).ln();
        assert!(f64::abs(l.value - f64::ln(2.0)) < TOL);
        assert!(f64::abs(l.deriv - 0.5) < TOL);
    }

    #[test]
    fn test_chain_exp_of_sin() {
        // f(x) = exp(sin(x)), f'(x) = cos(x) exp(sin(x))
        let x0 = 0.6_f64;
        let y = Dual::variable(x0).sin().exp();
        assert!(f64::abs(y.value - f64::exp(f64::sin(x0))) < TOL);
        assert!(f64::abs(y.deriv - f64::cos(x0) * f64::exp(f64::sin(x0))) < TOL);
    }

    #[test]
    fn test_rational() {
        // f(x) = x / (1 + x^2), f'(x) = (1 - x^2) / (1 + x^2)^2
        let x0 = 1.5_f64;
        let x = Dual::variable(x0);
        let y = x / (Dual::constant(1.0) + x * x);
        let denom = (1.0 + x0 * x0) * (1.0 + x0 * x0);
        assert!(f64::abs(y.value - x0 / (1.0 + x0 * x0)) < TOL);
        assert!(f64::abs(y.deriv - (1.0 - x0 * x0) / denom) < TOL);
    }

    #[test]
    fn test_product_sin_cos() {
        // f(x) = sin(x) cos(x), f'(x) = cos^2(x) - sin^2(x)
        let x0 = 0.9_f64;
        let x = Dual::variable(x0);
        let y = x.sin() * x.cos();
        assert!(f64::abs(y.value - f64::sin(x0) * f64::cos(x0)) < TOL);
        let expected = f64::cos(x0) * f64::cos(x0) - f64::sin(x0) * f64::sin(x0);
        assert!(f64::abs(y.deriv - expected) < TOL);
    }

    #[test]
    fn test_abs_both_sides() {
        // derivative of |x| is +1 for x > 0 and -1 for x < 0
        let pos = Dual::variable(2.0_f64).abs();
        assert!(f64::abs(pos.value - 2.0) < TOL);
        assert!(f64::abs(pos.deriv - 1.0) < TOL);

        let neg = Dual::variable(-2.0_f64).abs();
        assert!(f64::abs(neg.value - 2.0) < TOL);
        assert!(f64::abs(neg.deriv - (-1.0)) < TOL);
    }

    #[test]
    fn test_generic_over_numeric() {
        // The same function runs with a plain float or with a Dual.
        fn poly<T: Numeric>(t: T) -> T {
            t.powi(3) + T::from_f64(2.0) * t
        }

        let x0 = 1.7_f64;
        let plain = poly(x0);
        let dual = poly(Dual::variable(x0));
        assert!(f64::abs(dual.value - plain) < TOL);
        // f'(x) = 3x^2 + 2
        assert!(f64::abs(dual.deriv - (3.0 * x0 * x0 + 2.0)) < TOL);
    }

    #[test]
    fn test_partial_derivatives() {
        // f(x, y) = x^2 * y + sin(x)
        fn f<T: Numeric>(v: &[T; 2]) -> T {
            v[0] * v[0] * v[1] + v[0].sin()
        }

        let (x0, y0) = (1.0_f64, 2.0_f64);

        // seed x: df/dx = 2xy + cos(x)
        let dfdx = f(&[Dual::variable(x0), Dual::constant(y0)]).deriv;
        assert!(f64::abs(dfdx - (2.0 * x0 * y0 + f64::cos(x0))) < TOL);

        // seed y: df/dy = x^2
        let dfdy = f(&[Dual::constant(x0), Dual::variable(y0)]).deriv;
        assert!(f64::abs(dfdy - x0 * x0) < TOL);
    }

    #[test]
    fn test_generic_over_f32() {
        // Dual is generic over the scalar; here it carries f32.
        let y = Dual::variable(2.0_f32).powi(3);
        assert!(f32::abs(y.value - 8.0) < TOL_F32);
        assert!(f32::abs(y.deriv - 12.0) < TOL_F32);
    }

    #[test]
    fn test_powi_zero() {
        // x^0 = 1, derivative 0
        let y = Dual::variable(3.0_f64).powi(0);
        assert!(f64::abs(y.value - 1.0) < TOL);
        assert!(f64::abs(y.deriv) < TOL);
    }

    #[test]
    fn test_constant_has_zero_derivative() {
        // a constant carries no derivative through any operation
        let c = Dual::constant(1.3_f64);
        let y = c.exp() * c.sin() + c.powi(2);
        assert!(f64::abs(y.deriv) < TOL);
    }

    #[test]
    fn test_sqrt_zero_derivative_is_infinite() {
        // the derivative of sqrt at 0 is unbounded, while the value stays finite
        let y = Dual::variable(0.0_f64).sqrt();
        assert!(f64::abs(y.value) < TOL);
        assert!(y.deriv.is_infinite());
        // is_finite reflects the value only, so it still reports finite here
        assert!(y.is_finite());
    }

    #[test]
    fn test_ln_zero_blows_up() {
        // ln(0) = -inf with an unbounded derivative
        let y = Dual::variable(0.0_f64).ln();
        assert!(y.value.is_infinite() && y.value < 0.0);
        assert!(y.deriv.is_infinite() && y.deriv > 0.0);
    }

    #[test]
    fn test_atan2_derivative() {
        // f(y) = atan2(y, x), x constant: f'(y) = x/(x²+y²)
        let (y0, x0) = (1.0_f64, 2.0_f64);
        let r = Dual::variable(y0).atan2(Dual::constant(x0));
        assert!(f64::abs(r.value - y0.atan2(x0)) < TOL);
        assert!(f64::abs(r.deriv - x0 / (x0 * x0 + y0 * y0)) < TOL);
    }

    #[test]
    fn test_inverse_trig_derivatives() {
        let x0 = 0.3_f64;
        let s = Dual::variable(x0).asin();
        assert!(f64::abs(s.value - x0.asin()) < TOL);
        assert!(f64::abs(s.deriv - 1.0 / (1.0 - x0 * x0).sqrt()) < TOL); // asin' = 1/√(1−x²)
        let c = Dual::variable(x0).acos();
        assert!(f64::abs(c.deriv + 1.0 / (1.0 - x0 * x0).sqrt()) < TOL); // acos' = −asin'
        let a = Dual::variable(x0).atan();
        assert!(f64::abs(a.deriv - 1.0 / (1.0 + x0 * x0)) < TOL); // atan' = 1/(1+x²)
    }

    #[test]
    fn test_hyperbolic_derivatives() {
        let x0 = 0.8_f64;
        let sh = Dual::variable(x0).sinh();
        assert!(f64::abs(sh.deriv - x0.cosh()) < TOL); // sinh' = cosh
        let ch = Dual::variable(x0).cosh();
        assert!(f64::abs(ch.deriv - x0.sinh()) < TOL); // cosh' = sinh
        let th = Dual::variable(x0).tanh();
        assert!(f64::abs(th.deriv - (1.0 - x0.tanh() * x0.tanh())) < 1e-10); // tanh' = 1−tanh²
    }

    #[test]
    fn test_powf_recip_hypot_derivatives() {
        let x0 = 2.0_f64;
        let p = Dual::variable(x0).powf(Dual::constant(3.5)); // d/dx x^3.5 = 3.5 x^2.5
        assert!(f64::abs(p.deriv - 3.5 * x0.powf(2.5)) < 1e-10);
        let r = Dual::variable(x0).recip(); // d/dx 1/x = −1/x²
        assert!(f64::abs(r.deriv + 0.25) < TOL);
        // hypot(x, 4) with x variable: d/dx = x/hypot
        let h = Dual::variable(3.0_f64).hypot(Dual::constant(4.0));
        assert!(f64::abs(h.value - 5.0) < TOL);
        assert!(f64::abs(h.deriv - 3.0 / 5.0) < TOL);
    }

    #[test]
    fn test_floor_derivative_is_zero() {
        let y = Dual::variable(2.7_f64).floor();
        assert!(f64::abs(y.value - 2.0) < TOL);
        assert!(f64::abs(y.deriv) < TOL);
    }

    #[test]
    fn test_copysign_derivative() {
        let same = Dual::variable(3.0_f64).copysign(Dual::constant(1.0));
        assert!(f64::abs(same.value - 3.0) < TOL && f64::abs(same.deriv - 1.0) < TOL);
        let flip = Dual::variable(3.0_f64).copysign(Dual::constant(-1.0));
        assert!(f64::abs(flip.value + 3.0) < TOL && f64::abs(flip.deriv + 1.0) < TOL);
    }

    #[test]
    fn test_atan2_derivative_random() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        // d/dy atan2(y, x) = x/(x²+y²); d/dx atan2(y, x) = −y/(x²+y²). Sweep both, all
        // quadrants and near the axes, comparing AD to the closed form.
        let mut rng = StdRng::seed_from_u64(0xA7A2);
        for _ in 0..1000 {
            let y0: f64 = rng.gen_range(-5.0..5.0);
            let x0: f64 = rng.gen_range(-5.0..5.0);
            let denom = x0 * x0 + y0 * y0;
            if denom < 1e-6 {
                continue; // origin is the documented singularity
            }
            // seed y (x constant): expect x/denom
            let dy = Dual::variable(y0).atan2(Dual::constant(x0));
            assert!(f64::abs(dy.value - y0.atan2(x0)) < TOL);
            assert!(f64::abs(dy.deriv - x0 / denom) < 1e-9);
            // seed x (y constant): expect −y/denom
            let dx = Dual::constant(y0).atan2(Dual::variable(x0));
            assert!(f64::abs(dx.deriv + y0 / denom) < 1e-9);
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
    fn test_single_var_cubic() {
        // f(x) = x^3 -> f'(x) = 3x^2, f''(x) = 6x; at x = 3: 27, 27, 18
        let y = HyperDual::variable(3.0_f64).powi(3);
        assert!(f64::abs(y.real - 27.0) < TOL);
        assert!(f64::abs(y.eps1 - 27.0) < TOL);
        assert!(f64::abs(y.eps2 - 27.0) < TOL);
        assert!(f64::abs(y.eps1eps2 - 18.0) < TOL);
    }

    #[test]
    fn test_powi_second_order() {
        // f(x) = x^4 -> f'(x) = 4x^3, f''(x) = 12x^2; at x = 2: 16, 32, 48
        let y = HyperDual::variable(2.0_f64).powi(4);
        assert!(f64::abs(y.real - 16.0) < TOL);
        assert!(f64::abs(y.eps1 - 32.0) < TOL);
        assert!(f64::abs(y.eps1eps2 - 48.0) < TOL);
    }

    #[test]
    fn test_sin_second_order() {
        // f(x) = sin(x) -> f' = cos(x), f'' = -sin(x)
        let x0 = 0.7_f64;
        let y = HyperDual::variable(x0).sin();
        assert!(f64::abs(y.real - f64::sin(x0)) < TOL);
        assert!(f64::abs(y.eps1 - f64::cos(x0)) < TOL);
        assert!(f64::abs(y.eps1eps2 - (-f64::sin(x0))) < TOL);
    }

    #[test]
    fn test_exp_second_order() {
        // f(x) = exp(x) is its own derivative to all orders
        let x0 = 1.3_f64;
        let y = HyperDual::variable(x0).exp();
        assert!(f64::abs(y.real - f64::exp(x0)) < TOL);
        assert!(f64::abs(y.eps1 - f64::exp(x0)) < TOL);
        assert!(f64::abs(y.eps1eps2 - f64::exp(x0)) < TOL);
    }

    #[test]
    fn test_reciprocal_second_order() {
        // f(x) = 1/x -> f' = -1/x^2, f'' = 2/x^3; at x = 2: 0.5, -0.25, 0.25
        let y = HyperDual::constant(1.0_f64) / HyperDual::variable(2.0_f64);
        assert!(f64::abs(y.real - 0.5) < TOL);
        assert!(f64::abs(y.eps1 - (-0.25)) < TOL);
        assert!(f64::abs(y.eps1eps2 - 0.25) < TOL);
    }

    #[test]
    fn test_full_hessian_matches_analytic() {
        // f(x, y) = x^2 * y + sin(x)
        // grad  = [2xy + cos x, x^2]
        // H     = [[2y - sin x, 2x], [2x, 0]]
        fn f<T: Numeric>(v: &[T; 2]) -> T {
            v[0] * v[0] * v[1] + v[0].sin()
        }

        let (x0, y0) = (1.0_f64, 2.0_f64);

        // diagonal Hxx and the x-gradient: seed x on both directions, y constant
        let hxx = f(&[HyperDual::variable(x0), HyperDual::constant(y0)]);
        assert!(f64::abs(hxx.eps1 - (2.0 * x0 * y0 + f64::cos(x0))) < TOL); // df/dx
        assert!(f64::abs(hxx.eps1eps2 - (2.0 * y0 - f64::sin(x0))) < TOL); // d2f/dx2

        // diagonal Hyy and the y-gradient: seed y on both directions, x constant
        let hyy = f(&[HyperDual::constant(x0), HyperDual::variable(y0)]);
        assert!(f64::abs(hyy.eps1 - x0 * x0) < TOL); // df/dy
        assert!(f64::abs(hyy.eps1eps2) < TOL); // d2f/dy2 = 0

        // mixed Hxy: seed x on direction 1, y on direction 2
        let hxy = f(&[
            HyperDual::new(x0, 1.0, 0.0, 0.0),
            HyperDual::new(y0, 0.0, 1.0, 0.0),
        ]);
        assert!(f64::abs(hxy.eps1eps2 - 2.0 * x0) < TOL);

        // symmetry: swapping the directions gives the same mixed partial
        let hyx = f(&[
            HyperDual::new(x0, 0.0, 1.0, 0.0),
            HyperDual::new(y0, 1.0, 0.0, 0.0),
        ]);
        assert!(f64::abs(hxy.eps1eps2 - hyx.eps1eps2) < TOL);
    }

    #[test]
    fn test_generic_over_numeric() {
        // The same function runs with a plain float or with a HyperDual.
        fn g<T: Numeric>(t: T) -> T {
            t.powi(3) + T::from_f64(2.0) * t
        }

        let x0 = 1.7_f64;
        let plain = g(x0);
        let hd = g(HyperDual::variable(x0));
        assert!(f64::abs(hd.real - plain) < TOL);
        assert!(f64::abs(hd.eps1 - (3.0 * x0 * x0 + 2.0)) < TOL); // f'
        assert!(f64::abs(hd.eps1eps2 - 6.0 * x0) < TOL); // f''
    }

    #[test]
    fn test_generic_over_f32() {
        // HyperDual is generic over the scalar; here it carries f32.
        let y = HyperDual::variable(2.0_f32).powi(4);
        assert!(f32::abs(y.real - 16.0) < TOL_F32);
        assert!(f32::abs(y.eps1 - 32.0) < TOL_F32);
        assert!(f32::abs(y.eps1eps2 - 48.0) < TOL_F32);
    }

    #[test]
    fn test_powi_zero() {
        // x^0 = 1 with all derivatives 0
        let y = HyperDual::variable(3.0_f64).powi(0);
        assert!(f64::abs(y.real - 1.0) < TOL);
        assert!(f64::abs(y.eps1) < TOL);
        assert!(f64::abs(y.eps1eps2) < TOL);
    }

    #[test]
    fn test_constant_has_zero_derivatives() {
        // a constant carries no derivative through any operation
        let c = HyperDual::constant(1.3_f64);
        let y = c.exp() * c.sin() + c.powi(2);
        assert!(f64::abs(y.eps1) < TOL);
        assert!(f64::abs(y.eps2) < TOL);
        assert!(f64::abs(y.eps1eps2) < TOL);
    }

    #[test]
    fn test_sqrt_zero_blows_up() {
        // the derivative of sqrt at 0 is unbounded, while the value stays finite
        let y = HyperDual::variable(0.0_f64).sqrt();
        assert!(f64::abs(y.real) < TOL);
        assert!(y.eps1.is_infinite());
        // is_finite reflects the real part only
        assert!(y.is_finite());
    }

    #[test]
    fn test_ln_zero_blows_up() {
        // ln(0) = -inf with an unbounded first derivative
        let y = HyperDual::variable(0.0_f64).ln();
        assert!(y.real.is_infinite() && y.real < 0.0);
        assert!(y.eps1.is_infinite() && y.eps1 > 0.0);
    }

    #[test]
    fn test_atan2_second_order() {
        // f(y) = atan2(y, x), x constant: f' = x/(x²+y²), f'' = −2xy/(x²+y²)²
        let (y0, x0) = (1.0_f64, 2.0_f64);
        let r = HyperDual::variable(y0).atan2(HyperDual::constant(x0));
        let d = x0 * x0 + y0 * y0;
        assert!(f64::abs(r.real - y0.atan2(x0)) < TOL);
        assert!(f64::abs(r.eps1 - x0 / d) < TOL);
        assert!(f64::abs(r.eps1eps2 - (-2.0 * x0 * y0) / (d * d)) < TOL);
    }

    #[test]
    fn test_atan2_mixed_partial() {
        // ∂²/∂y∂x atan2(y, x) = (y²−x²)/(x²+y²)²; seed y on dir 1, x on dir 2.
        let (y0, x0) = (1.0_f64, 2.0_f64);
        let r = HyperDual::new(y0, 1.0, 0.0, 0.0).atan2(HyperDual::new(x0, 0.0, 1.0, 0.0));
        let d = x0 * x0 + y0 * y0;
        assert!(f64::abs(r.eps1eps2 - (y0 * y0 - x0 * x0) / (d * d)) < TOL);
    }
}

mod jet {
    use multicalc::scalar::Jet;
    use multicalc::scalar::{Dual, Numeric};

    const TOL: f64 = 1e-9;
    const TOL_F32: f32 = 1e-3;

    #[test]
    fn test_exp_all_orders() {
        // every derivative of exp(x) is exp(x)
        let x0 = 0.4_f64;
        let y = Jet::<f64, 6>::variable(x0).exp();
        for k in 0..6 {
            assert!(f64::abs(y.derivative(k) - f64::exp(x0)) < TOL);
        }
    }

    #[test]
    fn test_high_order_polynomial() {
        // f(x) = x^4: f'=4x^3, f''=12x^2, f'''=24x, f''''=24, f'''''=0
        let x0 = 2.0_f64;
        let y = Jet::<f64, 6>::variable(x0).powi(4);
        assert!(f64::abs(y.value() - 16.0) < TOL);
        assert!(f64::abs(y.derivative(1) - 32.0) < TOL);
        assert!(f64::abs(y.derivative(2) - 48.0) < TOL);
        assert!(f64::abs(y.derivative(3) - 48.0) < TOL);
        assert!(f64::abs(y.derivative(4) - 24.0) < TOL);
        assert!(f64::abs(y.derivative(5)) < TOL);
    }

    #[test]
    fn test_sin_derivative_cycle() {
        // derivatives of sin cycle: sin, cos, -sin, -cos, sin
        let x0 = 0.6_f64;
        let y = Jet::<f64, 5>::variable(x0).sin();
        assert!(f64::abs(y.derivative(0) - f64::sin(x0)) < TOL);
        assert!(f64::abs(y.derivative(1) - f64::cos(x0)) < TOL);
        assert!(f64::abs(y.derivative(2) - (-f64::sin(x0))) < TOL);
        assert!(f64::abs(y.derivative(3) - (-f64::cos(x0))) < TOL);
        assert!(f64::abs(y.derivative(4) - f64::sin(x0)) < TOL);
    }

    #[test]
    fn test_reciprocal_all_orders() {
        // f(x) = 1/(1+x): f^(k) = (-1)^k k! / (1+x)^(k+1)
        let x0 = 0.3_f64;
        let denom = Jet::<f64, 5>::constant(1.0) + Jet::variable(x0);
        let y = Jet::<f64, 5>::constant(1.0) / denom;
        let mut sign = 1.0;
        let mut factorial = 1.0;
        for k in 0..5 {
            if k >= 1 {
                factorial *= k as f64;
            }
            let expected = sign * factorial / (1.0 + x0).powi(k as i32 + 1);
            assert!(f64::abs(y.derivative(k) - expected) < TOL);
            sign = -sign;
        }
    }

    #[test]
    fn test_sqrt_orders() {
        // f(x) = sqrt(x): f'=1/(2√x), f''=-1/(4 x^{3/2}), f'''=3/(8 x^{5/2})
        let x0 = 1.7_f64;
        let y = Jet::<f64, 4>::variable(x0).sqrt();
        assert!(f64::abs(y.derivative(0) - f64::sqrt(x0)) < TOL);
        assert!(f64::abs(y.derivative(1) - 1.0 / (2.0 * f64::sqrt(x0))) < TOL);
        assert!(f64::abs(y.derivative(2) - (-1.0 / (4.0 * x0 * f64::sqrt(x0)))) < TOL);
        assert!(f64::abs(y.derivative(3) - 3.0 / (8.0 * x0 * x0 * f64::sqrt(x0))) < TOL);
    }

    #[test]
    fn test_ln_orders() {
        // f(x) = ln(x): f'=1/x, f''=-1/x^2, f'''=2/x^3
        let x0 = 2.0_f64;
        let y = Jet::<f64, 4>::variable(x0).ln();
        assert!(f64::abs(y.derivative(0) - f64::ln(x0)) < TOL);
        assert!(f64::abs(y.derivative(1) - 1.0 / x0) < TOL);
        assert!(f64::abs(y.derivative(2) - (-1.0 / (x0 * x0))) < TOL);
        assert!(f64::abs(y.derivative(3) - 2.0 / (x0 * x0 * x0)) < TOL);
    }

    #[test]
    fn test_tan_orders() {
        // f(x) = tan(x): f' = 1+tan^2, f'' = 2 tan (1+tan^2)
        let x0 = 0.5_f64;
        let t = f64::tan(x0);
        let y = Jet::<f64, 3>::variable(x0).tan();
        assert!(f64::abs(y.derivative(0) - t) < TOL);
        assert!(f64::abs(y.derivative(1) - (1.0 + t * t)) < TOL);
        assert!(f64::abs(y.derivative(2) - 2.0 * t * (1.0 + t * t)) < TOL);
    }

    #[test]
    fn test_matches_dual_at_order_one() {
        // Jet<T,2> carries the same first derivative as Dual.
        fn f<T: Numeric>(t: T) -> T {
            t.sin() * t.exp() + t.powi(3)
        }
        let x0 = 0.8_f64;
        let j = f(Jet::<f64, 2>::variable(x0));
        let d = f(Dual::<f64>::variable(x0));
        assert!(f64::abs(j.value() - d.value) < TOL);
        assert!(f64::abs(j.coeffs[1] - d.deriv) < TOL);
    }

    #[test]
    fn test_generic_over_numeric() {
        // The same function runs with a plain float or with a Jet.
        fn g<T: Numeric>(t: T) -> T {
            t.powi(3) + T::from_f64(2.0) * t
        }
        let x0 = 1.5_f64;
        let plain = g(x0);
        let j = g(Jet::<f64, 4>::variable(x0));
        assert!(f64::abs(j.value() - plain) < TOL);
        assert!(f64::abs(j.derivative(1) - (3.0 * x0 * x0 + 2.0)) < TOL); // f'
        assert!(f64::abs(j.derivative(2) - 6.0 * x0) < TOL); // f''
        assert!(f64::abs(j.derivative(3) - 6.0) < TOL); // f'''
    }

    #[test]
    fn test_generic_over_f32() {
        // Jet is generic over the scalar; here it carries f32.
        let y = Jet::<f32, 4>::variable(2.0).powi(3);
        assert!(f32::abs(y.derivative(0) - 8.0) < TOL_F32);
        assert!(f32::abs(y.derivative(1) - 12.0) < TOL_F32);
        assert!(f32::abs(y.derivative(2) - 12.0) < TOL_F32);
        assert!(f32::abs(y.derivative(3) - 6.0) < TOL_F32);
    }

    #[test]
    fn test_single_coefficient_is_scalar() {
        // N = 1 carries only the value and behaves like the bare scalar.
        let y = Jet::<f64, 1>::constant(2.0) * Jet::<f64, 1>::constant(3.0);
        assert!(f64::abs(y.value() - 6.0) < TOL);
        assert!(f64::abs(Jet::<f64, 1>::constant(1.0).exp().value() - f64::exp(1.0)) < TOL);
    }

    #[test]
    fn test_constant_has_zero_higher_coeffs() {
        let c = Jet::<f64, 4>::constant(2.0);
        for k in 1..4 {
            assert!(f64::abs(c.coeffs[k]) < TOL);
        }
    }

    #[test]
    fn test_sqrt_zero_blows_up() {
        // the derivative of sqrt at 0 is unbounded, while the value stays finite
        let y = Jet::<f64, 3>::variable(0.0).sqrt();
        assert!(f64::abs(y.value()) < TOL);
        assert!(y.coeffs[1].is_infinite());
        // is_finite reflects the value only
        assert!(y.is_finite());
    }

    #[test]
    fn test_atan2_orders() {
        // atan2(y, 1) = atan(y): f'=1/(1+y²), f''=−2y/(1+y²)², f'''=(6y²−2)/(1+y²)³
        let y0 = 0.5_f64;
        let j = Jet::<f64, 4>::variable(y0).atan2(Jet::constant(1.0));
        let d = 1.0 + y0 * y0;
        assert!(f64::abs(j.derivative(0) - y0.atan()) < TOL);
        assert!(f64::abs(j.derivative(1) - 1.0 / d) < TOL);
        assert!(f64::abs(j.derivative(2) - (-2.0 * y0) / (d * d)) < TOL);
        assert!(f64::abs(j.derivative(3) - (6.0 * y0 * y0 - 2.0) / (d * d * d)) < TOL);
    }

    #[test]
    fn test_atan2_matches_dual_first_order() {
        let (y0, x0) = (1.3_f64, 0.7_f64);
        let j = Jet::<f64, 2>::variable(y0).atan2(Jet::constant(x0));
        let d = Dual::variable(y0).atan2(Dual::constant(x0));
        assert!(f64::abs(j.value() - d.value) < TOL);
        assert!(f64::abs(j.coeffs[1] - d.deriv) < TOL);
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
        let f = Cubic;
        // plain f64 (finite-difference path): 4*8 - 3*4 = 20
        assert!(f64::abs(f.eval(2.0_f64) - 20.0) < 1e-12);
        // Dual: f'(x) = 12x^2 - 6x = 36 at x = 2
        assert!(f64::abs(f.eval(Dual::variable(2.0_f64)).deriv - 36.0) < 1e-12);
        // HyperDual: f''(x) = 24x - 6 = 42 at x = 2
        assert!(f64::abs(f.eval(HyperDual::variable(2.0_f64)).eps1eps2 - 42.0) < 1e-12);
        // Jet: f'''(x) = 24
        assert!(f64::abs(f.eval(Jet::<f64, 4>::variable(2.0_f64)).derivative(3) - 24.0) < 1e-9);
    }

    // g(x, y, z) = y*sin(x) + 2*x*e^z, hand-written over the scalar.
    struct Mixed;
    impl ScalarFnN<3> for Mixed {
        fn eval<S: Numeric>(&self, v: &[S; 3]) -> S {
            v[1] * v[0].sin() + S::from_f64(2.0) * v[0] * v[2].exp()
        }
    }

    #[test]
    fn multivariable_partial_via_seeding() {
        let g = Mixed;
        let point = [1.0_f64, 2.0, 0.5];
        let expected = 2.0 * f64::sin(1.0) + 2.0 * f64::exp(0.5);
        assert!(f64::abs(g.eval(&point) - expected) < 1e-12);

        // partial dg/dx via Dual seeding of index 0:
        // dg/dx = y*cos(x) + 2*e^z = 2*cos(1) + 2*e^0.5
        let seeded = [
            Dual::variable(1.0_f64),
            Dual::constant(2.0),
            Dual::constant(0.5),
        ];
        let expected_dx = 2.0 * f64::cos(1.0) + 2.0 * f64::exp(0.5);
        assert!(f64::abs(g.eval(&seeded).deriv - expected_dx) < 1e-12);
    }

    #[test]
    fn macro_single_var() {
        // f(x) = 4x^3 - 3x^2, authored via the macro.
        let f = scalar_fn!(|x| c(4.0) * x * x * x - c(3.0) * x * x);
        assert!(f64::abs(f.eval(2.0_f64) - 20.0) < 1e-12);
        assert!(f64::abs(f.eval(Dual::variable(2.0_f64)).deriv - 36.0) < 1e-12);
        assert!(f64::abs(f.eval(HyperDual::variable(2.0_f64)).eps1eps2 - 42.0) < 1e-12);
        assert!(f64::abs(f.eval(Jet::<f64, 4>::variable(2.0_f64)).derivative(3) - 24.0) < 1e-9);
    }

    #[test]
    fn macro_single_var_typed_param() {
        let f = scalar_fn!(|x: f64| c(2.0) * x.sin());
        assert!(f64::abs(f.eval(0.5_f64) - 2.0 * f64::sin(0.5)) < 1e-12);
    }

    #[test]
    fn macro_multivariable() {
        let f = scalar_fn!(|v: &[f64; 3]| v[1] * v[0].sin() + c(2.0) * v[0] * v[2].exp());
        let point = [1.0_f64, 2.0, 0.5];
        let expected = 2.0 * f64::sin(1.0) + 2.0 * f64::exp(0.5);
        assert!(f64::abs(f.eval(&point) - expected) < 1e-12);
    }

    #[test]
    fn macro_vector_valued() {
        // f(x, y) = [x*y, sin(y)]
        let f = scalar_fn_vec!(|v: &[f64; 2]| [v[0] * v[1], v[1].sin()]);
        let out = f.eval(&[3.0_f64, 0.5]);
        assert!(f64::abs(out[0] - 1.5) < 1e-12);
        assert!(f64::abs(out[1] - f64::sin(0.5)) < 1e-12);

        // a Jacobian column via Dual: d/dx [x*y, sin(y)] = [y, 0] at (3, 0.5)
        let seeded = [Dual::variable(3.0_f64), Dual::constant(0.5)];
        let col = f.eval(&seeded);
        assert!(f64::abs(col[0].deriv - 0.5) < 1e-12);
        assert!(f64::abs(col[1].deriv) < 1e-12);
    }
}
