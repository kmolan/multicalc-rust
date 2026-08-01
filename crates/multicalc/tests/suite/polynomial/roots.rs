//! Polynomial root tests: the exact formulas up to the fourth power, the range every root sits in,
//! and the counting and halving that handle any degree.

use multicalc::error::PolynomialError;
use multicalc::polynomial::Polynomial;
use multicalc::scalar::Dual;

/// x(x - 1)(x - 2)(x - 4), whose roots are 0, 1, 2 and 4.
const QUARTIC_FOUR_ROOTS: [f64; 5] = [0.0, -8.0, 14.0, -7.0, 1.0];

/// (x - 2)²(x + 1), which has a doubled root at 2 and a single one at -1.
const CUBIC_REPEATED: [f64; 4] = [4.0, 0.0, -3.0, 1.0];

fn assert_close(found: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(found.len(), expected.len());
    for (found, expected) in found.iter().zip(expected.iter()) {
        assert!(
            (found - expected).abs() < tolerance,
            "expected {expected}, found {found}"
        );
    }
}

// ---- exact formulas ---------------------------------------------------------

#[test]
fn linear_root() {
    // 2x - 6
    let p: Polynomial<2> = Polynomial::new([-6.0, 2.0]);
    assert_close(p.real_roots().unwrap().as_slice(), &[3.0], 1e-12);
}

#[test]
fn quadratic_two_roots() {
    // (x - 1)(x - 2)
    let p: Polynomial<3> = Polynomial::new([2.0, -3.0, 1.0]);
    assert_close(p.real_roots().unwrap().as_slice(), &[1.0, 2.0], 1e-12);
}

#[test]
fn quadratic_repeated_root() {
    // (x - 3)², which reports its root twice.
    let p: Polynomial<3> = Polynomial::new([9.0, -6.0, 1.0]);
    assert_close(p.real_roots().unwrap().as_slice(), &[3.0, 3.0], 1e-12);
}

#[test]
fn quadratic_no_real_roots() {
    // x² + 1
    let p: Polynomial<3> = Polynomial::new([1.0, 0.0, 1.0]);
    assert!(p.real_roots().unwrap().is_empty());
}

#[test]
fn quadratic_survives_a_dominant_linear_term() {
    // x² + 1e8·x + 1. Subtracting the square root from the x coefficient directly would wipe out
    // every digit of the smaller root.
    let p: Polynomial<3> = Polynomial::new([1.0, 1e8, 1.0]);

    let roots = p.real_roots().unwrap();
    assert_eq!(roots.len(), 2);
    for (found, expected) in roots.as_slice().iter().zip([-1e8 + 1e-8, -1e-8]) {
        assert!(((found - expected) / expected).abs() < 1e-9);
    }
}

#[test]
fn cubic_three_real_roots() {
    // (x - 1)(x - 2)(x - 3)
    let p: Polynomial<4> = Polynomial::new([-6.0, 11.0, -6.0, 1.0]);
    assert_close(p.real_roots().unwrap().as_slice(), &[1.0, 2.0, 3.0], 1e-10);
}

#[test]
fn cubic_one_real_root() {
    // x³ + x + 1
    let p: Polynomial<4> = Polynomial::new([1.0, 1.0, 0.0, 1.0]);
    let roots = p.real_roots().unwrap();
    assert_close(roots.as_slice(), &[-0.682_327_803_828_019_3], 1e-10);
}

#[test]
fn cubic_repeated_root() {
    // (x - 2)²(x + 1), so 2 comes back twice.
    let p: Polynomial<4> = Polynomial::new(CUBIC_REPEATED);
    assert_close(p.real_roots().unwrap().as_slice(), &[-1.0, 2.0, 2.0], 1e-10);
}

#[test]
fn quartic_four_real_roots() {
    let p: Polynomial<5> = Polynomial::new(QUARTIC_FOUR_ROOTS);
    assert_close(
        p.real_roots().unwrap().as_slice(),
        &[0.0, 1.0, 2.0, 4.0],
        1e-9,
    );
}

#[test]
fn quartic_two_real_roots() {
    // (x² + 1)(x - 1)(x + 2), whose only real roots are -2 and 1.
    let p: Polynomial<5> = Polynomial::new([-2.0, 1.0, -1.0, 1.0, 1.0]);
    assert_close(p.real_roots().unwrap().as_slice(), &[-2.0, 1.0], 1e-10);
}

#[test]
fn quartic_no_real_roots() {
    // (x² + 1)(x² + 4)
    let p: Polynomial<5> = Polynomial::new([4.0, 0.0, 5.0, 0.0, 1.0]);
    assert!(p.real_roots().unwrap().is_empty());
}

#[test]
fn quartic_biquadratic() {
    // x⁴ - 5x² + 4, which has no odd powers and so takes the shorter route.
    let p: Polynomial<5> = Polynomial::new([4.0, 0.0, -5.0, 0.0, 1.0]);
    assert_close(
        p.real_roots().unwrap().as_slice(),
        &[-2.0, -1.0, 1.0, 2.0],
        1e-10,
    );
}

#[test]
fn closed_form_roots_differentiate() {
    // x² - 3x + c at c = 2, whose roots are 1 and 2. Nudging c moves a root by -1 divided by the
    // slope there, which is 1 at the first root and -1 at the second.
    let p: Polynomial<3, Dual<f64>> = Polynomial::new([
        Dual::variable(2.0),
        Dual::constant(-3.0),
        Dual::constant(1.0),
    ]);

    let roots = p.real_roots().unwrap();
    assert_eq!(roots.len(), 2);
    for (found, (root, movement)) in roots.as_slice().iter().zip([(1.0, 1.0), (2.0, -1.0)]) {
        assert!((found.value - root).abs() < 1e-9);
        assert!((found.deriv - movement).abs() < 1e-9);
    }
}

// ---- the error paths --------------------------------------------------------

#[test]
fn leading_coefficient_zero_is_an_error() {
    let expected = Some(PolynomialError::LeadingCoefficientZero);
    assert_eq!(
        Polynomial::<2>::new([1.0, 0.0]).real_roots().err(),
        expected
    );
    assert_eq!(
        Polynomial::<3>::new([1.0, 1.0, 0.0]).real_roots().err(),
        expected
    );
    assert_eq!(
        Polynomial::<4>::new([1.0, 1.0, 1.0, 0.0])
            .real_roots()
            .err(),
        expected
    );
    assert_eq!(
        Polynomial::<5>::new([1.0, 1.0, 1.0, 1.0, 0.0])
            .real_roots()
            .err(),
        expected
    );
}

#[test]
fn non_finite_coefficient_is_an_error() {
    let expected = Some(PolynomialError::NonFinite);
    assert_eq!(
        Polynomial::<2>::new([f64::NAN, 2.0]).real_roots().err(),
        expected
    );
    assert_eq!(
        Polynomial::<3>::new([f64::INFINITY, 1.0, 1.0])
            .real_roots()
            .err(),
        expected
    );
    assert_eq!(
        Polynomial::<4>::new([1.0, f64::NAN, 1.0, 1.0])
            .real_roots()
            .err(),
        expected
    );
    assert_eq!(
        Polynomial::<5>::new([1.0, 1.0, f64::NEG_INFINITY, 1.0, 1.0])
            .real_roots()
            .err(),
        expected
    );
}

// ---- the range every root sits in -------------------------------------------

#[test]
fn cauchy_bound_contains_every_root() {
    let quadratic: Polynomial<3> = Polynomial::new([2.0, -3.0, 1.0]);
    let cubic: Polynomial<4> = Polynomial::new([-6.0, 11.0, -6.0, 1.0]);
    let quartic: Polynomial<5> = Polynomial::new(QUARTIC_FOUR_ROOTS);

    for root in quadratic.real_roots().unwrap().as_slice() {
        assert!(root.abs() <= quadratic.cauchy_root_bound().unwrap());
    }
    for root in cubic.real_roots().unwrap().as_slice() {
        assert!(root.abs() <= cubic.cauchy_root_bound().unwrap());
    }
    for root in quartic.real_roots().unwrap().as_slice() {
        assert!(root.abs() <= quartic.cauchy_root_bound().unwrap());
    }

    // Every coefficient zero leaves nothing to bound.
    assert_eq!(
        Polynomial::<4>::zeros().cauchy_root_bound().err(),
        Some(PolynomialError::LeadingCoefficientZero)
    );
}

// ---- counting and halving ---------------------------------------------------

#[test]
fn count_real_roots_matches_the_closed_form() {
    let p: Polynomial<5> = Polynomial::new(QUARTIC_FOUR_ROOTS);
    let bound = p.cauchy_root_bound().unwrap();

    assert_eq!(p.count_real_roots(-bound, bound).unwrap(), 4);
    assert_eq!(p.real_roots().unwrap().len(), 4);

    // Roots at 0 and 1, then 2 and 4, then 4 on its own.
    assert_eq!(p.count_real_roots(-1.0, 1.5).unwrap(), 2);
    assert_eq!(p.count_real_roots(1.5, 5.0).unwrap(), 2);
    assert_eq!(p.count_real_roots(2.5, 5.0).unwrap(), 1);
    assert_eq!(p.count_real_roots(4.5, 5.0).unwrap(), 0);
}

#[test]
fn count_real_roots_counts_distinct_roots() {
    // (x - 2)²(x + 1): counting says two places, the closed form lists three roots.
    let p: Polynomial<4> = Polynomial::new(CUBIC_REPEATED);
    assert_eq!(p.count_real_roots(-5.0, 5.0).unwrap(), 2);
    assert_eq!(p.real_roots().unwrap().len(), 3);
}

#[test]
fn real_roots_in_matches_the_closed_form_quartic() {
    let p: Polynomial<5> = Polynomial::new(QUARTIC_FOUR_ROOTS);
    let bound = p.cauchy_root_bound().unwrap();

    let narrowed = p.real_roots_in(-bound, bound, 1e-11, 1000).unwrap();
    assert_close(
        narrowed.as_slice(),
        p.real_roots().unwrap().as_slice(),
        1e-9,
    );
}

#[test]
fn real_roots_in_finds_a_degree_six_polynomial() {
    // (x + 3)(x + 1)(x - 0.5)(x - 2)(x - 4)(x - 7)
    let p: Polynomial<7> = Polynomial::new([84.0, -131.0, -126.5, 104.5, 5.5, -9.5, 1.0]);

    let roots = p.real_roots_in(-10.0, 10.0, 1e-10, 1000).unwrap();
    assert_close(roots.as_slice(), &[-3.0, -1.0, 0.5, 2.0, 4.0, 7.0], 1e-8);
}

#[test]
fn real_roots_in_reports_running_out_of_steps() {
    let p: Polynomial<5> = Polynomial::new(QUARTIC_FOUR_ROOTS);
    assert!(matches!(
        p.real_roots_in(-15.0, 15.0, 1e-12, 2),
        Err(PolynomialError::DidNotConverge { .. })
    ));
}
