//! Multivariate polynomial tests: evaluation, the symbolic derivatives checked against autodiff,
//! substitution, products, term collection, the bridge to the dense type, and the literal macros.

use multicalc::error::PolynomialError;
use multicalc::polynomial::{MultivariatePolynomial, MultivariateTerm, Polynomial};
use multicalc::scalar::Dual;
use multicalc::{multivariate_polynomial, polynomial};

/// `3x²y + 2xy - 1`, the polynomial most of these work on.
fn sample() -> MultivariatePolynomial<2, 3> {
    MultivariatePolynomial::try_from_terms(&[
        MultivariateTerm::new(3.0, [2, 1]),
        MultivariateTerm::new(2.0, [1, 1]),
        MultivariateTerm::new(-1.0, [0, 0]),
    ])
    .unwrap()
}

#[test]
fn evaluate_matches_hand_expansion() {
    // At x = 1.5, y = -2: 3·2.25·(-2) + 2·1.5·(-2) - 1 = -13.5 - 6 - 1
    assert!((sample().evaluate(&[1.5, -2.0]) + 20.5).abs() < 1e-12);
}

#[test]
fn partial_derivative_matches_autodiff() {
    // The same polynomial carrying dual numbers, so differentiating through `evaluate` is available
    // to check the symbolic route against.
    let dual: MultivariatePolynomial<2, 3, Dual<f64>> = MultivariatePolynomial::try_from_terms(&[
        MultivariateTerm::new(Dual::constant(3.0), [2, 1]),
        MultivariateTerm::new(Dual::constant(2.0), [1, 1]),
        MultivariateTerm::new(Dual::constant(-1.0), [0, 0]),
    ])
    .unwrap();

    let (x, y) = (1.5, -2.0);
    let symbolic = sample();

    // Seeding x reads the slope in x, and seeding y reads the slope in y.
    let in_x = dual.evaluate(&[Dual::variable(x), Dual::constant(y)]);
    let in_y = dual.evaluate(&[Dual::constant(x), Dual::variable(y)]);

    assert!((in_x.deriv - symbolic.partial_derivative(0).unwrap().evaluate(&[x, y])).abs() < 1e-12);
    assert!((in_y.deriv - symbolic.partial_derivative(1).unwrap().evaluate(&[x, y])).abs() < 1e-12);

    // And against the values worked out by hand: 6xy + 2y, and 3x² + 2x.
    assert!((in_x.deriv + 22.0).abs() < 1e-12);
    assert!((in_y.deriv - 9.75).abs() < 1e-12);
}

#[test]
fn gradient_at_matches_the_partial_derivatives() {
    let p = sample();
    let point = [1.5, -2.0];

    let gradient = p.gradient_at(&point);
    for variable in 0..2 {
        let separately = p.partial_derivative(variable).unwrap().evaluate(&point);
        assert!((gradient[variable] - separately).abs() < 1e-12);
    }
}

#[test]
fn partial_antiderivative_then_partial_derivative_returns_original() {
    let p = sample();

    for variable in 0..2 {
        let round_tripped = p
            .partial_antiderivative(variable)
            .unwrap()
            .partial_derivative(variable)
            .unwrap();
        for point in [[1.5, -2.0], [0.0, 0.0], [-0.75, 3.25]] {
            assert!((round_tripped.evaluate(&point) - p.evaluate(&point)).abs() < 1e-12);
        }
    }
}

#[test]
fn substitute_fixes_one_variable() {
    let p = sample();
    let fixed = p.substitute(1, -2.0).unwrap();

    // With y pinned, the value no longer depends on what is passed for it.
    for x in [-1.0, 0.0, 2.5] {
        assert!((fixed.evaluate(&[x, 99.0]) - p.evaluate(&[x, -2.0])).abs() < 1e-12);
    }
    assert_eq!(fixed.degree_in(1), Some(0));
}

#[test]
fn univariate_bridge_round_trips() {
    let dense: Polynomial<4> = Polynomial::new([1.5, 0.0, -2.0, 0.25]);

    let terms = MultivariatePolynomial::<1, 4>::from_univariate(&dense).unwrap();
    // The zero coefficient does not become a term.
    assert_eq!(terms.len(), 3);

    let back: Polynomial<4> = terms.to_univariate().unwrap();
    for (found, expected) in back.coefficients().iter().zip(*dense.coefficients()) {
        assert!((found - expected).abs() < 1e-12);
    }

    // A power past the size asked for has nowhere to go.
    assert_eq!(
        terms.to_univariate::<3>().err(),
        Some(PolynomialError::DegreeOverflow)
    );
}

#[test]
fn multiply_into_matches_evaluating_both() {
    let left = sample();
    let right: MultivariatePolynomial<2, 2> = MultivariatePolynomial::try_from_terms(&[
        MultivariateTerm::new(1.0, [1, 0]),
        MultivariateTerm::new(1.0, [0, 1]),
    ])
    .unwrap();

    // Three terms by two terms, so six before matching ones are merged.
    let product = left.multiply_into::<2, 6>(&right).unwrap();
    for point in [[1.5, -2.0], [0.0, 1.0], [-0.5, 3.0]] {
        let expected = left.evaluate(&point) * right.evaluate(&point);
        assert!((product.evaluate(&point) - expected).abs() < 1e-12);
    }
}

#[test]
fn multiply_into_reports_a_small_output() {
    let left = sample();
    let right: MultivariatePolynomial<2, 2> = MultivariatePolynomial::try_from_terms(&[
        MultivariateTerm::new(1.0, [1, 0]),
        MultivariateTerm::new(1.0, [0, 1]),
    ])
    .unwrap();

    assert_eq!(
        left.multiply_into::<2, 3>(&right).err(),
        Some(PolynomialError::CapacityExceeded)
    );
}

#[test]
fn add_into_merges_matching_terms() {
    let left = sample();
    let right: MultivariatePolynomial<2, 1> =
        MultivariatePolynomial::try_from_terms(&[MultivariateTerm::new(-3.0, [2, 1])]).unwrap();

    // The x²y terms cancel, leaving 2xy - 1.
    let sum = left.add_into::<1, 4>(&right).unwrap();
    assert_eq!(sum.len(), 2);
    for point in [[1.5, -2.0], [2.0, 0.5]] {
        let expected = left.evaluate(&point) + right.evaluate(&point);
        assert!((sum.evaluate(&point) - expected).abs() < 1e-12);
    }
}

#[test]
fn variable_out_of_range_is_an_error() {
    let p = sample();
    assert_eq!(
        p.partial_derivative(2).err(),
        Some(PolynomialError::VariableOutOfRange)
    );
    assert_eq!(
        p.partial_antiderivative(2).err(),
        Some(PolynomialError::VariableOutOfRange)
    );
    assert_eq!(
        p.substitute(9, 1.0).err(),
        Some(PolynomialError::VariableOutOfRange)
    );
    // The query reports nothing rather than failing.
    assert_eq!(p.degree_in(2), None);
}

#[test]
fn collect_like_terms_merges_and_drops_zeros() {
    let mut p: MultivariatePolynomial<2, 4> = MultivariatePolynomial::try_from_terms(&[
        MultivariateTerm::new(1.0, [1, 0]),
        MultivariateTerm::new(2.0, [1, 0]),
        MultivariateTerm::new(-3.0, [1, 0]),
        MultivariateTerm::new(5.0, [0, 1]),
    ])
    .unwrap();

    p.collect_like_terms();
    // The three x terms add to nothing and disappear, leaving 5y.
    assert_eq!(p.len(), 1);
    assert_eq!(p.terms()[0].exponents(), &[0, 1]);
    assert!((p.terms()[0].coefficient() - 5.0).abs() < 1e-12);
}

#[test]
fn total_degree_and_degree_in() {
    let p = sample();
    assert_eq!(p.total_degree(), Some(3));
    assert_eq!(p.degree_in(0), Some(2));
    assert_eq!(p.degree_in(1), Some(1));

    let empty = MultivariatePolynomial::<2, 3>::new();
    assert!(empty.is_empty());
    assert_eq!(empty.total_degree(), None);
    assert_eq!(empty.degree_in(0), None);
}

#[test]
fn runs_in_f32() {
    let p: MultivariatePolynomial<2, 3, f32> = MultivariatePolynomial::try_from_terms(&[
        MultivariateTerm::new(3.0, [2, 1]),
        MultivariateTerm::new(2.0, [1, 1]),
        MultivariateTerm::new(-1.0, [0, 0]),
    ])
    .unwrap();
    assert!((p.evaluate(&[1.5, -2.0]) + 20.5).abs() < 1e-5);
}

#[test]
fn macros_build_the_same_values() {
    assert_eq!(
        polynomial![1.0, -2.0, 3.0],
        Polynomial::new([1.0, -2.0, 3.0])
    );

    let written: MultivariatePolynomial<2, 2> =
        multivariate_polynomial![(2.5, [2, 3]), (-1.0, [0, 1])].unwrap();
    let explicit: MultivariatePolynomial<2, 2> = MultivariatePolynomial::try_from_terms(&[
        MultivariateTerm::new(2.5, [2, 3]),
        MultivariateTerm::new(-1.0, [0, 1]),
    ])
    .unwrap();
    assert_eq!(written, explicit);
}
