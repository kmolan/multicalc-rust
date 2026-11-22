#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks the polynomial module against numpy goldens.
//!
//! Every fixture names a `kind` saying what was computed, so one directory covers
//! evaluation, calculus, products, composition, interpolation, fitting, exact real
//! roots, and polynomials in several variables.
//!
//! The multivariate cases are the interesting ones: numpy works from a dense grid
//! of coefficients and multicalc works from a list of terms, so agreement means two
//! different layouts produce the same numbers rather than the same code running
//! twice.

use multicalc::polynomial::{MultivariatePolynomial, MultivariateTerm, Polynomial};
use multicalc_qa::load::*;
use multicalc_qa::schema::*;

/// A fixed-size coefficient array from a stored vector, padded with zeros.
#[must_use]
fn coefficients<const N: usize>(value: &Value) -> [f64; N] {
    let data = value.as_vector();
    assert!(data.len() <= N, "fixture holds {} coefficients", data.len());
    let mut out = [0.0; N];
    for (slot, found) in out.iter_mut().zip(data.iter()) {
        *slot = *found;
    }
    out
}

/// A sparse term list from the stored `terms` matrix: coefficient first, then a power per variable.
#[must_use]
fn terms<const VARIABLES: usize>(value: &Value) -> Vec<MultivariateTerm<VARIABLES, f64>> {
    let (rows, columns, data) = value.as_matrix();
    assert_eq!(columns, VARIABLES + 1, "term row width");
    (0..rows)
        .map(|row| {
            let base = row * columns;
            let mut exponents = [0_u32; VARIABLES];
            for (slot, offset) in exponents.iter_mut().zip(1..columns) {
                *slot = data[base + offset] as u32;
            }
            MultivariateTerm::new(data[base], exponents)
        })
        .collect()
}

fn run_evaluation(fixture: &Fixture) {
    let polynomial = Polynomial::<8>::new(coefficients(&fixture.inputs["coefficients"]));
    let points = fixture.inputs["points"].as_vector();
    let (rows, columns, expected) = fixture.expected["orders"].as_matrix();
    assert_eq!((rows, columns), (points.len(), 4), "orders shape");

    for (row, point) in points.iter().enumerate() {
        let found: [f64; 4] = polynomial.evaluate_with_derivatives(*point);
        for (order, value) in found.iter().enumerate() {
            let want = expected[row * columns + order];
            assert!(
                close(*value, want, fixture.tolerances.f64),
                "{}: order {order} at {point}: got {value}, want {want}",
                fixture.case
            );
        }
    }
}

fn run_coefficients(fixture: &Fixture) {
    let tolerance = fixture.tolerances.f64;
    let found: Vec<f64> = match fixture.case.as_str() {
        "derivative_degree5" => {
            let polynomial = Polynomial::<6>::new(coefficients(&fixture.inputs["coefficients"]));
            polynomial.derivative().coefficients().to_vec()
        }
        "product_degree3_by_degree4" => {
            let left = Polynomial::<4>::new(coefficients(&fixture.inputs["coefficients"]));
            let right = Polynomial::<5>::new(coefficients(&fixture.inputs["other"]));
            left.multiply_into::<5, 8>(&right)
                .unwrap()
                .coefficients()
                .to_vec()
        }
        "composition_degree3_in_degree2" => {
            let outer = Polynomial::<4>::new(coefficients(&fixture.inputs["coefficients"]));
            let inner = Polynomial::<3>::new(coefficients(&fixture.inputs["other"]));
            outer
                .compose_into::<3, 7>(&inner)
                .unwrap()
                .coefficients()
                .to_vec()
        }
        "interpolation_five_points" => {
            let nodes: [f64; 5] = coefficients(&fixture.inputs["nodes"]);
            let values: [f64; 5] = coefficients(&fixture.inputs["values"]);
            Polynomial::<5>::from_points(&nodes, &values)
                .unwrap()
                .coefficients()
                .to_vec()
        }
        "least_squares_cubic_fit" => {
            let nodes: [f64; 20] = coefficients(&fixture.inputs["nodes"]);
            let values: [f64; 20] = coefficients(&fixture.inputs["values"]);
            Polynomial::<4>::fit_least_squares(&nodes, &values)
                .unwrap()
                .coefficients()
                .to_vec()
        }
        other => panic!("unknown coefficients case {other}"),
    };

    let expected = fixture.expected["result"].as_vector();
    for (index, want) in expected.iter().enumerate() {
        let got = found.get(index).copied().unwrap_or(0.0);
        assert!(
            close(got, *want, tolerance),
            "{}: coefficient {index}: got {got}, want {want}",
            fixture.case
        );
    }
    // Anything beyond what numpy reported has to be zero, not merely unchecked.
    for (index, got) in found.iter().enumerate().skip(expected.len()) {
        assert!(
            close(*got, 0.0, tolerance),
            "{}: coefficient {index} past the expected end is {got}",
            fixture.case
        );
    }
}

fn run_scalar_case(fixture: &Fixture) {
    let polynomial = Polynomial::<6>::new(coefficients(&fixture.inputs["coefficients"]));
    let bounds = fixture.inputs["bounds"].as_vector();
    assert_scalar(
        polynomial.definite_integral(bounds[0], bounds[1]),
        &fixture.expected["result"],
        fixture.tolerances.f64,
        "area",
    );
}

fn run_roots(fixture: &Fixture) {
    let stored = fixture.inputs["coefficients"].as_vector();
    let found: Vec<f64> = match stored.len() {
        3 => Polynomial::<3>::new(coefficients(&fixture.inputs["coefficients"]))
            .real_roots()
            .unwrap()
            .as_slice()
            .to_vec(),
        4 => Polynomial::<4>::new(coefficients(&fixture.inputs["coefficients"]))
            .real_roots()
            .unwrap()
            .as_slice()
            .to_vec(),
        5 => Polynomial::<5>::new(coefficients(&fixture.inputs["coefficients"]))
            .real_roots()
            .unwrap()
            .as_slice()
            .to_vec(),
        other => panic!("no closed form for {other} coefficients"),
    };

    let expected = fixture.expected["roots"].as_vector();
    // Checked first, so a missing root fails loudly rather than passing on a prefix.
    assert_eq!(found.len(), expected.len(), "{}: root count", fixture.case);
    for (index, (got, want)) in found.iter().zip(expected.iter()).enumerate() {
        assert!(
            close(*got, *want, fixture.tolerances.f64),
            "{}: root {index}: got {got}, want {want}",
            fixture.case
        );
    }
}

fn run_roots_any_degree(fixture: &Fixture) {
    let polynomial = Polynomial::<7>::new(coefficients(&fixture.inputs["coefficients"]));
    let bound = polynomial.cauchy_root_bound().unwrap();

    let counted = polynomial.count_real_roots(-bound, bound).unwrap();
    assert_eq!(
        counted as i64,
        fixture.expected["count"].as_int(),
        "{}: how many roots the range holds",
        fixture.case
    );

    let found = polynomial.real_roots_in(-bound, bound, 1e-11, 400).unwrap();
    let expected = fixture.expected["roots"].as_vector();
    assert_eq!(found.len(), expected.len(), "{}: root count", fixture.case);
    for (index, (got, want)) in found.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            close(*got, *want, fixture.tolerances.f64),
            "{}: root {index}: got {got}, want {want}",
            fixture.case
        );
    }
}

/// Evaluation and partial derivatives, for whichever variable count the fixture names.
fn run_multivariate_values<const VARIABLES: usize>(fixture: &Fixture, partials: bool) {
    let polynomial = MultivariatePolynomial::<VARIABLES, 16>::try_from_terms(&terms::<VARIABLES>(
        &fixture.inputs["terms"],
    ))
    .unwrap();
    let (rows, columns, points) = fixture.inputs["points"].as_matrix();
    assert_eq!(columns, VARIABLES, "point width");

    for row in 0..rows {
        let mut point = [0.0; VARIABLES];
        for (slot, offset) in point.iter_mut().zip(0..columns) {
            *slot = points[row * columns + offset];
        }

        if partials {
            let (_, width, expected) = fixture.expected["partials"].as_matrix();
            let gradient = polynomial.gradient_at(&point);
            for variable in 0..VARIABLES {
                let want = expected[row * width + variable];
                assert!(
                    close(gradient[variable], want, fixture.tolerances.f64),
                    "{}: partial {variable} at row {row}: got {}, want {want}",
                    fixture.case,
                    gradient[variable]
                );
                // The symbolic route has to agree with the one-pass gradient.
                let symbolic = polynomial
                    .partial_derivative(variable)
                    .unwrap()
                    .evaluate(&point);
                assert!(
                    close(symbolic, want, fixture.tolerances.f64),
                    "{}: symbolic partial {variable} at row {row}: got {symbolic}, want {want}",
                    fixture.case
                );
            }
        } else {
            let want = fixture.expected["values"].as_vector()[row];
            let got = polynomial.evaluate(&point);
            assert!(
                close(got, want, fixture.tolerances.f64),
                "{}: value at row {row}: got {got}, want {want}",
                fixture.case
            );
        }
    }
}

fn run_multivariate_product(fixture: &Fixture) {
    let left =
        MultivariatePolynomial::<2, 16>::try_from_terms(&terms::<2>(&fixture.inputs["terms"]))
            .unwrap();
    let right = MultivariatePolynomial::<2, 16>::try_from_terms(&terms::<2>(
        &fixture.inputs["other_terms"],
    ))
    .unwrap();
    let product = left.multiply_into::<16, 32>(&right).unwrap();

    // Which order the terms come out in depends on how each side walks the pairs,
    // so this compares them as sets rather than in sequence.
    let (rows, columns, expected) = fixture.expected["result_terms"].as_matrix();
    assert_eq!(product.len(), rows, "{}: term count", fixture.case);
    for row in 0..rows {
        let base = row * columns;
        let wanted_powers: Vec<u32> = (1..columns)
            .map(|col| expected[base + col] as u32)
            .collect();
        let found = product
            .terms()
            .iter()
            .find(|term| term.exponents().as_slice() == wanted_powers.as_slice())
            .unwrap_or_else(|| panic!("{}: no term with powers {wanted_powers:?}", fixture.case));
        assert!(
            close(found.coefficient(), expected[base], fixture.tolerances.f64),
            "{}: term {wanted_powers:?}: got {}, want {}",
            fixture.case,
            found.coefficient(),
            expected[base]
        );
    }
}

fn run_multivariate_substitution(fixture: &Fixture) {
    let polynomial =
        MultivariatePolynomial::<2, 16>::try_from_terms(&terms::<2>(&fixture.inputs["terms"]))
            .unwrap();
    let variable = fixture.inputs["variable"].as_int() as usize;
    let fixed = polynomial
        .substitute(variable, fixture.inputs["value"].as_scalar())
        .unwrap();

    // With one variable pinned the value depends on the other alone, so reading it
    // off at a row of points recovers the coefficients numpy collapsed to.
    let expected = fixture.expected["result"].as_vector();
    for (power, want) in expected.iter().enumerate() {
        // Compare through values rather than coefficients: the remaining variable is
        // still variable 0 here, and its powers are what numpy reported.
        let found = fixed
            .terms()
            .iter()
            .filter(|term| term.exponents()[0] as usize == power)
            .map(MultivariateTerm::coefficient)
            .sum::<f64>();
        assert!(
            close(found, *want, fixture.tolerances.f64),
            "{}: power {power}: got {found}, want {want}",
            fixture.case
        );
    }
}

#[test]
fn polynomial_matches_numpy() {
    let fixtures = load_dir("polynomial");
    assert!(fixtures.len() >= 18, "expected the whole polynomial family");

    for fixture in &fixtures {
        match fixture.inputs["kind"].as_str() {
            "evaluation" => run_evaluation(fixture),
            "coefficients" => run_coefficients(fixture),
            "scalar" => run_scalar_case(fixture),
            "roots" => run_roots(fixture),
            "roots_any_degree" => run_roots_any_degree(fixture),
            "multivariate_evaluation" => match fixture.inputs["variable_count"].as_int() {
                2 => run_multivariate_values::<2>(fixture, false),
                3 => run_multivariate_values::<3>(fixture, false),
                other => panic!("unsupported variable count {other}"),
            },
            "multivariate_partials" => match fixture.inputs["variable_count"].as_int() {
                2 => run_multivariate_values::<2>(fixture, true),
                3 => run_multivariate_values::<3>(fixture, true),
                other => panic!("unsupported variable count {other}"),
            },
            "multivariate_product" => run_multivariate_product(fixture),
            "multivariate_substitution" => run_multivariate_substitution(fixture),
            other => panic!("unknown kind {other} in {}", fixture.case),
        }
    }
}
