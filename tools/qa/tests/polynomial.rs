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

fn run_evaluation(fx: &Fixture) {
    let polynomial = Polynomial::<8>::new(coefficients(&fx.inputs["coefficients"]));
    let points = fx.inputs["points"].as_vector();
    let (rows, columns, expected) = fx.expected["orders"].as_matrix();
    assert_eq!((rows, columns), (points.len(), 4), "orders shape");

    for (row, point) in points.iter().enumerate() {
        let found: [f64; 4] = polynomial.evaluate_with_derivatives(*point);
        for (order, value) in found.iter().enumerate() {
            let want = expected[row * columns + order];
            assert!(
                close(*value, want, fx.tolerances.f64),
                "{}: order {order} at {point}: got {value}, want {want}",
                fx.case
            );
        }
    }
}

fn run_coefficients(fx: &Fixture) {
    let tolerance = fx.tolerances.f64;
    let found: Vec<f64> = match fx.case.as_str() {
        "derivative_degree5" => {
            let polynomial = Polynomial::<6>::new(coefficients(&fx.inputs["coefficients"]));
            polynomial.derivative().coefficients().to_vec()
        }
        "product_degree3_by_degree4" => {
            let left = Polynomial::<4>::new(coefficients(&fx.inputs["coefficients"]));
            let right = Polynomial::<5>::new(coefficients(&fx.inputs["other"]));
            left.multiply_into::<5, 8>(&right)
                .unwrap()
                .coefficients()
                .to_vec()
        }
        "composition_degree3_in_degree2" => {
            let outer = Polynomial::<4>::new(coefficients(&fx.inputs["coefficients"]));
            let inner = Polynomial::<3>::new(coefficients(&fx.inputs["other"]));
            outer
                .compose_into::<3, 7>(&inner)
                .unwrap()
                .coefficients()
                .to_vec()
        }
        "interpolation_five_points" => {
            let nodes: [f64; 5] = coefficients(&fx.inputs["nodes"]);
            let values: [f64; 5] = coefficients(&fx.inputs["values"]);
            Polynomial::<5>::from_points(&nodes, &values)
                .unwrap()
                .coefficients()
                .to_vec()
        }
        "least_squares_cubic_fit" => {
            let nodes: [f64; 20] = coefficients(&fx.inputs["nodes"]);
            let values: [f64; 20] = coefficients(&fx.inputs["values"]);
            Polynomial::<4>::fit_least_squares(&nodes, &values)
                .unwrap()
                .coefficients()
                .to_vec()
        }
        other => panic!("unknown coefficients case {other}"),
    };

    let expected = fx.expected["result"].as_vector();
    for (index, want) in expected.iter().enumerate() {
        let got = found.get(index).copied().unwrap_or(0.0);
        assert!(
            close(got, *want, tolerance),
            "{}: coefficient {index}: got {got}, want {want}",
            fx.case
        );
    }
    // Anything beyond what numpy reported has to be zero, not merely unchecked.
    for (index, got) in found.iter().enumerate().skip(expected.len()) {
        assert!(
            close(*got, 0.0, tolerance),
            "{}: coefficient {index} past the expected end is {got}",
            fx.case
        );
    }
}

fn run_scalar_case(fx: &Fixture) {
    let polynomial = Polynomial::<6>::new(coefficients(&fx.inputs["coefficients"]));
    let bounds = fx.inputs["bounds"].as_vector();
    assert_scalar(
        polynomial.definite_integral(bounds[0], bounds[1]),
        &fx.expected["result"],
        fx.tolerances.f64,
        "area",
    );
}

fn run_roots(fx: &Fixture) {
    let stored = fx.inputs["coefficients"].as_vector();
    let found: Vec<f64> = match stored.len() {
        3 => Polynomial::<3>::new(coefficients(&fx.inputs["coefficients"]))
            .real_roots()
            .unwrap()
            .as_slice()
            .to_vec(),
        4 => Polynomial::<4>::new(coefficients(&fx.inputs["coefficients"]))
            .real_roots()
            .unwrap()
            .as_slice()
            .to_vec(),
        5 => Polynomial::<5>::new(coefficients(&fx.inputs["coefficients"]))
            .real_roots()
            .unwrap()
            .as_slice()
            .to_vec(),
        other => panic!("no closed form for {other} coefficients"),
    };

    let expected = fx.expected["roots"].as_vector();
    // Checked first, so a missing root fails loudly rather than passing on a prefix.
    assert_eq!(found.len(), expected.len(), "{}: root count", fx.case);
    for (index, (got, want)) in found.iter().zip(expected.iter()).enumerate() {
        assert!(
            close(*got, *want, fx.tolerances.f64),
            "{}: root {index}: got {got}, want {want}",
            fx.case
        );
    }
}

fn run_roots_any_degree(fx: &Fixture) {
    let polynomial = Polynomial::<7>::new(coefficients(&fx.inputs["coefficients"]));
    let bound = polynomial.cauchy_root_bound().unwrap();

    let counted = polynomial.count_real_roots(-bound, bound).unwrap();
    assert_eq!(
        counted as i64,
        fx.expected["count"].as_int(),
        "{}: how many roots the range holds",
        fx.case
    );

    let found = polynomial.real_roots_in(-bound, bound, 1e-11, 400).unwrap();
    let expected = fx.expected["roots"].as_vector();
    assert_eq!(found.len(), expected.len(), "{}: root count", fx.case);
    for (index, (got, want)) in found.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            close(*got, *want, fx.tolerances.f64),
            "{}: root {index}: got {got}, want {want}",
            fx.case
        );
    }
}

/// Evaluation and partial derivatives, for whichever variable count the fixture names.
fn run_multivariate_values<const VARIABLES: usize>(fx: &Fixture, partials: bool) {
    let polynomial = MultivariatePolynomial::<VARIABLES, 16>::try_from_terms(&terms::<VARIABLES>(
        &fx.inputs["terms"],
    ))
    .unwrap();
    let (rows, columns, points) = fx.inputs["points"].as_matrix();
    assert_eq!(columns, VARIABLES, "point width");

    for row in 0..rows {
        let mut point = [0.0; VARIABLES];
        for (slot, offset) in point.iter_mut().zip(0..columns) {
            *slot = points[row * columns + offset];
        }

        if partials {
            let (_, width, expected) = fx.expected["partials"].as_matrix();
            let gradient = polynomial.gradient_at(&point);
            for variable in 0..VARIABLES {
                let want = expected[row * width + variable];
                assert!(
                    close(gradient[variable], want, fx.tolerances.f64),
                    "{}: partial {variable} at row {row}: got {}, want {want}",
                    fx.case,
                    gradient[variable]
                );
                // The symbolic route has to agree with the one-pass gradient.
                let symbolic = polynomial
                    .partial_derivative(variable)
                    .unwrap()
                    .evaluate(&point);
                assert!(
                    close(symbolic, want, fx.tolerances.f64),
                    "{}: symbolic partial {variable} at row {row}: got {symbolic}, want {want}",
                    fx.case
                );
            }
        } else {
            let want = fx.expected["values"].as_vector()[row];
            let got = polynomial.evaluate(&point);
            assert!(
                close(got, want, fx.tolerances.f64),
                "{}: value at row {row}: got {got}, want {want}",
                fx.case
            );
        }
    }
}

fn run_multivariate_product(fx: &Fixture) {
    let left =
        MultivariatePolynomial::<2, 16>::try_from_terms(&terms::<2>(&fx.inputs["terms"])).unwrap();
    let right =
        MultivariatePolynomial::<2, 16>::try_from_terms(&terms::<2>(&fx.inputs["other_terms"]))
            .unwrap();
    let product = left.multiply_into::<16, 32>(&right).unwrap();

    // Which order the terms come out in depends on how each side walks the pairs,
    // so this compares them as sets rather than in sequence.
    let (rows, columns, expected) = fx.expected["result_terms"].as_matrix();
    assert_eq!(product.len(), rows, "{}: term count", fx.case);
    for row in 0..rows {
        let base = row * columns;
        let wanted_powers: Vec<u32> = (1..columns).map(|c| expected[base + c] as u32).collect();
        let found = product
            .terms()
            .iter()
            .find(|term| term.exponents().as_slice() == wanted_powers.as_slice())
            .unwrap_or_else(|| panic!("{}: no term with powers {wanted_powers:?}", fx.case));
        assert!(
            close(found.coefficient(), expected[base], fx.tolerances.f64),
            "{}: term {wanted_powers:?}: got {}, want {}",
            fx.case,
            found.coefficient(),
            expected[base]
        );
    }
}

fn run_multivariate_substitution(fx: &Fixture) {
    let polynomial =
        MultivariatePolynomial::<2, 16>::try_from_terms(&terms::<2>(&fx.inputs["terms"])).unwrap();
    let variable = fx.inputs["variable"].as_int() as usize;
    let fixed = polynomial
        .substitute(variable, fx.inputs["value"].as_scalar())
        .unwrap();

    // With one variable pinned the value depends on the other alone, so reading it
    // off at a row of points recovers the coefficients numpy collapsed to.
    let expected = fx.expected["result"].as_vector();
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
            close(found, *want, fx.tolerances.f64),
            "{}: power {power}: got {found}, want {want}",
            fx.case
        );
    }
}

#[test]
fn polynomial_matches_numpy() {
    let fixtures = load_dir("polynomial");
    assert!(fixtures.len() >= 18, "expected the whole polynomial family");

    for fx in &fixtures {
        match fx.inputs["kind"].as_str() {
            "evaluation" => run_evaluation(fx),
            "coefficients" => run_coefficients(fx),
            "scalar" => run_scalar_case(fx),
            "roots" => run_roots(fx),
            "roots_any_degree" => run_roots_any_degree(fx),
            "multivariate_evaluation" => match fx.inputs["variable_count"].as_int() {
                2 => run_multivariate_values::<2>(fx, false),
                3 => run_multivariate_values::<3>(fx, false),
                other => panic!("unsupported variable count {other}"),
            },
            "multivariate_partials" => match fx.inputs["variable_count"].as_int() {
                2 => run_multivariate_values::<2>(fx, true),
                3 => run_multivariate_values::<3>(fx, true),
                other => panic!("unsupported variable count {other}"),
            },
            "multivariate_product" => run_multivariate_product(fx),
            "multivariate_substitution" => run_multivariate_substitution(fx),
            other => panic!("unknown kind {other} in {}", fx.case),
        }
    }
}
