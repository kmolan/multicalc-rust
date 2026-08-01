//! Polynomial construction tests: building from roots, from a series, through given points, from
//! more samples than fit, and from values and derivatives at each end.

use multicalc::error::PolynomialError;
use multicalc::polynomial::Polynomial;
use multicalc::scalar::{Jet, Numeric};

// ---- from roots -------------------------------------------------------------

#[test]
fn from_roots_expands_correctly() {
    // (x - 1)(x - 2)(x - 3) = -6 + 11x - 6x² + x³
    let p = Polynomial::<4>::from_roots(&[1.0, 2.0, 3.0]).unwrap();
    for (found, expected) in p.coefficients().iter().zip([-6.0, 11.0, -6.0, 1.0]) {
        assert!((found - expected).abs() < 1e-12);
    }
}

#[test]
fn from_roots_round_trips_through_real_roots() {
    let roots = [-2.5, 0.75, 3.0, 4.25];
    let p = Polynomial::<5>::from_roots(&roots).unwrap();

    let found = p.real_roots().unwrap();
    assert_eq!(found.len(), 4);
    for (found, expected) in found.as_slice().iter().zip(roots) {
        assert!((found - expected).abs() < 1e-10);
    }
}

#[test]
fn from_roots_rejects_more_roots_than_fit() {
    // A cubic holds three roots at most.
    assert_eq!(
        Polynomial::<4>::from_roots(&[1.0, 2.0, 3.0, 4.0]).err(),
        Some(PolynomialError::DegreeOverflow)
    );
    assert_eq!(
        Polynomial::<4>::from_roots(&[1.0, f64::NAN]).err(),
        Some(PolynomialError::NonFinite)
    );
}

// ---- from a series ----------------------------------------------------------

#[test]
fn from_jet_reproduces_the_series() {
    // Expanding exp about zero gives 1, 1, 1/2, 1/6, 1/24.
    let expanded = Jet::<f64, 5>::variable(0.0).exp();
    let p = Polynomial::<5>::from_jet(&expanded).unwrap();

    let expected = [1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0];
    for (found, expected) in p.coefficients().iter().zip(expected) {
        assert!((found - expected).abs() < 1e-12);
    }

    // Five coefficients do not fit in four.
    assert_eq!(
        Polynomial::<4>::from_jet(&expanded).err(),
        Some(PolynomialError::DegreeOverflow)
    );
}

// ---- through given points ---------------------------------------------------

#[test]
fn from_points_passes_through_every_point() {
    // Five points off 2 - x + 0.5x² - 0.25x³ + 0.125x⁴.
    let source: Polynomial<5> = Polynomial::new([2.0, -1.0, 0.5, -0.25, 0.125]);
    let nodes = [-2.0, -0.5, 0.75, 1.5, 3.0];
    let values = nodes.map(|node| source.evaluate(node));

    let p = Polynomial::<5>::from_points(&nodes, &values).unwrap();
    for (node, value) in nodes.iter().zip(values.iter()) {
        assert!((p.evaluate(*node) - value).abs() < 1e-10);
    }
    // And it agrees away from the points too, since it is the same polynomial.
    assert!((p.evaluate(5.0) - source.evaluate(5.0)).abs() < 1e-9);
}

#[test]
fn from_points_rejects_a_duplicate_node() {
    let nodes = [0.0, 1.0, 1.0];
    let values = [1.0, 2.0, 3.0];
    assert_eq!(
        Polynomial::<3>::from_points(&nodes, &values).err(),
        Some(PolynomialError::DuplicateNode)
    );
}

#[test]
fn from_points_handles_a_wide_range() {
    // The same fit over positions spread across a thousand units, where the values run from 2 up to
    // 1.2e11. Working on shifted and stretched positions is what keeps the table itself from
    // degenerating; the misses stay at rounding against the size of the values involved.
    let source: Polynomial<5> = Polynomial::new([2.0, -1.0, 0.5, -0.25, 0.125]);
    let nodes = [0.0, 250.0, 500.0, 750.0, 1000.0];
    let values = nodes.map(|node| source.evaluate(node));

    let p = Polynomial::<5>::from_points(&nodes, &values).unwrap();
    let peak = values
        .iter()
        .fold(0.0_f64, |peak, value| peak.max(value.abs()));
    for (node, value) in nodes.iter().zip(values.iter()) {
        assert!((p.evaluate(*node) - value).abs() / peak < 1e-14);
    }
}

#[test]
fn chebyshev_nodes_lie_inside_the_range_and_ascend() {
    let nodes = Polynomial::<6>::chebyshev_nodes(-3.0, 5.0);

    for node in &nodes {
        assert!(*node > -3.0 && *node < 5.0);
    }
    for pair in nodes.windows(2) {
        assert!(pair[0] < pair[1]);
    }
}

// ---- through more points than fit -------------------------------------------

#[test]
fn fit_least_squares_recovers_an_exact_polynomial() {
    // Twenty samples straight off a cubic, so the closest fit is the cubic itself.
    let source: Polynomial<4> = Polynomial::new([1.5, -2.0, 0.75, 0.25]);
    let mut nodes = [0.0; 20];
    for (index, node) in nodes.iter_mut().enumerate() {
        *node = -3.0 + 0.4 * index as f64;
    }
    let values = nodes.map(|node| source.evaluate(node));

    let p = Polynomial::<4>::fit_least_squares(&nodes, &values).unwrap();
    for (found, expected) in p.coefficients().iter().zip(*source.coefficients()) {
        assert!((found - expected).abs() < 1e-10);
    }
}

#[test]
fn fit_least_squares_is_least_squares() {
    // One sample pulled off the line. The fit has to sit closer to the whole set than the line it
    // came from does.
    let source: Polynomial<3> = Polynomial::new([1.0, 2.0, -0.5]);
    let mut nodes = [0.0; 12];
    for (index, node) in nodes.iter_mut().enumerate() {
        *node = -2.0 + 0.5 * index as f64;
    }
    let mut values = nodes.map(|node| source.evaluate(node));
    values[4] += 3.0;

    let fitted = Polynomial::<3>::fit_least_squares(&nodes, &values).unwrap();
    let missed = |p: &Polynomial<3>| -> f64 {
        nodes
            .iter()
            .zip(values.iter())
            .map(|(node, value)| (p.evaluate(*node) - value).powi(2))
            .sum()
    };
    assert!(missed(&fitted) <= missed(&source));
}

#[test]
fn fit_least_squares_rejects_too_few_samples() {
    // Four coefficients cannot be pinned down by three samples.
    let nodes = [0.0, 1.0, 2.0];
    let values = [1.0, 2.0, 3.0];
    assert_eq!(
        Polynomial::<4>::fit_least_squares(&nodes, &values).err(),
        Some(PolynomialError::TooFewSamples)
    );
}

// ---- from values and derivatives at each end --------------------------------

#[test]
fn cubic_hermite_matches_its_endpoints() {
    let span = 2.5;
    let (start_value, start_slope) = (1.0, -0.5);
    let (end_value, end_slope) = (4.0, 2.0);

    let p = Polynomial::<4>::from_endpoint_derivatives(
        start_value,
        start_slope,
        end_value,
        end_slope,
        span,
    )
    .unwrap();

    assert!((p.evaluate(0.0) - start_value).abs() < 1e-12);
    assert!((p.evaluate(1.0) - end_value).abs() < 1e-12);
    // Slopes are against the outer parameter, so undo the piece's own clock.
    let [_, start_found] = p.evaluate_with_derivatives(0.0);
    let [_, end_found] = p.evaluate_with_derivatives(1.0);
    assert!((start_found / span - start_slope).abs() < 1e-12);
    assert!((end_found / span - end_slope).abs() < 1e-12);
}

#[test]
fn septic_hermite_matches_its_endpoints() {
    let span = 3.0;
    let start = [1.0, -0.5, 0.25, 0.125];
    let end = [4.0, 2.0, -1.0, 0.5];

    let p = Polynomial::<8>::from_endpoint_derivatives(&start, &end, span).unwrap();

    let at_start: [f64; 4] = p.evaluate_with_derivatives(0.0);
    let at_end: [f64; 4] = p.evaluate_with_derivatives(1.0);
    // Each derivative is one more division by the span than the one before it.
    let mut span_raised = 1.0;
    for order in 0..4 {
        assert!((at_start[order] / span_raised - start[order]).abs() < 1e-9);
        assert!((at_end[order] / span_raised - end[order]).abs() < 1e-9);
        span_raised *= span;
    }
}

#[test]
fn endpoint_construction_rejects_a_non_positive_span() {
    assert_eq!(
        Polynomial::<4>::from_endpoint_derivatives(0.0, 0.0, 1.0, 0.0, 0.0).err(),
        Some(PolynomialError::SpanNotPositive)
    );
    assert_eq!(
        Polynomial::<8>::from_endpoint_derivatives(&[0.0; 4], &[1.0, 0.0, 0.0, 0.0], -1.0).err(),
        Some(PolynomialError::SpanNotPositive)
    );
}
