//! Polynomials: evaluating one, finding its roots, building one from data, and several variables.
//!
//! Run with: `cargo run -p multicalc-demos --example polynomials`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::polynomial::{MultivariatePolynomial, MultivariateTerm, Polynomial};
use multicalc::scalar::Dual;

fn main() {
    evaluating();
    roots();
    building_from_data();
    several_variables();
}

/// One pass gives the value and as many derivatives as asked for.
fn evaluating() {
    println!("== Evaluating ==");

    // 1 + 2x + 3x² + 4x³ + 5x⁴ + 6x⁵ + 7x⁶ + 8x⁷, read at x = 1/2 where every power is exact.
    let p: Polynomial<8> = Polynomial::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let [value, slope, bend] = p.evaluate_with_derivatives(0.5);

    // Worked out by hand: 1 + 1 + 3/4 + 1/2 + 5/16 + 3/16 + 7/64 + 1/16, and likewise for the rest.
    let (want_value, want_slope, want_bend) = (3.921875, 14.5625, 71.625);
    println!(
        "  value at x = 0.5   {value:>12.6}   |err| = {:.1e}",
        (value - want_value).abs()
    );
    println!(
        "  slope              {slope:>12.6}   |err| = {:.1e}",
        (slope - want_slope).abs()
    );
    println!(
        "  bend               {bend:>12.6}   |err| = {:.1e}",
        (bend - want_bend).abs()
    );
    println!("  (one pass over eight coefficients, not one pass per derivative)");

    assert!((value - want_value).abs() < 1e-12);
    assert!((slope - want_slope).abs() < 1e-12);
    assert!((bend - want_bend).abs() < 1e-12);
    println!();
}

/// Up to the fourth power there is a formula; past that, count and close in.
fn roots() {
    println!("== Roots ==");

    // (x - 1)(x - 2)(x - 3), by the exact cubic formula: no iteration, no starting guess.
    let cubic: Polynomial<4> = Polynomial::new([-6.0, 11.0, -6.0, 1.0]);
    let found = cubic.real_roots().unwrap();
    print!("  (x-1)(x-2)(x-3) exactly ->");
    for (root, expected) in found.as_slice().iter().zip([1.0, 2.0, 3.0]) {
        print!("  {root:.9} (|err| {:.1e})", (root - expected).abs());
        assert!((root - expected).abs() < 1e-8);
    }
    println!();

    // (x+3)(x+1)(x-0.5)(x-2)(x-4)(x-7): too high for any formula, so the roots are separated by
    // counting sign changes and then closed in on by halving.
    let degree6: Polynomial<7> = Polynomial::new([84.0, -131.0, -126.5, 104.5, 5.5, -9.5, 1.0]);
    let bound = degree6.cauchy_root_bound().unwrap();
    println!(
        "  every root of the sixth-power one lies inside [{:.1}, {:.1}]",
        -bound, bound
    );

    let counted = degree6.count_real_roots(-bound, bound).unwrap();
    println!("  it holds {counted} real roots, known before any of them is located");

    let located = degree6.real_roots_in(-bound, bound, 1e-12, 400).unwrap();
    print!("  found ->");
    for (root, expected) in located
        .as_slice()
        .iter()
        .zip([-3.0, -1.0, 0.5, 2.0, 4.0, 7.0])
    {
        print!("  {root:.6}");
        assert!((root - expected).abs() < 1e-8);
    }
    println!();
    assert_eq!(counted, 6);
    println!();
}

/// Interpolating through points, and fitting through more points than fit.
fn building_from_data() {
    println!("== Building from data ==");

    // Sampling at points that bunch toward the ends keeps the fit from swinging there.
    let nodes = Polynomial::<9>::chebyshev_nodes(-1.0, 1.0);
    let values = nodes.map(f64::sin);
    let interpolated = Polynomial::<9>::from_points(&nodes, &values).unwrap();

    let mut worst = 0.0_f64;
    for step in 0..=100 {
        let x = -1.0 + 2.0 * step as f64 / 100.0;
        worst = worst.max((interpolated.evaluate(x) - x.sin()).abs());
    }
    println!("  sin through 9 points, largest miss over [-1, 1]   {worst:.2e}");
    println!("  (that is what a curve of this degree can do for sin, not a shortfall in the fit)");
    assert!(worst < 1e-7);

    // Twenty samples off a cubic, with noise: no cubic passes through them all, so this finds the
    // one that misses by the least overall.
    let source: Polynomial<4> = Polynomial::new([1.5, -2.0, 0.75, 0.25]);
    let mut sample_nodes = [0.0; 20];
    let mut sample_values = [0.0; 20];
    for (index, (node, value)) in sample_nodes
        .iter_mut()
        .zip(sample_values.iter_mut())
        .enumerate()
    {
        *node = -3.0 + 0.4 * index as f64;
        // A fixed wobble rather than a random one, so the demo prints the same figure every run.
        *value = source.evaluate(*node) + 0.05 * (index as f64 * 1.7).sin();
    }
    let fitted = Polynomial::<4>::fit_least_squares(&sample_nodes, &sample_values).unwrap();

    let mut squared_miss = 0.0;
    for (node, value) in sample_nodes.iter().zip(sample_values.iter()) {
        squared_miss += (fitted.evaluate(*node) - value).powi(2);
    }
    println!("  cubic fitted to 20 noisy samples, squared miss     {squared_miss:.2e}");
    print!("  coefficients ->");
    for (found, expected) in fitted.coefficients().iter().zip(*source.coefficients()) {
        print!("  {found:.4} (true {expected:.4})");
    }
    println!();
    assert!(squared_miss < 1.0);
    println!();
}

/// Symbolic partial derivatives, cross-checked against differentiating through the evaluation.
fn several_variables() {
    println!("== Several variables ==");

    // 3x²y + 2xy - 1, held as three terms rather than a grid of coefficients.
    let p = MultivariatePolynomial::<2, 3>::try_from_terms(&[
        MultivariateTerm::new(3.0, [2, 1]),
        MultivariateTerm::new(2.0, [1, 1]),
        MultivariateTerm::new(-1.0, [0, 0]),
    ])
    .unwrap();

    let point = [1.5, -2.0];
    let value = p.evaluate(&point);
    let gradient = p.gradient_at(&point);
    println!("  3x²y + 2xy - 1 at (1.5, -2)      {value:>10.6}");
    println!("  slope in x (symbolically)        {:>10.6}", gradient[0]);
    println!("  slope in y (symbolically)        {:>10.6}", gradient[1]);

    // The same slopes again, this time by carrying dual numbers through the evaluation. Two
    // unrelated routes to the same answer.
    let dual = MultivariatePolynomial::<2, 3, Dual<f64>>::try_from_terms(&[
        MultivariateTerm::new(Dual::constant(3.0), [2, 1]),
        MultivariateTerm::new(Dual::constant(2.0), [1, 1]),
        MultivariateTerm::new(Dual::constant(-1.0), [0, 0]),
    ])
    .unwrap();
    let in_x = dual.evaluate(&[Dual::variable(point[0]), Dual::constant(point[1])]);
    let in_y = dual.evaluate(&[Dual::constant(point[0]), Dual::variable(point[1])]);
    println!(
        "  slope in x (through the values)  {:>10.6}   |err| = {:.1e}",
        in_x.deriv,
        (in_x.deriv - gradient[0]).abs()
    );
    println!(
        "  slope in y (through the values)  {:>10.6}   |err| = {:.1e}",
        in_y.deriv,
        (in_y.deriv - gradient[1]).abs()
    );

    assert!((value + 20.5).abs() < 1e-12);
    assert!((in_x.deriv - gradient[0]).abs() < 1e-12);
    assert!((in_y.deriv - gradient[1]).abs() < 1e-12);
    println!();
}
