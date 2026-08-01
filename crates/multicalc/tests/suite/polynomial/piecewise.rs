//! Piecewise polynomial tests. The rest of the surface lands with the wider piecewise suite; these
//! cover the slope curves.

use multicalc::polynomial::{PiecewisePolynomial, Polynomial};

/// A two-piece curve in two dimensions, with pieces of different spans so a wrong conversion between
/// a piece's own clock and the shared parameter cannot pass unnoticed.
fn two_piece_curve() -> PiecewisePolynomial<2, 4, 2, f64> {
    let first = [
        Polynomial::<4>::new([0.0, 1.0, -0.5, 0.25]),
        Polynomial::<4>::new([1.0, 0.0, 2.0, -1.0]),
    ];
    let second = [
        Polynomial::<4>::new([0.75, 2.0, 0.5, 0.125]),
        Polynomial::<4>::new([2.0, -1.5, 0.25, 0.5]),
    ];
    PiecewisePolynomial::try_from_pieces(&[first, second], &[2.0, 0.5]).unwrap()
}

#[test]
fn derivative_curve_matches_evaluate_with_derivatives() {
    let curve = two_piece_curve();

    // One parameter inside each piece, and one past the end where both routes clamp.
    for parameter in [0.4, 1.9, 2.1, 2.4, 7.0] {
        let [_, slope, bend] = curve.evaluate_with_derivatives(parameter).unwrap();

        let from_curve = curve.derivative().evaluate(parameter).unwrap();
        let from_second = curve.nth_derivative(2).evaluate(parameter).unwrap();
        for axis in 0..2 {
            assert!((from_curve[axis] - slope[axis]).abs() < 1e-12);
            assert!((from_second[axis] - bend[axis]).abs() < 1e-12);
        }
    }
}

#[test]
fn derivative_matches_a_finite_difference() {
    let curve = two_piece_curve();
    let step = 1e-6;

    // Well inside a piece, so the difference does not straddle a join.
    for parameter in [0.9, 2.3] {
        let ahead = curve.evaluate(parameter + step).unwrap();
        let behind = curve.evaluate(parameter - step).unwrap();
        let slope = curve.derivative().evaluate(parameter).unwrap();
        for axis in 0..2 {
            let approximate = (ahead[axis] - behind[axis]) / (2.0 * step);
            assert!((slope[axis] - approximate).abs() < 1e-6);
        }
    }
}

#[test]
fn definite_integral_matches_the_single_polynomial_routine() {
    // One piece spanning exactly one unit is the same thing as the polynomial on its own.
    let polynomial = Polynomial::<4>::new([1.5, -2.0, 0.75, 0.25]);
    let curve =
        PiecewisePolynomial::<1, 4, 1, f64>::try_from_pieces(&[[polynomial]], &[1.0]).unwrap();

    for (lower, upper) in [(0.0, 1.0), (0.25, 0.9), (0.0, 0.5)] {
        let [found] = curve.definite_integral(lower, upper).unwrap().into_array();
        assert!((found - polynomial.definite_integral(lower, upper)).abs() < 1e-12);
    }
}

#[test]
fn definite_integral_adds_up_across_pieces() {
    let curve = two_piece_curve();
    let total = curve.total_span();

    // Splitting the range anywhere covers the same area, including across a join.
    let whole = curve.definite_integral(0.0, total).unwrap();
    for split in [0.5, 2.0, 2.25] {
        let first = curve.definite_integral(0.0, split).unwrap();
        let second = curve.definite_integral(split, total).unwrap();
        for axis in 0..2 {
            assert!((first[axis] + second[axis] - whole[axis]).abs() < 1e-12);
        }
    }

    // Bounds the wrong way round negate.
    let backward = curve.definite_integral(total, 0.0).unwrap();
    for axis in 0..2 {
        assert!((backward[axis] + whole[axis]).abs() < 1e-12);
    }
}

#[test]
fn definite_integral_trims_to_the_curve() {
    let curve = two_piece_curve();
    let total = curve.total_span();
    let whole = curve.definite_integral(0.0, total).unwrap();

    // Reaching well past both ends adds nothing, unlike evaluation, which holds the end values.
    let overreaching = curve.definite_integral(-50.0, 500.0).unwrap();
    for axis in 0..2 {
        assert!((overreaching[axis] - whole[axis]).abs() < 1e-12);
    }

    // A range entirely outside the curve covers nothing at all.
    let outside = curve.definite_integral(10.0, 20.0).unwrap();
    assert!(outside[0].abs() < 1e-12 && outside[1].abs() < 1e-12);
}

#[test]
fn nth_derivative_keeps_the_pieces_and_runs_out_to_zero() {
    let curve = two_piece_curve();

    let unchanged = curve.nth_derivative(0);
    assert_eq!(unchanged, curve);

    let slope = curve.derivative();
    assert_eq!(slope.piece_count(), curve.piece_count());
    assert!((slope.total_span() - curve.total_span()).abs() < 1e-12);

    // Four coefficients per piece, so the fourth slope has nothing left.
    let flat = curve.nth_derivative(4).evaluate(1.0).unwrap();
    assert!(flat[0].abs() < 1e-12 && flat[1].abs() < 1e-12);
}
