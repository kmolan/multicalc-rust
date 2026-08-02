# Polynomials

A polynomial held as its coefficients, lowest power first: `coefficients[k]` multiplies `x^k`, so
`[1.0, -2.0, 3.0]` is `1 - 2x + 3x²`. That order holds everywhere in the module, root finders
included, so there is never a reversal to remember. The array is always `COEFFICIENT_COUNT` long,
which is one more than the highest power it can hold; anything above the degree is zero.

## Evaluating

`evaluate` works down from the top coefficient, multiplying by `x` and adding the next one.
`evaluate_with_derivatives` returns the value and as many derivatives as asked for from **one pass**,
not one pass per order — what a trajectory tracker calls to get position, velocity and acceleration
together. Orders above the degree come back zero, which is the right answer rather than a gap.

```rust
use multicalc::Polynomial;

let p: Polynomial<3> = Polynomial::new([1.0, -2.0, 3.0]);   // 1 - 2x + 3x²
assert!((p.evaluate(2.0) - 9.0).abs() < 1e-12);

let [value, slope, bend] = p.evaluate_with_derivatives(2.0);
assert!((value - 9.0).abs() < 1e-12);
assert!((slope - 10.0).abs() < 1e-12);
assert!((bend - 6.0).abs() < 1e-12);
```

## Arithmetic

`Add`, `Sub`, `Neg`, `Mul<T>`, `Div<T>` and `scale` all keep the same size. The ones that **grow**
cannot: a product runs to the sum of the two degrees, which neither input's size gives, so
`multiply_into`, `compose_into` and `divide` take the output size from the caller and report
`DegreeOverflow` rather than quietly dropping a term.

```rust
use multicalc::Polynomial;

let left: Polynomial<2> = Polynomial::new([1.0, 1.0]);      // 1 + x
let right: Polynomial<2> = Polynomial::new([2.0, -1.0]);    // 2 - x
let product = left.multiply_into::<2, 3>(&right).unwrap();  // 2 + x - x²
assert_eq!(product.coefficients(), &[2.0, 1.0, -1.0]);

// Two coefficients cannot hold the x² term, and say so.
assert!(left.multiply_into::<2, 2>(&right).is_err());
```

`divide` takes three sizes — the divisor's, the quotient's, and the remainder's — because neither
output's size follows from the inputs. `shift_argument`, `scale_argument` and `reverse` move the
curve along its variable without changing its size.

## Calculus

`derivative` returns the **same-sized** type with the highest coefficient zero; returning a smaller
one would need arithmetic in the type signature, which this crate does not do. `try_resize` shrinks
it afterwards when that matters, refusing if it would drop a coefficient that is not zero.
`definite_integral` adds up each term's contribution directly, so it needs no spare coefficient
however high the degree.

```rust
use multicalc::Polynomial;

let p: Polynomial<3> = Polynomial::new([0.0, 0.0, 3.0]);    // 3x²
assert_eq!(p.derivative().coefficients(), &[0.0, 6.0, 0.0]);
assert!((p.definite_integral(0.0, 2.0) - 8.0).abs() < 1e-12);
```

## Roots

Up to the fourth power there are exact formulas, so `real_roots` answers with no iteration and no
starting guess. Because those formulas are built from arithmetic and roots alone, differentiating
through them with `Dual` gives an exact derivative of a root against whatever built the
coefficients.

Past the fourth power no formula exists. `count_real_roots` says how many real roots a range holds
by comparing sign changes at its two ends, with no iteration at all, and `real_roots_in` separates
them and closes in by halving within a step budget. Two differences worth knowing: counting reports
**distinct** roots, so a doubled root counts once where the exact formulas list it twice; and a root
found by halving is a number with no meaningful derivative, unlike the exact ones.

```rust
use multicalc::Polynomial;

// Exact: (x - 1)(x - 2)(x - 3)
let cubic: Polynomial<4> = Polynomial::new([-6.0, 11.0, -6.0, 1.0]);
assert_eq!(cubic.real_roots().unwrap().len(), 3);

// Past where a formula reaches: (x + 3)(x + 1)(x - 0.5)(x - 2)(x - 4)(x - 7)
let degree6: Polynomial<7> = Polynomial::new([84.0, -131.0, -126.5, 104.5, 5.5, -9.5, 1.0]);
let bound = degree6.cauchy_root_bound().unwrap();   // every root lies inside ±bound
assert_eq!(degree6.count_real_roots(-bound, bound).unwrap(), 6);
let roots = degree6.real_roots_in(-bound, bound, 1e-10, 400).unwrap();
assert!((roots.as_slice()[0] + 3.0).abs() < 1e-8);
```

## Building one from data

`from_roots` multiplies out a set of roots. `from_points` finds the one polynomial through exactly
as many points as it has coefficients. `fit_least_squares` finds the closest one to more points than
it can pass through. `from_jet` takes the series a `Jet` already carries. `chebyshev_nodes` gives
sample positions that bunch toward the ends of a range, which keeps a fit from swinging there.

Both fitting routines shift and stretch the sample positions into the range -1 to 1 internally,
which is what keeps them well behaved to about the eighth power. One caveat worth stating: putting
the answer back into powers of the caller's variable gives up digits when the samples sit far from
zero, whatever the fit did.

```rust
use multicalc::Polynomial;

let p = Polynomial::<4>::from_roots(&[1.0, 2.0, 3.0]).unwrap();
assert_eq!(p.coefficients(), &[-6.0, 11.0, -6.0, 1.0]);

// Five points off 1 + 2x, fitted with a straight line.
let fitted = Polynomial::<2>::fit_least_squares(
    &[0.0, 1.0, 2.0, 3.0, 4.0],
    &[1.0, 3.0, 5.0, 7.0, 9.0],
).unwrap();
assert!((fitted.evaluate(10.0) - 21.0).abs() < 1e-10);
```

`Polynomial::<4>::from_endpoint_derivatives` builds the cubic matching a value and a slope at each
end — what every smooth-step and spline piece needs. `Polynomial::<8>::from_endpoint_derivatives`
takes a value and three derivatives at each end, which is what a minimum-snap segment is. Both run on
the piece's own clock, from 0 at the start to 1 at the end, with `span` converting the caller's
derivatives against the outer parameter.

## Curves made of pieces

`PiecewisePolynomial<MAX_PIECES, COEFFICIENTS_PER_PIECE, DIMENSION, T>` lays polynomial pieces end to
end. Each runs on its own 0-to-1 clock and carries a **span** saying how much of the shared parameter
it covers; keeping every piece on the same clock is what stops the numbers in a high-degree solve from
spanning many powers of the piece width. Evaluation is fixed work with no allocation, so it is safe in
a tight loop, and a parameter before the start or past the end clamps to that end.

```rust
use multicalc::{PiecewisePolynomial, Polynomial};

// One axis: climbing to 1 over two units of the parameter, then on to 3 over one more.
let first = [Polynomial::<2>::new([0.0, 1.0])];
let second = [Polynomial::<2>::new([1.0, 2.0])];
let curve = PiecewisePolynomial::<2, 2, 1>::try_from_pieces(&[first, second], &[2.0, 1.0]).unwrap();

assert!((curve.total_span() - 3.0).abs() < 1e-12);
let [handover] = curve.evaluate(2.0).unwrap().into_array();
assert!((handover - 1.0).abs() < 1e-12);
```

`derivative` and `nth_derivative` give the slope as a curve in its own right, and `definite_integral`
the area under it. Evaluation clamps outside the range; integration instead trims the range to the
curve, so reaching past either end adds nothing.

## Several variables

`MultivariatePolynomial<VARIABLES, MAX_TERMS, T>` holds a list of terms, each carrying its own power
for every variable. There is no ordering to remember and no index to work out, and storage grows with
how many terms there are rather than with the degree — a three-variable, eight-term polynomial is
under 200 bytes against 1728 for every coefficient up to the fifth power. Only whole-number powers
are held, and that is what keeps every operation closed: the sum, product, partial derivative,
partial antiderivative and substitution of one of these is another one.

```rust
use multicalc::{MultivariatePolynomial, MultivariateTerm};

// 3x²y + 2xy - 1
let p = MultivariatePolynomial::<2, 3>::try_from_terms(&[
    MultivariateTerm::new(3.0, [2, 1]),
    MultivariateTerm::new(2.0, [1, 1]),
    MultivariateTerm::new(-1.0, [0, 0]),
]).unwrap();

assert!((p.evaluate(&[1.5, -2.0]) + 20.5).abs() < 1e-12);

// The slope in every variable at once: 6xy + 2y, and 3x² + 2x.
let gradient = p.gradient_at(&[1.5, -2.0]);
assert!((gradient[0] + 22.0).abs() < 1e-12);
assert!((gradient[1] - 9.75).abs() < 1e-12);
```

`partial_derivative` gives the symbolic answer as another polynomial, and because the type is generic
over the scalar, evaluating at `Dual` inputs differentiates through the evaluation instead — two
unrelated routes to the same slope. `substitute` fixes one variable at a value; note it does **not**
reduce the variable count, so the fixed variable stays in the type with a power of zero. Crossing to
the dense `Polynomial` needs a polynomial that already names one variable, via `to_univariate`.

Two literal macros build these more briefly: `polynomial![1.0, -2.0, 3.0]` and
`multivariate_polynomial![(2.5, [2, 3]), (-1.0, [0, 1])]`.

Errors: every fallible call returns [`PolynomialError`](error-handling.md). Demo:
[polynomials.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/polynomials.rs).


---

[Back to the tutorial index](README.md)
