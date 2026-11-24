# Polynomials

`Polynomial<COEFFICIENT_COUNT, T>` holds its coefficients lowest power first: `coefficients[k]`
multiplies `x^k`, so `[1.0, -2.0, 3.0]` is `1 - 2x + 3x²`. The whole module, root finders included,
uses this order. The array length is `COEFFICIENT_COUNT`, one more than the highest power it can
hold. The coefficients above the degree are zero.

## Evaluation

`evaluate` uses Horner's method. `evaluate_with_derivatives` returns the value and as many
derivatives as the output array holds, in **one pass**, for example position, velocity and
acceleration of a trajectory. Orders above the degree are zero.

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

`Add`, `Sub`, `Neg`, `Mul<T>`, `Div<T>` and `scale` keep the same size. A product has the sum of
the two degrees, which the input sizes do not give. Thus `multiply_into`, `compose_into` and
`divide` take the output size from the caller. If a term does not fit, they return
`DegreeOverflow` and do not drop the term.

```rust
use multicalc::Polynomial;

let left: Polynomial<2> = Polynomial::new([1.0, 1.0]);      // 1 + x
let right: Polynomial<2> = Polynomial::new([2.0, -1.0]);    // 2 - x
let product = left.multiply_into::<2, 3>(&right).unwrap();  // 2 + x - x²
assert_eq!(product.coefficients(), &[2.0, 1.0, -1.0]);

// Two coefficients cannot hold the x² term.
assert!(left.multiply_into::<2, 2>(&right).is_err());
```

`divide` takes three sizes: the divisor, the quotient and the remainder. `shift_argument`,
`scale_argument` and `reverse` transform the variable and keep the size.

## Calculus

`derivative` returns the **same-sized** type, with the highest coefficient zero. A smaller return
type would need const-generic arithmetic, which the crate does not use. `try_resize` shrinks the
result, and returns `None` if a dropped coefficient is not zero. `definite_integral` sums each
term's antiderivative directly, so it needs no spare coefficient.

```rust
use multicalc::Polynomial;

let p: Polynomial<3> = Polynomial::new([0.0, 0.0, 3.0]);    // 3x²
assert_eq!(p.derivative().coefficients(), &[0.0, 6.0, 0.0]);
assert!((p.definite_integral(0.0, 2.0) - 8.0).abs() < 1e-12);
```

## Roots

To degree 4, `real_roots` uses the closed-form formulas, with no initial guess. The quartic polishes
its resolvent root with at most two Newton steps, so a small odd term keeps full accuracy.
The formulas use only arithmetic and roots, so a `Dual` through them gives the exact derivative of
a root with respect to the coefficients.

Above degree 4:

- `count_real_roots` counts the real roots in a range from the Sturm sequence, with no iteration.
- `real_roots_in` isolates the roots in `(lower, upper]` with the Sturm sequence and refines each
  one by bisection. If the bisection budget runs out, it returns `DidNotConverge`.
- `cauchy_root_bound` gives a bound `B` such that each root lies in `[-B, B]`.

The Sturm count gives **distinct** roots, so a double root counts once. The closed-form formulas
list a double root twice. A root from bisection carries no derivative information, unlike a
closed-form root.

```rust
use multicalc::Polynomial;

// Closed form: (x - 1)(x - 2)(x - 3)
let cubic: Polynomial<4> = Polynomial::new([-6.0, 11.0, -6.0, 1.0]);
assert_eq!(cubic.real_roots().unwrap().len(), 3);

// Degree 6: (x + 3)(x + 1)(x - 0.5)(x - 2)(x - 4)(x - 7)
let degree6: Polynomial<7> = Polynomial::new([84.0, -131.0, -126.5, 104.5, 5.5, -9.5, 1.0]);
let bound = degree6.cauchy_root_bound().unwrap();   // each root lies in [-bound, bound]
assert_eq!(degree6.count_real_roots(-bound, bound).unwrap(), 6);
let roots = degree6.real_roots_in(-bound, bound, 1e-10, 400).unwrap();
assert!((roots.as_slice()[0] + 3.0).abs() < 1e-8);
```

## Construction from data

- `from_roots`: the product `∏ (x − rᵢ)`.
- `from_points`: the interpolating polynomial through exactly `COEFFICIENT_COUNT` points.
- `fit_least_squares`: the least-squares fit to more points.
- `from_jet`: the Taylor series that a `Jet` carries.
- `chebyshev_nodes`: Chebyshev sample positions on a range, which suppress Runge oscillation at the
  ends of a fit.

Both fit routines map the samples to `[-1, 1]` internally, which keeps them well-conditioned to
about degree 8. Caution: the conversion back to powers of the caller's variable loses digits when
the samples are far from zero.

```rust
use multicalc::Polynomial;

let p = Polynomial::<4>::from_roots(&[1.0, 2.0, 3.0]).unwrap();
assert_eq!(p.coefficients(), &[-6.0, 11.0, -6.0, 1.0]);

// Five points on 1 + 2x, fitted with a straight line.
let fitted = Polynomial::<2>::fit_least_squares(
    &[0.0, 1.0, 2.0, 3.0, 4.0],
    &[1.0, 3.0, 5.0, 7.0, 9.0],
).unwrap();
assert!((fitted.evaluate(10.0) - 21.0).abs() < 1e-10);
```

`Polynomial::<4>::from_endpoint_derivatives` builds the cubic Hermite piece that matches a value
and a slope at each end. `Polynomial::<8>::from_endpoint_derivatives` matches a value and three
derivatives at each end, which is a minimum-snap segment. Both run on the local parameter `[0, 1]`.
The `span` argument is the width of the piece in the outer parameter, and scales the given
derivatives.

## Piecewise polynomials

`PiecewisePolynomial<MAX_PIECES, COEFFICIENTS_PER_PIECE, DIMENSION, T>` joins polynomial pieces end
to end. Each piece runs on its own local parameter `[0, 1]` and has a **span**: its width in the
shared parameter. The local parameter keeps the coefficients of a high-degree solve in a narrow
range. Evaluation is fixed work with no allocation. A parameter outside the range clamps to the
nearest end.

```rust
use multicalc::{PiecewisePolynomial, Polynomial};

// One axis: up to 1 over two units of the parameter, then up to 3 over one more.
let first = [Polynomial::<2>::new([0.0, 1.0])];
let second = [Polynomial::<2>::new([1.0, 2.0])];
let curve = PiecewisePolynomial::<2, 2, 1>::try_from_pieces(&[first, second], &[2.0, 1.0]).unwrap();

assert!((curve.total_span() - 3.0).abs() < 1e-12);
let [handover] = curve.evaluate(2.0).unwrap().into_array();
assert!((handover - 1.0).abs() < 1e-12);
```

`derivative` and `nth_derivative` return the derivative as a new piecewise polynomial.
`definite_integral` returns the integral. Evaluation clamps outside the range, but integration
clips the range to the curve, so a range past either end adds nothing.

## Multivariate polynomials

`MultivariatePolynomial<VARIABLES, MAX_TERMS, T>` holds a list of terms. Each term has a
coefficient and one integer exponent for each variable. Storage grows with the number of terms,
not with the degree. For example, three variables and eight terms take 200 bytes, against 1728
bytes for a dense array of each coefficient to the fifth power. Exponents are non-negative
integers, so the set is closed under sum, product, partial derivative, partial antiderivative and
substitution.

```rust
use multicalc::{MultivariatePolynomial, MultivariateTerm};

// 3x²y + 2xy - 1
let p = MultivariatePolynomial::<2, 3>::try_from_terms(&[
    MultivariateTerm::new(3.0, [2, 1]),
    MultivariateTerm::new(2.0, [1, 1]),
    MultivariateTerm::new(-1.0, [0, 0]),
]).unwrap();

assert!((p.evaluate(&[1.5, -2.0]) + 20.5).abs() < 1e-12);

// ∇p = (6xy + 2y, 3x² + 2x)
let gradient = p.gradient_at(&[1.5, -2.0]);
assert!((gradient[0] + 22.0).abs() < 1e-12);
assert!((gradient[1] - 9.75).abs() < 1e-12);
```

`partial_derivative` returns the symbolic partial as another polynomial. The type is generic over
the scalar, so evaluation at `Dual` inputs also gives the slope, by autodiff. `substitute` fixes
one variable at a value. It does **not** decrease the variable count: the fixed variable stays in
the type with exponent zero. `to_univariate` converts a polynomial in one variable to the dense
`Polynomial`.

Two literal macros: `polynomial![1.0, -2.0, 3.0]` and
`multivariate_polynomial![(2.5, [2, 3]), (-1.0, [0, 1])]`.

Errors: each fallible call returns [`PolynomialError`](error-handling.md). Demo:
[polynomials.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/polynomials.rs).

---

[Back to the tutorial index](README.md)
