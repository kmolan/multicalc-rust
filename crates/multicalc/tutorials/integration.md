# Integration

Definite integrals, single or repeated, by iterative Newton-Cotes rules and Gaussian quadrature,
over finite, semi-infinite and infinite limits.

- `integral`: a one-call function for the common case. See
  [One-call functions and strategy objects](importing.md#one-call-functions-and-strategy-objects).
- `IterativeSingle`: Boole (default), Simpson and Trapezoidal rules. `from_parameters` sets the rule
  and the interval count, rounded up to a multiple that the rule needs.
- The iterative rules use pairwise summation by default. Chain `.with_kahan_summation()` for Kahan
  summation.
- `GaussianSingle`: Gauss-Legendre (default), Gauss-Hermite and Gauss-Laguerre.
- Both implement `IntegratorSingleVariable` (`integrate`, `single_integral`, `double_integral`)
  and `IntegratorMultiVariable` (`integrate`, `single_partial_integral`,
  `double_partial_integral`) for the `*Multi` forms.

Iterative rules over finite and infinite limits, and the rule and interval count set by
`from_parameters`:

```rust
use multicalc::IntegratorSingleVariable;
use multicalc::{IterativeMethod, IterativeSingle};

let integrator = IterativeSingle::default();     // Boole's rule, 120 intervals

let line = |x: f64| 2.0 * x;
let limits = [0.0, 2.0];
let area = integrator.single_integral(&line, &limits).unwrap();   // 4.0

// infinite and semi-infinite limits, for integrands that decay
let bell_curve = |x: f64| (-x * x).exp();
let real_line = [f64::NEG_INFINITY, f64::INFINITY];
let bell = integrator.single_integral(&bell_curve, &real_line).unwrap();   // sqrt(pi)

// Simpson's rule, 120 intervals
let interval_count = 120;
let simpson: IterativeSingle =
    IterativeSingle::from_parameters(interval_count, IterativeMethod::Simpsons);
```

Each Gaussian rule integrates over a fixed domain. Pass the bare integrand `f(x)`, because the
weights already include the weight function:

```rust
use multicalc::IntegratorSingleVariable;
use multicalc::GaussianSingle;
use multicalc::GaussianQuadratureMethod;

// Gauss-Hermite integrates f(x) * e^(-x^2) over the whole real line.
let node_count = 5;
let hermite = GaussianSingle::from_parameters(node_count, GaussianQuadratureMethod::GaussHermite);

let square = |x: f64| x * x;
let real_line = [f64::NEG_INFINITY, f64::INFINITY];
let val = hermite.single_integral(&square, &real_line).unwrap();   // sqrt(pi)/2
```

| Rule           | Computes                                              |
| -------------- | ---------------------------------------------------- |
| Gauss-Legendre | $\int_a^b f(x)\, \mathrm{d}x$                         |
| Gauss-Laguerre | $\int_0^\infty f(x)\, e^{-x}\, \mathrm{d}x$           |
| Gauss-Hermite  | $\int_{-\infty}^\infty f(x)\, e^{-x^2}\, \mathrm{d}x$ |

Gaussian nodes and weights come from the [quadrature tables](gaussian-quadrature-tables.md).

Errors: integration calls return [`IntegrateError`](error-handling.md): `IterationsZero`,
`LimitsIllDefined`, `QuadratureOrderOutOfRange`, `NonFinite`, or `IndexOutOfRange` (partial
integrals). Full demos:
[iterative_integration.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/iterative_integration.rs)
and
[gaussian_integration.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/gaussian_integration.rs).


---

[Back to the tutorial index](README.md)
