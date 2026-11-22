# Integration

Definite integration of any order: iterative Newton-Cotes rules and Gaussian quadrature, over
finite, semi-infinite, and infinite limits.

- `integral`: a one-call function covering the common case. See
  [the note on the two paths](importing.md#the-easy-path-and-the-configurable-one).
- `IterativeSingle`: Boole (default), Simpson, and Trapezoidal rules; pick the rule and interval
  count with `from_parameters`.
- Pairwise summation is the default; chain `.with_kahan_summation()` to opt into Kahan.
- `GaussianSingle`: Gauss-Legendre, Gauss-Hermite, and Gauss-Laguerre. Pass the **bare**
  integrand; the weights already carry the weighting factor.
- Both implement the `IntegratorSingleVariable` / `IntegratorMultiVariable` traits
  (`integrate`, `single_integral`, `double_integral`, …).

Iterative rules over finite and infinite limits:

```rust
use multicalc::IntegratorSingleVariable;
use multicalc::IterativeSingle;

let integrator = IterativeSingle::default();     // Boole's rule, 120 intervals

let line = |x: f64| 2.0 * x;
let limits = [0.0, 2.0];
let area = integrator.single_integral(&line, &limits).unwrap();   // 4.0

// infinite / semi-infinite limits are supported for decaying integrands
let bell_curve = |x: f64| (-x * x).exp();
let real_line = [f64::NEG_INFINITY, f64::INFINITY];
let bell = integrator.single_integral(&bell_curve, &real_line).unwrap();   // sqrt(pi)
```

Choose the rule and interval count with `from_parameters`:

```rust
use multicalc::{IterativeMethod, IterativeSingle};

let interval_count = 120;
let integrator: IterativeSingle =
    IterativeSingle::from_parameters(interval_count, IterativeMethod::Simpsons);
```

Each Gaussian rule integrates over a fixed domain. Pass the bare integrand `f(x)`; the weights
already carry the weighting factor:

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
`LimitsIllDefined`, `QuadratureOrderOutOfRange`, or `NonFinite`. Full demos:
[iterative_integration.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/iterative_integration.rs)
and
[gaussian_integration.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/gaussian_integration.rs).


---

[Back to the tutorial index](README.md)
