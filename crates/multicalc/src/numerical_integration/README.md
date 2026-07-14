# numerical_integration

Definite integration of any order: iterative Newton-Cotes rules and Gaussian quadrature, over finite,
semi-infinite, and infinite limits.

- [`iterative_integration::IterativeSingle`](iterative_integration.rs) — Boole (default), Simpson, and
  Trapezoidal rules; pick the rule and interval count with `from_parameters`.
- Pairwise summation is the default; chain `.with_kahan_summation()` to opt into Kahan.
- [`gaussian_integration::GaussianSingle`](gaussian_integration.rs) — Gauss-Legendre, Gauss-Hermite,
  and Gauss-Laguerre. Pass the **bare** integrand; the weights already carry the weighting factor.
- Both implement the [`integrator`](integrator.rs)`::IntegratorSingleVariable` / `MultiVariable`
  traits (`get_single`, `get_double`, ...); the rules live in [`mode`](mode.rs).

```rust
use multicalc::numerical_integration::integrator::IntegratorSingleVariable;
use multicalc::numerical_integration::iterative_integration::IterativeSingle;

let integrator = IterativeSingle::default();                 // Boole's rule, 120 intervals
let area = integrator.get_single(&|x: f64| 2.0 * x, &[0.0, 2.0]).unwrap();   // 4.0

// infinite / semi-infinite limits are supported for decaying integrands
let bell = integrator
    .get_single(&|x| (-x * x).exp(), &[f64::NEG_INFINITY, f64::INFINITY])
    .unwrap();                                               // sqrt(pi)
```

Gaussian nodes and weights come from [`gaussian_tables`](../gaussian_tables).
