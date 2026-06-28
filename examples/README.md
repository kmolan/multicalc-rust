# Examples

Runnable, self-contained examples for each module of `multicalc`. Every file has a `main`
that prints its results against the known analytic value (with the `|err|`), so you can see
the API in action and check accuracy at the same time:

```sh
cargo run --example <name>
```

The three numerical examples also reproduce the accuracy figures in [`BENCHMARKS.md`](../BENCHMARKS.md)
sections 1–4 (noted below), so the published tables stay honest.

| Example | Module(s) | What it shows |
| --- | --- | --- |
| [`differentiation`](differentiation.rs) | `numerical_derivative` | Single- and multi-variable finite-difference derivatives (orders 1–3, partials, mixed partials). Reproduces BENCHMARKS.md §1–2. |
| [`jacobian_hessian`](jacobian_hessian.rs) | `numerical_derivative::{jacobian, hessian}` | Jacobian of a vector of functions and the Hessian of a scalar function. |
| [`iterative_integration`](iterative_integration.rs) | `numerical_integration::iterative_integration` | Boole / Simpson / Trapezoidal rules, multi-variable partial integrals, and infinite / semi-infinite limits. Reproduces BENCHMARKS.md §3. |
| [`gaussian_integration`](gaussian_integration.rs) | `numerical_integration::gaussian_integration` | Gauss-Legendre (finite), Gauss-Hermite and Gauss-Laguerre (infinite), with the bare-integrand convention. Reproduces BENCHMARKS.md §4. |
| [`vector_field`](vector_field.rs) | `vector_field` | Curl, divergence, line integrals and flux integrals. |
| [`approximation`](approximation.rs) | `approximation` | Linear and quadratic Taylor approximations, `predict`, and goodness-of-fit metrics. |