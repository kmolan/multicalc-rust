# Examples

Runnable, self-contained examples for each module of `multicalc`. Every file has a `main`
that prints its results against the known analytic value (with the `|err|`), so you can see
the API in action and check accuracy at the same time:

```sh
cargo run --example <name>
```

Several examples also reproduce the published accuracy figures in [`benches/`](../benches)
(noted below), so the tables stay honest.

| Example | Module(s) | What it shows |
| --- | --- | --- |
| [`differentiation`](differentiation.rs) | `numerical_derivative` | Single- and multi-variable finite-difference derivatives (orders 1-3, partials, mixed partials). Reproduces the differentiation accuracy tables in benches/calculus.md. |
| [`jacobian_hessian`](jacobian_hessian.rs) | `numerical_derivative::{jacobian, hessian}` | Jacobian of a vector of functions and the Hessian of a scalar function. |
| [`iterative_integration`](iterative_integration.rs) | `numerical_integration::iterative_integration` | Boole / Simpson / Trapezoidal rules, multi-variable partial integrals, and infinite / semi-infinite limits. Reproduces the iterative-integration accuracy table in benches/calculus.md. |
| [`gaussian_integration`](gaussian_integration.rs) | `numerical_integration::gaussian_integration` | Gauss-Legendre (finite), Gauss-Hermite and Gauss-Laguerre (infinite), with the bare-integrand convention. Reproduces the Gaussian-quadrature accuracy table in benches/calculus.md. |
| [`vector_field`](vector_field.rs) | `vector_field` | Curl, divergence, line integrals and flux integrals. |
| [`approximation`](approximation.rs) | `approximation` | Linear and quadratic Taylor approximations, `predict`, and goodness-of-fit metrics. |
| [`linear_algebra`](linear_algebra.rs) | `linear_algebra` | LU and Cholesky factorizations, linear solves, and the direct 4x4 inverse under a latency + approximation-error stress test on well- and ill-conditioned inputs. Reproduces the LU / Cholesky / inverse accuracy tables in benches/linear_algebra.md. |
| [`svd`](svd.rs) | `linear_algebra::svd` | Singular value decomposition and Moore-Penrose pseudo-inverse under a robotics stress test (Kabsch rotation recovery, a redundant-arm pseudo-inverse, a near-singular Jacobian, and an overdetermined fit) with latency + approximation error. Reproduces the SVD / pseudo-inverse accuracy table in benches/linear_algebra.md. |
| [`root_finding`](root_finding.rs) | `root_finding` | Bracketed bisection, Newton with exact derivatives, damped (backtracking) Newton rescuing a far start, and a square-system Newton solve, each printed against its known root. |
| [`curve_fit`](curve_fit.rs) | `optimization` | Levenberg-Marquardt fit of `y = a·e^(b·t)` to sensor samples with exact autodiff Jacobians; prints recovered `a`, `b`, and `\|err\|`. |
- `optimization_solvers` — Gauss-Newton linear residual walkthrough
