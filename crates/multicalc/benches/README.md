# Benchmarks

The `criterion` benchmark suites for `multicalc`. Each suite has a results doc next to its
source: **accuracy** figures (approximation error against the analytic value) and **latency**
figures (wall-clock time per call).

| Suite | Source | Results | What it covers |
| --- | --- | --- | --- |
| calculus | [`calculus.rs`](calculus.rs) | [`calculus.md`](calculus.md) | Single- and multi-variable differentiation, iterative integration (Boole / Simpson / Trapezoidal), Gaussian quadrature, Jacobian / Hessian, vector field, and approximation. |
| linear_algebra | [`linear_algebra.rs`](linear_algebra.rs) | [`linear_algebra.md`](linear_algebra.md) | Vector and matrix ops, and the LU / Cholesky / column-pivoted QR / SVD factorizations. |
| optimization | [`optimization.rs`](optimization.rs) | [`optimization.md`](optimization.md) | Nonlinear least-squares solvers: Levenberg-Marquardt and Gauss-Newton. |
| root_finding | [`root_finding.rs`](root_finding.rs) | [`root_finding.md`](root_finding.md) | Scalar and system root finders: bisection, Newton, damped Newton, and square-system Newton. |

Accuracy vs latency: every suite doc reports latency (wall-clock time per call); all four also
report accuracy (how close the result lands to the known value). The examples in
[`examples/`](../examples) reproduce those accuracy tables, so the published figures stay honest.

## Running

```sh
cargo bench                       # all suites
cargo bench --bench calculus      # one suite
```

Each suite sets a criterion sample size of 50, a 500 ms warm-up, and a 2 s measurement window.

## Environment

The published latency figures were measured on: 12th Gen Intel Core i7-12650H · `rustc` 1.95.0
(release, `opt-level = 3`) · criterion 0.5 · WSL2 (Ubuntu). Latency is wall-clock and therefore
machine- and build-specific; re-run locally for numbers that match your hardware. Iterative
integrals use the default **120** intervals; Gaussian quadrature uses the listed order.
