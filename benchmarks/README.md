# Accuracy benchmarks

Per-module tables showing multicalc's numerics tested against established external libraries
(numpy, scipy, mpmath). Each row lists the operation, the equation, the tolerance it must match
within, and which library it is checked against. The tables are generated from the fixtures under
[`tools/qa`](../tools/qa) and kept in sync by a CI `git diff` guard.

| Module | Doc | What it covers |
| --- | --- | --- |
| calculus | [`calculus.md`](calculus.md) | Differentiation, partials, Jacobian / Hessian, vector-field operators, Taylor approximation, and single-variable quadrature. |
| linear_algebra | [`linear_algebra.md`](linear_algebra.md) | LU / Cholesky / column-pivoted QR / SVD factorizations and solves. |
| optimization | [`optimization.md`](optimization.md) | Levenberg-Marquardt and Gauss-Newton least-squares minimizers. |
| root_finding | [`root_finding.md`](root_finding.md) | Scalar and system root finders: bisection, Newton, damped Newton, square-system Newton. |
| ode | [`ode.md`](ode.md) | RK45 integrator trajectories against scipy `solve_ivp`. |

Regenerate with `cargo run -p multicalc-qa --bin gen_accuracy_tables`; CI fails if a regenerated
table differs from the committed doc. Runnable, self-checking demos live in [`demos/`](../demos).
