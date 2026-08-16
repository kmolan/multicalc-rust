# Benchmarks

Generated tables of multicalc's behavior: **accuracy** (correctness against external references)
and **latency** (criterion measurements).

## Accuracy

Per-module tables showing multicalc's numerics tested against established external libraries
(numpy, scipy, mpmath, filterpy, MuJoCo). Each row lists the operation, the equation, the tolerance it must match
within, and which library it is checked against. The tables are generated from the fixtures under
[`tools/qa`](../tools/qa) and kept in sync by a CI `git diff` guard.

| Module | Doc | What it covers |
| --- | --- | --- |
| calculus | [`calculus.md`](calculus.md) | Differentiation, partials, Jacobian / Hessian, vector-field operators, Taylor approximation, and single-variable quadrature. |
| linear_algebra | [`linear_algebra.md`](linear_algebra.md) | LU / Cholesky / column-pivoted QR / SVD factorizations and solves. |
| optimization | [`optimization.md`](optimization.md) | Levenberg-Marquardt and Gauss-Newton least-squares minimizers. |
| root_finding | [`root_finding.md`](root_finding.md) | Scalar and system root finders: bisection, Newton, damped Newton, square-system Newton. |
| ode | [`ode.md`](ode.md) | RK45 integrator trajectories against scipy `solve_ivp`. |
| estimation | [`estimation.md`](estimation.md) | Linear and extended Kalman filter predict/update runs against FilterPy, including a coordinated-turn motion model. |
| control | [`control.md`](control.md) | Riccati solver, and the Lyapunov certificate against scipy, plus the geometric attitude law's conventions. |
| signal_processing | [`signal_processing.md`](signal_processing.md) | Biquad design, frequency response, and filtered output against `scipy.signal`, including a fourth-order cascade and a harmonic notch. |
| mjcf | [`mjcf.md`](mjcf.md) | Reading a MuJoCo model file into multicalc's robot types — a free body's mass and spin, and a jointed robot's body tree, joint settings and per-link mass — against MuJoCo's own compile of it. |
| urdf | [`urdf.md`](urdf.md) | Reading a URDF model file into multicalc's robot types — the link list, joint order and kinds, travel limits and coupled joints, and where every link ends up — against Pinocchio's own parse of it. |
| kinematics | [`kinematics.md`](kinematics.md) | Working out where every part of a jointed robot ends up from its joint readings, and how each joint's rate moves each part, against MuJoCo's own solve of the same model — plus the reverse, the joint readings that put a chosen frame at a pose you name, against mink's differential IK. |
| dynamics | [`dynamics.md`](dynamics.md) | One rigid body's straight-line and turning accelerations under gravity and an applied push and turn, against MuJoCo's own solve of the same body. |
| polynomial | [`polynomial.md`](polynomial.md) | Evaluation, derivatives, areas, products, composition, interpolation, fitting, exact real roots, and polynomials in several variables, against numpy. |
| motion | [`motion.md`](motion.md) | Minimum-snap trajectory coefficients and sampled states, against an independent constrained solve in numpy. |

Regenerate with `cargo run -p multicalc-qa --bin gen_accuracy_tables`; CI fails if a regenerated
table differs from the committed doc. Runnable, self-checking demos live in [`demos/`](../demos).

## Latency

[`latency.md`](latency.md) —
Measured with criterion in the optimized bench profile. Each row lists the operation, the equation
it evaluates, and the median / mean time. Timings are machine-dependent, so this table is **not**
CI-guarded (unlike the accuracy tables), and the file records the machine it was measured on.

Regenerate with `cargo bench -p multicalc-qa`.
