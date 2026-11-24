# Benchmarks

Generated tables: **accuracy** against external references, **latency** from criterion, and
**silicon** from Board runs.

## Accuracy

Each row gives the operation, the equation, the tolerance, and the reference library (numpy,
SciPy, mpmath, FilterPy, MuJoCo, mink, Pinocchio). `gen_accuracy_tables` renders the tables from the
fixtures under [`tools/qa`](../tools/qa). CI regenerates them and fails on a `git diff`.

| Module | Doc | Coverage |
| --- | --- | --- |
| calculus | [`calculus.md`](calculus.md) | Derivatives, partials, Jacobian / Hessian, curl / divergence, line and flux integrals, Taylor approximation, and single-variable quadrature, against mpmath. |
| linear_algebra | [`linear_algebra.md`](linear_algebra.md) | LU / Cholesky / column-pivoted QR / SVD factorizations and solves, and the symmetric eigendecomposition, against numpy/LAPACK. |
| optimization | [`optimization.md`](optimization.md) | Levenberg-Marquardt and Gauss-Newton least squares, against SciPy/MINPACK. |
| root_finding | [`root_finding.md`](root_finding.md) | Bisection, Newton, damped Newton and square-system Newton, against SciPy. |
| ode | [`ode.md`](ode.md) | RK45 trajectories, exponential-map attitude steps and a free-body step, against SciPy `solve_ivp`. |
| estimation | [`estimation.md`](estimation.md) | Linear, extended and unscented Kalman filters against FilterPy; error-state Kalman filter, TRIAD, Mahony and Madgwick against numpy / SciPy. |
| control | [`control.md`](control.md) | Riccati and Lyapunov solves against SciPy, the geometric attitude law, and the model-based arm laws against Pinocchio. |
| signal_processing | [`signal_processing.md`](signal_processing.md) | Biquad design, frequency response and filtered output against `scipy.signal`, with a fourth-order cascade and a harmonic notch. |
| mjcf | [`mjcf.md`](mjcf.md) | MJCF ingestion (body tree, joint settings, mass, COM, inertia) against the MuJoCo compile of the same file. |
| urdf | [`urdf.md`](urdf.md) | URDF ingestion (link list, joint order and kinds, limits, mimic joints, link poses) against the Pinocchio parse of the same file. |
| kinematics | [`kinematics.md`](kinematics.md) | Forward kinematics and geometric Jacobians against MuJoCo; differential IK against mink. |
| dynamics | [`dynamics.md`](dynamics.md) | Single rigid-body accelerations against MuJoCo; RNEA, CRBA and ABA against Pinocchio. |
| spatial | [`spatial.md`](spatial.md) | Plücker transforms of twists and wrenches, the spatial cross products and power, and spatial-inertia algebra (momentum, bias wrench `v ×* (I·v)`, kinetic energy, frame change, composition), against Pinocchio. |
| polynomial | [`polynomial.md`](polynomial.md) | Evaluation, derivatives, integrals, products, composition, interpolation, fitting, exact real roots, and multivariate polynomials, against numpy. |
| motion | [`motion.md`](motion.md) | Minimum-snap coefficients and sampled states, and S-curve / trapezoidal profiles, against independent numpy / SciPy solves. |
| mapping | [`mapping.md`](mapping.md) | Exact Euclidean distance transform, costmap inflation, and the likelihood-field measurement model, against SciPy and closed-form numpy. |

Regenerate with `cargo run -p multicalc-qa --bin gen_accuracy_tables`. Runnable, self-checking
demos are in [`demos/`](../demos).

## Latency

[`latency.md`](latency.md): criterion medians and means in the optimized bench profile, with the
operation and the equation of each row. The file records the machine. Timings are
machine-dependent, so CI does not guard this table.

Regenerate with `cargo bench -p multicalc-qa`.

## Silicon

[`silicon.md`](silicon.md): the best and the worst warm time of each check of the HILShare apps on
the Nucleo-F411RE, the Raspberry Pi Pico 2 and the XIAO ESP32-C3. The boards are not in CI, so CI
does not guard this table.

Regenerate with `cargo run -p multicalc-qa --bin gen_silicon_table`, as `silicon.md` describes.
