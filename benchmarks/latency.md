# Latency benchmarks

Criterion measurements in the optimized bench profile. **The numbers are machine-dependent.**
Regenerate with `cargo bench -p multicalc-qa`.

Measured on a 12th Gen Intel Core i7-12650H (16 threads) under WSL2 (Linux 6.18).

<!-- BEGIN generated: latency -->
| Operation | Equation | Median | Mean |
|-----------|----------|-------:|-----:|
| `jacobian_large` | Jacobian of a 6-in/6-out map | 217.5 ns | 218.2 ns |
| `lu_solve` | solve A·x = b (10×10) | 390.3 ns | 390.7 ns |
| `svd_solve` | least-squares fit (30×3) | 826.8 ns | 830.1 ns |
| `symmetric_eigen` | eigenvalues and directions (6×6) | 1.48 µs | 1.49 µs |
| `rk45_solve` | y″ = −y, adaptive to 2π | 16.16 µs | 16.23 µs |
| `rigid_body_step` | one 1 ms tick of a free body: gravity + steady push, orientation on the manifold | 71.6 ns | 71.5 ns |
| `inverse_dynamics_double_pendulum` | 2-joint arm, recursive Newton-Euler: (q, q̇, q̈) -> τ | 157.9 ns | 157.4 ns |
| `inverse_dynamics_arm` | 7-joint arm, recursive Newton-Euler: (q, q̇, q̈) -> τ | 483.0 ns | 482.0 ns |
| `joint_space_inertia_double_pendulum` | 2-joint arm, composite-rigid-body: q -> H(q) | 141.1 ns | 141.8 ns |
| `joint_space_inertia_arm` | 7-joint arm, composite-rigid-body: q -> H(q) | 609.5 ns | 611.7 ns |
| `forward_dynamics_arm` | 7-joint arm, articulated-body: (q, q̇, τ) -> q̈, workspace on the stack | 722.8 ns | 722.5 ns |
| `forward_dynamics_arm_with_workspace` | 7-joint arm, articulated-body: (q, q̇, τ) -> q̈, caller-owned workspace | 697.5 ns | 696.0 ns |
| `momentum_observer_arm` | 7-joint arm, momentum observer: one 1 kHz tick (q, q̇, τ) -> r, solved poses | 454.9 ns | 452.5 ns |
| `inverse_kinematics_solve` | 7-joint arm, warm-started SE(3) pose solve with joint limits | 20.41 µs | 20.48 µs |
| `computed_torque_arm` | 7-joint arm, computed torque: (q, q̇, reference) -> τ, one RNEA pass | 522.6 ns | 521.5 ns |
| `cartesian_impedance_arm` | 7-joint arm, Cartesian impedance: Jᵀ·(k⊙e + d⊙ė) + bias | 689.5 ns | 687.7 ns |
| `cartesian_impedance_arm_with_posture` | the same, plus a null-space posture term (damped pseudo-inverse) | 3.12 µs | 3.13 µs |
| `position_servo_step` | 7 joints, one exactly-discretized servo tick | 9.9 ns | 9.8 ns |
| `newton_system` | x²+y² = 4, x·y = 1 | 175.8 ns | 174.9 ns |
| `particle_filter` | 1000 particles, diff-drive motion + process noise + systematic resample | 65.70 µs | 65.42 µs |
| `kalman_filter_step` | 2-state constant velocity, predict + update | 35.9 ns | 36.0 ns |
| `extended_kalman_filter_step` | 5-state coordinated turn + position fix, autodiff Jacobians, predict + update | 468.1 ns | 466.1 ns |
| `error_state_kalman_filter_predict` | 15-state error filter, one IMU step: closed-form transition + covariance | 1.56 µs | 1.56 µs |
| `unscented_kalman_filter_step` | 5-state coordinated turn + position fix, 11 sampled points, predict + update | 426.4 ns | 427.4 ns |
| `polynomial_root_solve` | six real roots of a sixth-power polynomial | 48.36 µs | 48.26 µs |
| `minimum_snap_plan` | smoothest path through 8 waypoints in 3D | 19.38 µs | 19.41 µs |
| `infinite_horizon_lqr_solve` | solve the Riccati equation and form K, 4 states / 1 input | 1.94 µs | 1.94 µs |

<!-- END generated -->
