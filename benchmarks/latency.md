# Latency benchmarks

Latency measurements with criterion in
the optimized bench profile. **Numbers are machine-dependent**.
Manually regenerate with `cargo bench -p multicalc-qa`.

Measured on a 12th Gen Intel Core i7-12650H (16 threads) under WSL2 (Linux 6.18).

<!-- BEGIN generated: latency -->
| Operation | Equation | Median | Mean |
|-----------|----------|-------:|-----:|
| `jacobian_large` | Jacobian of a 6-in/6-out map | 273.1 ns | 273.6 ns |
| `lu_solve` | solve A·x = b (10×10) | 402.6 ns | 403.3 ns |
| `svd_solve` | least-squares fit (30×3) | 823.3 ns | 825.4 ns |
| `symmetric_eigen` | eigenvalues and directions (6×6) | 1.55 µs | 1.56 µs |
| `rk45_solve` | y″ = −y, adaptive to 2π | 16.86 µs | 16.89 µs |
| `rigid_body_step` | one 1 ms tick of a free body: gravity + steady push, orientation on the manifold | 95.8 ns | 96.5 ns |
| `inverse_dynamics_double_pendulum` | 2-joint arm, recursive Newton-Euler: (q, q̇, q̈) -> τ | 162.0 ns | 162.3 ns |
| `inverse_dynamics_arm` | 7-joint arm, recursive Newton-Euler: (q, q̇, q̈) -> τ | 489.6 ns | 492.5 ns |
| `joint_space_inertia_double_pendulum` | 2-joint arm, composite-rigid-body: q -> H(q) | 149.1 ns | 149.5 ns |
| `joint_space_inertia_arm` | 7-joint arm, composite-rigid-body: q -> H(q) | 599.2 ns | 602.3 ns |
| `forward_dynamics_arm` | 7-joint arm, articulated-body: (q, q̇, τ) -> q̈, workspace on the stack | 675.2 ns | 677.9 ns |
| `forward_dynamics_arm_with_workspace` | 7-joint arm, articulated-body: (q, q̇, τ) -> q̈, caller-owned workspace | 646.0 ns | 648.5 ns |
| `inverse_kinematics_solve` | 7-joint arm, warm-started SE(3) pose solve with joint limits | 21.88 µs | 21.99 µs |
| `computed_torque_arm` | 7-joint arm, computed torque: (q, q̇, reference) -> τ, one RNEA pass | 503.3 ns | 504.2 ns |
| `cartesian_impedance_arm` | 7-joint arm, Cartesian impedance: Jᵀ·(k⊙e + d⊙ė) + bias | 696.2 ns | 695.6 ns |
| `cartesian_impedance_arm_with_posture` | the same, plus a null-space posture term (damped pseudo-inverse) | 3.30 µs | 3.32 µs |
| `position_servo_step` | 7 joints, one exactly-discretized servo tick | 9.1 ns | 9.3 ns |
| `newton_system` | x²+y² = 4, x·y = 1 | 329.4 ns | 339.1 ns |
| `particle_filter` | 1000 particles, diff-drive motion + process noise + systematic resample | 65.67 µs | 65.82 µs |
| `kalman_filter_step` | 2-state constant velocity, predict + update | 34.1 ns | 34.2 ns |
| `extended_kalman_filter_step` | 5-state coordinated turn + position fix, autodiff Jacobians, predict + update | 679.1 ns | 678.0 ns |
| `error_state_kalman_filter_predict` | 15-state error filter, one IMU step: closed-form transition + covariance | 2.04 µs | 2.04 µs |
| `unscented_kalman_filter_step` | 5-state coordinated turn + position fix, 11 sampled points, predict + update | 517.6 ns | 519.8 ns |
| `polynomial_root_solve` | six real roots of a sixth-power polynomial | 50.13 µs | 50.34 µs |
| `minimum_snap_plan` | smoothest path through 8 waypoints in 3D | 20.11 µs | 20.22 µs |
| `infinite_horizon_lqr_solve` | solve the Riccati equation and form K, 4 states / 1 input | 2.59 µs | 2.60 µs |

<!-- END generated -->
