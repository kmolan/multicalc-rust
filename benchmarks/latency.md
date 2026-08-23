# Latency benchmarks

Latency measurements with criterion in
the optimized bench profile. **Numbers are machine-dependent**.
Manually regenerate with `cargo bench -p multicalc-qa`.

Measured on a 12th Gen Intel Core i7-12650H (16 threads) under WSL2 (Linux 6.18).

<!-- BEGIN generated: latency -->
| Operation | Equation | Median | Mean |
|-----------|----------|-------:|-----:|
| `jacobian_large` | Jacobian of a 6-in/6-out map | 417.9 ns | 419.1 ns |
| `lu_solve` | solve A·x = b (10×10) | 406.1 ns | 410.8 ns |
| `svd_solve` | least-squares fit (30×3) | 837.9 ns | 845.3 ns |
| `symmetric_eigen` | eigenvalues and directions (6×6) | 1.74 µs | 1.75 µs |
| `rk45_solve` | y″ = −y, adaptive to 2π | 16.68 µs | 16.71 µs |
| `rigid_body_step` | one 1 ms tick of a free body: gravity + steady push, orientation on the manifold | 101.5 ns | 102.0 ns |
| `inverse_dynamics_double_pendulum` | 2-joint arm, recursive Newton-Euler: (q, q̇, q̈) -> τ | 168.7 ns | 170.1 ns |
| `inverse_dynamics_arm` | 7-joint arm, recursive Newton-Euler: (q, q̇, q̈) -> τ | 509.3 ns | 516.4 ns |
| `joint_space_inertia_double_pendulum` | 2-joint arm, composite-rigid-body: q -> H(q) | 154.1 ns | 154.9 ns |
| `joint_space_inertia_arm` | 7-joint arm, composite-rigid-body: q -> H(q) | 616.9 ns | 617.6 ns |
| `forward_dynamics_arm` | 7-joint arm, articulated-body: (q, q̇, τ) -> q̈, workspace on the stack | 743.4 ns | 751.2 ns |
| `forward_dynamics_arm_with_workspace` | 7-joint arm, articulated-body: (q, q̇, τ) -> q̈, caller-owned workspace | 667.5 ns | 664.3 ns |
| `inverse_kinematics_solve` | 7-joint arm, warm-started SE(3) pose solve with joint limits | 22.42 µs | 22.46 µs |
| `newton_system` | x²+y² = 4, x·y = 1 | 186.2 ns | 185.6 ns |
| `particle_filter` | 1000 particles, diff-drive motion + process noise + systematic resample | 74.16 µs | 74.48 µs |
| `kalman_filter_step` | 2-state constant velocity, predict + update | 34.0 ns | 34.2 ns |
| `extended_kalman_filter_step` | 5-state coordinated turn + position fix, autodiff Jacobians, predict + update | 536.5 ns | 532.2 ns |
| `error_state_kalman_filter_predict` | 15-state error filter, one IMU step: closed-form transition + covariance | 1.98 µs | 1.96 µs |
| `unscented_kalman_filter_step` | 5-state coordinated turn + position fix, 11 sampled points, predict + update | 486.7 ns | 482.1 ns |
| `polynomial_root_solve` | six real roots of a sixth-power polynomial | 53.34 µs | 53.35 µs |
| `minimum_snap_plan` | smoothest path through 8 waypoints in 3D | 21.18 µs | 21.09 µs |
| `infinite_horizon_lqr_solve` | solve the Riccati equation and form K, 4 states / 1 input | 2.35 µs | 2.35 µs |

<!-- END generated -->
