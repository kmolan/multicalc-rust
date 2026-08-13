# Latency benchmarks

Latency measurements with criterion in
the optimized bench profile. **Numbers are machine-dependent**.
Manually regenerate with `cargo bench -p multicalc-qa`.

Measured on a 12th Gen Intel Core i7-12650H (16 threads) under WSL2 (Linux 6.18).

<!-- BEGIN generated: latency -->
| Operation | Equation | Median | Mean |
|-----------|----------|-------:|-----:|
| `jacobian_large` | Jacobian of a 6-in/6-out map | 274.9 ns | 276.5 ns |
| `lu_solve` | solve A·x = b (10×10) | 410.3 ns | 411.3 ns |
| `svd_solve` | least-squares fit (30×3) | 838.4 ns | 841.0 ns |
| `symmetric_eigen` | eigenvalues and directions (6×6) | 1.55 µs | 1.56 µs |
| `rk45_solve` | y″ = −y, adaptive to 2π | 17.01 µs | 17.08 µs |
| `rigid_body_step` | one 1 ms tick of a free body: gravity + steady push, orientation on the manifold | 96.9 ns | 97.5 ns |
| `inverse_kinematics_solve` | 7-joint arm, warm-started SE(3) pose solve with joint limits | 22.70 µs | 22.88 µs |
| `newton_system` | x²+y² = 4, x·y = 1 | 324.1 ns | 326.0 ns |
| `particle_filter` | 1000 particles, diff-drive motion + process noise + systematic resample | 65.09 µs | 65.10 µs |
| `kalman_filter_step` | 2-state constant velocity, predict + update | 33.9 ns | 34.0 ns |
| `extended_kalman_filter_step` | 5-state coordinated turn + position fix, autodiff Jacobians, predict + update | 516.8 ns | 516.6 ns |
| `error_state_kalman_filter_predict` | 15-state error filter, one IMU step: closed-form transition + covariance | 1.93 µs | 1.94 µs |
| `unscented_kalman_filter_step` | 5-state coordinated turn + position fix, 11 sampled points, predict + update | 460.6 ns | 462.5 ns |
| `polynomial_root_solve` | six real roots of a sixth-power polynomial | 50.71 µs | 50.78 µs |
| `minimum_snap_plan` | smoothest path through 8 waypoints in 3D | 20.07 µs | 20.05 µs |
| `infinite_horizon_lqr_solve` | solve the Riccati equation and form K, 4 states / 1 input | 2.23 µs | 2.24 µs |

<!-- END generated -->
