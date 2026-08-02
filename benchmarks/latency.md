# Latency benchmarks

Latency measurements with criterion in
the optimized bench profile. **Numbers are machine-dependent**.
Manually regenerate with `cargo bench -p multicalc-qa`.

Measured on a 12th Gen Intel Core i7-12650H (16 threads) under WSL2 (Linux 6.18).

<!-- BEGIN generated: latency -->
| Operation | Equation | Median | Mean |
|-----------|----------|-------:|-----:|
| `jacobian_large` | Jacobian of a 6-in/6-out map | 268.2 ns | 269.4 ns |
| `lu_solve` | solve A·x = b (10×10) | 240.1 ns | 240.3 ns |
| `svd_solve` | least-squares fit (30×3) | 810.9 ns | 815.6 ns |
| `symmetric_eigen` | eigenvalues and directions (6×6) | 1.50 µs | 1.50 µs |
| `rk45_solve` | y″ = −y, adaptive to 2π | 16.67 µs | 16.78 µs |
| `newton_system` | x²+y² = 4, x·y = 1 | 271.8 ns | 272.8 ns |
| `particle_filter` | 1000 particles, diff-drive motion + process noise + systematic resample | 88.99 µs | 89.48 µs |
| `kalman_filter_step` | 2-state constant velocity, predict + update | 34.1 ns | 34.4 ns |
| `extended_kalman_filter_step` | 5-state coordinated turn + position fix, autodiff Jacobians, predict + update | 516.6 ns | 514.9 ns |
| `error_state_kalman_filter_predict` | 15-state error filter, one IMU step: closed-form transition + covariance | 1.91 µs | 1.96 µs |
| `unscented_kalman_filter_step` | 5-state coordinated turn + position fix, 11 sampled points, predict + update | 479.3 ns | 481.8 ns |
| `polynomial_root_solve` | six real roots of a sixth-power polynomial | 49.48 µs | 50.14 µs |
| `minimum_snap_plan` | smoothest path through 8 waypoints in 3D | 16.26 µs | 16.38 µs |
| `infinite_horizon_lqr_solve` | solve the Riccati equation and form K, 4 states / 1 input | 2.34 µs | 2.37 µs |

<!-- END generated -->
