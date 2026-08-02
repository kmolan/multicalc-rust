# Latency benchmarks

Latency measurements with criterion in
the optimized bench profile. **Numbers are machine-dependent**.
Manually regenerate with `cargo bench -p multicalc-qa`.

Measured on a 12th Gen Intel Core i7-12650H (16 threads) under WSL2 (Linux 6.18).

<!-- BEGIN generated: latency -->
| Operation | Equation | Median | Mean |
|-----------|----------|-------:|-----:|
| `jacobian_large` | Jacobian of a 6-in/6-out map | 302.0 ns | 304.7 ns |
| `lu_solve` | solve A·x = b (10×10) | 271.6 ns | 273.5 ns |
| `svd_solve` | least-squares fit (30×3) | 955.9 ns | 960.9 ns |
| `symmetric_eigen` | eigenvalues and directions (6×6) | 1.73 µs | 1.75 µs |
| `rk45_solve` | y″ = −y, adaptive to 2π | 19.29 µs | 19.37 µs |
| `rigid_body_step` | one 1 ms tick of a free body: gravity + steady push, orientation on the manifold | 109.7 ns | 110.4 ns |
| `newton_system` | x²+y² = 4, x·y = 1 | 298.2 ns | 301.3 ns |
| `particle_filter` | 1000 particles, diff-drive motion + process noise + systematic resample | 108.24 µs | 109.10 µs |
| `kalman_filter_step` | 2-state constant velocity, predict + update | 39.5 ns | 39.9 ns |
| `extended_kalman_filter_step` | 5-state coordinated turn + position fix, autodiff Jacobians, predict + update | 579.1 ns | 586.6 ns |
| `error_state_kalman_filter_predict` | 15-state error filter, one IMU step: closed-form transition + covariance | 2.16 µs | 2.16 µs |
| `unscented_kalman_filter_step` | 5-state coordinated turn + position fix, 11 sampled points, predict + update | 506.3 ns | 512.6 ns |
| `polynomial_root_solve` | six real roots of a sixth-power polynomial | 55.53 µs | 56.11 µs |
| `minimum_snap_plan` | smoothest path through 8 waypoints in 3D | 18.22 µs | 18.21 µs |
| `infinite_horizon_lqr_solve` | solve the Riccati equation and form K, 4 states / 1 input | 2.44 µs | 2.49 µs |

<!-- END generated -->
