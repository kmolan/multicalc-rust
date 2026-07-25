# Latency benchmarks

Latency measurements with criterion in
the optimized bench profile. **Numbers are machine-dependent**.
Manually regenerate with `cargo bench -p multicalc-qa`.

Measured on a 12th Gen Intel Core i7-12650H (16 threads) under WSL2 (Linux 6.18).

<!-- BEGIN generated: latency -->
| Operation | Equation | Median | Mean |
|-----------|----------|-------:|-----:|
| `derivative` | d³/dx³(x²·sin x) at x = 1 | 27.8 ns | 27.9 ns |
| `jacobian_small` | Jacobian of (x·y·z, x²+y²) | 9.4 ns | 9.4 ns |
| `jacobian_large` | Jacobian of a 6-in/6-out map | 268.6 ns | 269.4 ns |
| `gauss_quad` | ∫₀¹ (sin x − √x)·e⁻ˣ dx | 135.2 ns | 135.2 ns |
| `lu_solve` | solve A·x = b (10×10) | 247.2 ns | 248.9 ns |
| `svd_solve` | least-squares fit (30×3) | 824.7 ns | 825.2 ns |
| `expm` | matrix exponential eᴬ (6×6) | 672.2 ns | 669.8 ns |
| `rk45_solve` | y″ = −y, adaptive to 2π | 16.59 µs | 16.64 µs |
| `rk4_integrate` | y″ = −y, fixed-step to 2π | 16.39 µs | 16.41 µs |
| `lev_marq` | fit y = a·eᵇᵗ (8 points) | 2.04 µs | 2.04 µs |
| `newton_system` | x²+y² = 4, x·y = 1 | 273.1 ns | 273.4 ns |
| `particle_filter` | 1000 particles, diff-drive motion + process noise + systematic resample | 86.98 µs | 87.46 µs |
| `kalman_filter_step` | 2-state constant velocity, predict + update | 33.6 ns | 33.7 ns |
| `extended_kalman_filter_step` | 5-state coordinated turn + position fix, autodiff Jacobians, predict + update | 508.5 ns | 506.2 ns |
| `pid_update` | one PID tick with anti-windup and output limits | 3.5 ns | 3.5 ns |
| `pure_pursuit` | κ = 2·sin(α)/L_d toward a lookahead point | 11.6 ns | 11.7 ns |
| `follow_the_gap` | 61-beam scan, widest gap the robot fits through + speed ramp | 285.6 ns | 286.0 ns |

<!-- END generated -->
