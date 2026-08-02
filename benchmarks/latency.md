# Latency benchmarks

Latency measurements with criterion in
the optimized bench profile. **Numbers are machine-dependent**.
Manually regenerate with `cargo bench -p multicalc-qa`.

Measured on a 12th Gen Intel Core i7-12650H (16 threads) under WSL2 (Linux 6.18).

<!-- BEGIN generated: latency -->
| Operation | Equation | Median | Mean |
|-----------|----------|-------:|-----:|
| `derivative` | d³/dx³(x²·sin x) at x = 1 | 27.6 ns | 27.8 ns |
| `jacobian_small` | Jacobian of (x·y·z, x²+y²) | 1.9 ns | 1.9 ns |
| `jacobian_large` | Jacobian of a 6-in/6-out map | 259.2 ns | 260.0 ns |
| `gauss_quad` | ∫₀¹ (sin x − √x)·e⁻ˣ dx | 133.6 ns | 131.6 ns |
| `lu_solve` | solve A·x = b (10×10) | 247.6 ns | 247.7 ns |
| `svd_solve` | least-squares fit (30×3) | 818.5 ns | 820.0 ns |
| `expm` | matrix exponential eᴬ (6×6) | 695.4 ns | 695.0 ns |
| `symmetric_eigen` | eigenvalues and directions (6×6) | 1.57 µs | 1.57 µs |
| `rk45_solve` | y″ = −y, adaptive to 2π | 16.52 µs | 16.55 µs |
| `rk4_integrate` | y″ = −y, fixed-step to 2π | 16.32 µs | 16.39 µs |
| `lev_marq` | fit y = a·eᵇᵗ (8 points) | 2.08 µs | 2.09 µs |
| `newton_system` | x²+y² = 4, x·y = 1 | 272.7 ns | 274.8 ns |
| `particle_filter` | 1000 particles, diff-drive motion + process noise + systematic resample | 92.77 µs | 92.58 µs |
| `kalman_filter_step` | 2-state constant velocity, predict + update | 34.2 ns | 34.5 ns |
| `extended_kalman_filter_step` | 5-state coordinated turn + position fix, autodiff Jacobians, predict + update | 556.8 ns | 556.9 ns |
| `unscented_kalman_filter_step` | 5-state coordinated turn + position fix, 11 sampled points, predict + update | 476.3 ns | 482.8 ns |
| `polynomial_evaluate` | p(x) and its first two derivatives, seventh power | 45.0 ns | 45.6 ns |
| `polynomial_real_roots_sturm` | six real roots of a sixth-power polynomial | 51.78 µs | 52.54 µs |
| `multivariate_evaluate` | an eight-term polynomial in three variables | 27.9 ns | 27.8 ns |
| `minimum_snap_plan` | smoothest path through 8 waypoints in 3D | 16.20 µs | 16.32 µs |
| `pid_update` | one PID tick with anti-windup and output limits | 8.0 ns | 8.1 ns |
| `lqr_design` | solve the Riccati equation and form K, 4 states / 1 input | 2.25 µs | 2.25 µs |
| `pure_pursuit` | κ = 2·sin(α)/L_d toward a lookahead point | 11.1 ns | 11.2 ns |
| `biquad_filter` | one 2nd-order low-pass sample, 50 Hz at 1 kHz | 2.5 ns | 2.5 ns |
| `biquad_cascade_filter` | one sample through a 3-section notch at 80/160/240 Hz | 3.3 ns | 3.3 ns |
| `multi_channel_biquad_filter` | one 3-axis sample through a 50 Hz low-pass | 2.9 ns | 2.8 ns |
| `follow_the_gap` | 61-beam scan, widest gap the robot fits through + speed ramp | 296.7 ns | 295.1 ns |

<!-- END generated -->
