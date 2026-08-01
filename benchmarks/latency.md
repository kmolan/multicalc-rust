# Latency benchmarks

Latency measurements with criterion in
the optimized bench profile. **Numbers are machine-dependent**.
Manually regenerate with `cargo bench -p multicalc-qa`.

Measured on a 12th Gen Intel Core i7-12650H (16 threads) under WSL2 (Linux 6.18).

<!-- BEGIN generated: latency -->
| Operation | Equation | Median | Mean |
|-----------|----------|-------:|-----:|
| `derivative` | d³/dx³(x²·sin x) at x = 1 | 27.9 ns | 28.1 ns |
| `jacobian_small` | Jacobian of (x·y·z, x²+y²) | 9.4 ns | 9.3 ns |
| `jacobian_large` | Jacobian of a 6-in/6-out map | 269.6 ns | 270.3 ns |
| `gauss_quad` | ∫₀¹ (sin x − √x)·e⁻ˣ dx | 131.3 ns | 130.7 ns |
| `lu_solve` | solve A·x = b (10×10) | 251.9 ns | 251.4 ns |
| `svd_solve` | least-squares fit (30×3) | 835.1 ns | 835.3 ns |
| `expm` | matrix exponential eᴬ (6×6) | 718.7 ns | 715.1 ns |
| `symmetric_eigen` | eigenvalues and directions (6×6) | 1.54 µs | 1.55 µs |
| `rk45_solve` | y″ = −y, adaptive to 2π | 16.69 µs | 16.81 µs |
| `rk4_integrate` | y″ = −y, fixed-step to 2π | 16.46 µs | 16.59 µs |
| `lev_marq` | fit y = a·eᵇᵗ (8 points) | 2.01 µs | 2.01 µs |
| `newton_system` | x²+y² = 4, x·y = 1 | 272.6 ns | 273.2 ns |
| `particle_filter` | 1000 particles, diff-drive motion + process noise + systematic resample | 91.87 µs | 92.41 µs |
| `kalman_filter_step` | 2-state constant velocity, predict + update | 31.6 ns | 31.7 ns |
| `extended_kalman_filter_step` | 5-state coordinated turn + position fix, autodiff Jacobians, predict + update | 533.1 ns | 530.7 ns |
| `unscented_kalman_filter_step` | 5-state coordinated turn + position fix, 11 sampled points, predict + update | 465.5 ns | 468.6 ns |
| `polynomial_evaluate` | p(x) and its first two derivatives, seventh power | 43.3 ns | 43.1 ns |
| `polynomial_real_roots_sturm` | six real roots of a sixth-power polynomial | 49.90 µs | 50.07 µs |
| `multivariate_evaluate` | an eight-term polynomial in three variables | 27.8 ns | 27.8 ns |
| `minimum_snap_plan` | smoothest path through 8 waypoints in 3D | 16.90 µs | 16.85 µs |
| `pid_update` | one PID tick with anti-windup and output limits | 3.5 ns | 3.4 ns |
| `pure_pursuit` | κ = 2·sin(α)/L_d toward a lookahead point | 11.3 ns | 11.3 ns |
| `biquad_filter` | one 2nd-order low-pass sample, 50 Hz at 1 kHz | 2.5 ns | 2.5 ns |
| `biquad_cascade_filter` | one sample through a 3-section notch at 80/160/240 Hz | 3.3 ns | 3.3 ns |
| `multi_channel_biquad_filter` | one 3-axis sample through a 50 Hz low-pass | 2.8 ns | 2.8 ns |
| `follow_the_gap` | 61-beam scan, widest gap the robot fits through + speed ramp | 310.2 ns | 310.1 ns |

<!-- END generated -->
