# Latency benchmarks

Latency measurements with criterion in
the optimized bench profile. **Numbers are machine-dependent**.
Manually regenerate with `cargo bench -p multicalc-qa`.

Measured on a 12th Gen Intel Core i7-12650H (16 threads) under WSL2 (Linux 6.18).

<!-- BEGIN generated: latency -->
| Operation | Equation | Median | Mean |
|-----------|----------|-------:|-----:|
| `derivative` | d³/dx³(x²·sin x) at x = 1 | 33.1 ns | 33.7 ns |
| `jacobian_small` | Jacobian of (x·y·z, x²+y²) | 10.8 ns | 10.9 ns |
| `jacobian_large` | Jacobian of a 6-in/6-out map | 307.3 ns | 318.9 ns |
| `gauss_quad` | ∫₀¹ (sin x − √x)·e⁻ˣ dx | 143.4 ns | 146.9 ns |
| `lu_solve` | solve A·x = b (10×10) | 280.1 ns | 280.6 ns |
| `svd_solve` | least-squares fit (30×3) | 958.8 ns | 961.7 ns |
| `expm` | matrix exponential eᴬ (6×6) | 780.1 ns | 783.7 ns |
| `rk45_solve` | y″ = −y, adaptive to 2π | 19.84 µs | 20.00 µs |
| `rk4_integrate` | y″ = −y, fixed-step to 2π | 20.02 µs | 20.53 µs |
| `lev_marq` | fit y = a·eᵇᵗ (8 points) | 2.32 µs | 2.34 µs |
| `newton_system` | x²+y² = 4, x·y = 1 | 324.0 ns | 331.5 ns |
| `particle_filter` | 1000 particles, diff-drive motion + process noise + systematic resample | 105.66 µs | 109.64 µs |
| `kalman_filter_step` | 2-state constant velocity, predict + update | 38.0 ns | 38.7 ns |
| `extended_kalman_filter_step` | 5-state coordinated turn + position fix, autodiff Jacobians, predict + update | 599.7 ns | 609.2 ns |
| `polynomial_evaluate` | p(x) and its first two derivatives, seventh power | 49.1 ns | 49.5 ns |
| `polynomial_real_roots_sturm` | six real roots of a sixth-power polynomial | 55.06 µs | 55.01 µs |
| `multivariate_evaluate` | an eight-term polynomial in three variables | 32.2 ns | 32.1 ns |
| `minimum_snap_plan` | smoothest path through 8 waypoints in 3D | 18.43 µs | 18.52 µs |
| `pid_update` | one PID tick with anti-windup and output limits | 3.6 ns | 3.7 ns |
| `pure_pursuit` | κ = 2·sin(α)/L_d toward a lookahead point | 13.2 ns | 13.3 ns |
| `biquad_filter` | one 2nd-order low-pass sample, 50 Hz at 1 kHz | 3.0 ns | 3.0 ns |
| `biquad_cascade_filter` | one sample through a 3-section notch at 80/160/240 Hz | 3.7 ns | 3.7 ns |
| `multi_channel_biquad_filter` | one 3-axis sample through a 50 Hz low-pass | 3.1 ns | 3.1 ns |
| `follow_the_gap` | 61-beam scan, widest gap the robot fits through + speed ramp | 343.2 ns | 344.5 ns |

<!-- END generated -->
