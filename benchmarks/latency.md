# Latency benchmarks

Latency measurements with criterion in
the optimized bench profile. **Numbers are machine-dependent**.
Manually regenerate with `cargo bench -p multicalc-qa`.

Measured on a 12th Gen Intel Core i7-12650H (16 threads) under WSL2 (Linux 6.18).

<!-- BEGIN generated: latency -->
| Operation | Equation | Median | Mean |
|-----------|----------|-------:|-----:|
| `derivative` | d³/dx³(x²·sin x) at x = 1 | 31.9 ns | 32.0 ns |
| `jacobian_small` | Jacobian of (x·y·z, x²+y²) | 2.1 ns | 2.1 ns |
| `jacobian_large` | Jacobian of a 6-in/6-out map | 307.9 ns | 307.8 ns |
| `gauss_quad` | ∫₀¹ (sin x − √x)·e⁻ˣ dx | 139.6 ns | 140.0 ns |
| `lu_solve` | solve A·x = b (10×10) | 280.8 ns | 283.2 ns |
| `svd_solve` | least-squares fit (30×3) | 966.3 ns | 967.7 ns |
| `expm` | matrix exponential eᴬ (6×6) | 770.9 ns | 772.1 ns |
| `rk45_solve` | y″ = −y, adaptive to 2π | 19.51 µs | 19.62 µs |
| `rk4_integrate` | y″ = −y, fixed-step to 2π | 19.36 µs | 19.48 µs |
| `lev_marq` | fit y = a·eᵇᵗ (8 points) | 2.36 µs | 2.37 µs |
| `newton_system` | x²+y² = 4, x·y = 1 | 320.4 ns | 320.4 ns |
| `particle_filter` | 1000 particles, diff-drive motion + process noise + systematic resample | 105.10 µs | 107.60 µs |
| `kalman_filter_step` | 2-state constant velocity, predict + update | 39.0 ns | 39.1 ns |
| `extended_kalman_filter_step` | 5-state coordinated turn + position fix, autodiff Jacobians, predict + update | 579.8 ns | 583.9 ns |
| `pid_update` | one PID tick with anti-windup and output limits | 3.8 ns | 3.8 ns |
| `pure_pursuit` | κ = 2·sin(α)/L_d toward a lookahead point | 13.1 ns | 13.2 ns |
| `biquad_filter` | one 2nd-order low-pass sample, 50 Hz at 1 kHz | 2.9 ns | 2.9 ns |
| `biquad_cascade_filter` | one sample through a 3-section notch at 80/160/240 Hz | 3.6 ns | 3.6 ns |
| `multi_channel_biquad_filter` | one 3-axis sample through a 50 Hz low-pass | 3.1 ns | 3.1 ns |
| `follow_the_gap` | 61-beam scan, widest gap the robot fits through + speed ramp | 319.7 ns | 326.0 ns |

<!-- END generated -->
