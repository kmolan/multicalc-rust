# Latency benchmarks

Latency measurements with criterion in
the optimized bench profile. **Numbers are machine-dependent**.
Manually regenerate with `cargo bench -p multicalc-qa`.

Measured on a 12th Gen Intel Core i7-12650H (16 threads) under WSL2 (Linux 6.18).

<!-- BEGIN generated: latency -->
| Operation | Equation | Median | Mean |
|-----------|----------|-------:|-----:|
| `derivative` | d³/dx³(x²·sin x) at x = 1 | 27.0 ns | 27.0 ns |
| `jacobian_small` | Jacobian of (x·y·z, x²+y²) | 1.8 ns | 1.8 ns |
| `jacobian_large` | Jacobian of a 6-in/6-out map | 270.1 ns | 270.4 ns |
| `gauss_quad` | ∫₀¹ (sin x − √x)·e⁻ˣ dx | 128.7 ns | 127.9 ns |
| `lu_solve` | solve A·x = b (10×10) | 250.6 ns | 250.6 ns |
| `svd_solve` | least-squares fit (30×3) | 819.3 ns | 821.0 ns |
| `expm` | matrix exponential eᴬ (6×6) | 668.0 ns | 669.4 ns |
| `rk45_solve` | y″ = −y, adaptive to 2π | 16.65 µs | 16.69 µs |
| `rk4_integrate` | y″ = −y, fixed-step to 2π | 16.47 µs | 16.54 µs |
| `lev_marq` | fit y = a·eᵇᵗ (8 points) | 2.01 µs | 2.01 µs |
| `newton_system` | x²+y² = 4, x·y = 1 | 271.4 ns | 272.2 ns |
| `particle_filter` | 1000 particles, diff-drive motion + process noise + systematic resample | 88.49 µs | 88.69 µs |
| `kalman_filter_step` | 2-state constant velocity, predict + update | 33.7 ns | 33.9 ns |
| `extended_kalman_filter_step` | 5-state coordinated turn + position fix, autodiff Jacobians, predict + update | 505.1 ns | 505.6 ns |
| `pid_update` | one PID tick with anti-windup and output limits | 3.5 ns | 3.5 ns |
| `pure_pursuit` | κ = 2·sin(α)/L_d toward a lookahead point | 11.2 ns | 11.2 ns |
| `biquad_filter` | one 2nd-order low-pass sample, 50 Hz at 1 kHz | 2.5 ns | 2.5 ns |
| `biquad_cascade_filter` | one sample through a 3-section notch at 80/160/240 Hz | 3.2 ns | 3.2 ns |
| `multi_channel_biquad_filter` | one 3-axis sample through a 50 Hz low-pass | 2.8 ns | 2.8 ns |
| `follow_the_gap` | 61-beam scan, widest gap the robot fits through + speed ramp | 287.9 ns | 286.2 ns |

<!-- END generated -->
