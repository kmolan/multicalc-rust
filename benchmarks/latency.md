# Latency benchmarks

Latency measurements with criterion in
the optimized bench profile. **Numbers are machine-dependent**.
Manually regenerate with `cargo bench -p multicalc-qa`.

Measured on a 12th Gen Intel Core i7-12650H (16 threads) under WSL2 (Linux 6.18).

<!-- BEGIN generated: latency -->
| Operation | Equation | Median | Mean |
|-----------|----------|-------:|-----:|
| `derivative` | d³/dx³(x²·sin x) at x = 1 | 31.8 ns | 32.2 ns |
| `jacobian_small` | Jacobian of (x·y·z, x²+y²) | 2.1 ns | 2.1 ns |
| `jacobian_large` | Jacobian of a 6-in/6-out map | 338.2 ns | 341.0 ns |
| `gauss_quad` | ∫₀¹ (sin x − √x)·e⁻ˣ dx | 140.0 ns | 140.0 ns |
| `lu_solve` | solve A·x = b (10×10) | 280.1 ns | 281.1 ns |
| `svd_solve` | least-squares fit (30×3) | 962.5 ns | 961.6 ns |
| `expm` | matrix exponential eᴬ (6×6) | 762.7 ns | 766.4 ns |
| `rk45_solve` | y″ = −y, adaptive to 2π | 20.00 µs | 20.16 µs |
| `rk4_integrate` | y″ = −y, fixed-step to 2π | 19.29 µs | 19.53 µs |
| `lev_marq` | fit y = a·eᵇᵗ (8 points) | 2.36 µs | 2.36 µs |
| `newton_system` | x²+y² = 4, x·y = 1 | 324.3 ns | 325.3 ns |
| `particle_filter` | 1000 particles, diff-drive motion + process noise + systematic resample | 101.22 µs | 101.10 µs |
| `kalman_filter_step` | 2-state constant velocity, predict + update | 39.8 ns | 40.5 ns |
| `extended_kalman_filter_step` | 5-state coordinated turn + position fix, autodiff Jacobians, predict + update | 609.0 ns | 614.0 ns |
| `pid_update` | one PID tick with anti-windup and output limits | 3.7 ns | 3.7 ns |
| `pure_pursuit` | κ = 2·sin(α)/L_d toward a lookahead point | 13.2 ns | 13.3 ns |
| `biquad_filter` | one 2nd-order low-pass sample, 50 Hz at 1 kHz | 3.0 ns | 3.0 ns |
| `biquad_cascade_filter` | one sample through a 3-section notch at 80/160/240 Hz | 3.7 ns | 3.7 ns |
| `multi_channel_biquad_filter` | one 3-axis sample through a 50 Hz low-pass | 3.1 ns | 3.1 ns |
| `moving_average_filter` | one sample through a 16-wide mean | 2.2 ns | 2.3 ns |
| `running_median_filter` | one sample through a 9-wide median | 4.6 ns | 4.6 ns |
| `savitzky_golay_filter` | one sample through an 11-wide curve fit, value + slope + bend | 7.2 ns | 7.2 ns |
| `follow_the_gap` | 61-beam scan, widest gap the robot fits through + speed ramp | 307.9 ns | 308.5 ns |

<!-- END generated -->
