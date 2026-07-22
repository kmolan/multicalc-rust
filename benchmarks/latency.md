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
| `jacobian_large` | Jacobian of a 6-in/6-out map | 261.0 ns | 262.3 ns |
| `gauss_quad` | ∫₀¹ (sin x − √x)·e⁻ˣ dx | 123.4 ns | 122.1 ns |
| `lu_solve` | solve A·x = b (10×10) | 242.5 ns | 243.5 ns |
| `svd_solve` | least-squares fit (30×3) | 822.2 ns | 825.4 ns |
| `expm` | matrix exponential eᴬ (6×6) | 692.9 ns | 699.6 ns |
| `rk45_solve` | y″ = −y, adaptive to 2π | 16.65 µs | 16.69 µs |
| `rk4_integrate` | y″ = −y, fixed-step to 2π | 16.53 µs | 16.71 µs |
| `lev_marq` | fit y = a·eᵇᵗ (8 points) | 2.06 µs | 2.06 µs |
| `newton_system` | x²+y² = 4, x·y = 1 | 274.6 ns | 275.1 ns |
| `particle_filter` | 1000 particles, diff-drive motion + process noise + systematic resample | 95.10 µs | 95.13 µs |

<!-- END generated -->
