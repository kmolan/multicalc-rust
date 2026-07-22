# Latency benchmarks

Latency measurements with criterion in
the optimized bench profile. **Numbers are machine-dependent**.
Manually regenerate with `cargo bench -p multicalc-qa`.

Measured on a 12th Gen Intel Core i7-12650H (16 threads) under WSL2 (Linux 6.18).

<!-- BEGIN generated: latency -->
| Operation | Equation | Median | Mean |
|-----------|----------|-------:|-----:|
| `derivative` | d³/dx³(x²·sin x) at x = 1 | 31.5 ns | 31.8 ns |
| `jacobian_small` | Jacobian of (x·y·z, x²+y²) | 2.0 ns | 2.0 ns |
| `jacobian_large` | Jacobian of a 6-in/6-out map | 288.2 ns | 289.2 ns |
| `gauss_quad` | ∫₀¹ (sin x − √x)·e⁻ˣ dx | 131.1 ns | 131.6 ns |
| `lu_solve` | solve A·x = b (10×10) | 274.6 ns | 275.1 ns |
| `svd_solve` | least-squares fit (30×3) | 933.1 ns | 936.6 ns |
| `expm` | matrix exponential eᴬ (6×6) | 735.8 ns | 738.4 ns |
| `rk45_solve` | y″ = −y, adaptive to 2π | 19.02 µs | 19.12 µs |
| `rk4_integrate` | y″ = −y, fixed-step to 2π | 18.86 µs | 18.93 µs |
| `lev_marq` | fit y = a·eᵇᵗ (8 points) | 2.27 µs | 2.28 µs |
| `newton_system` | x²+y² = 4, x·y = 1 | 311.9 ns | 312.7 ns |

<!-- END generated -->
