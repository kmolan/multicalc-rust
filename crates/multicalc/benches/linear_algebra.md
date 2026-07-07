# Linear algebra benchmarks

Results for the [`linear_algebra`](linear_algebra.rs) suite
(`cargo bench -- linear_algebra`). The accuracy tables report approximation error
(reconstruction, solve residual, identity, and Moore–Penrose conditions); the latency tables
report wall-clock medians on the machine noted in [README.md](README.md).

## Accuracy

Measured by the [`linear_algebra`](../examples/linear_algebra.rs) and [`svd`](../examples/svd.rs)
stress-test examples on well- and ill-conditioned inputs. Unlike latency, these are deterministic
numerical errors and reproduce on any machine. Reconstruction is the entrywise error of the
factorization; the solve residual is $$\lVert Ax - b\rVert$$ for a known solution.

### LU and Cholesky (decompose + solve)

| Problem              | Reconstruction | Solve residual | Notes                                                    |
| -------------------- | -------------- | -------------- | -------------------------------------------------------- |
| LU, general 4×4      | 9e-16          | 4e-15          | Well-conditioned; reconstruction is $$\lVert PA - LU\rVert$$ |
| LU, general 8×8      | 2e-15          | 3e-14          | Well-conditioned                                         |
| LU, Hilbert 6×6      | 3e-17          | 1e-15          | Ill-conditioned, yet backward-stable                     |
| Cholesky, SPD 4×4    | 2e-15          | 4e-15          | Reconstruction is $$\lVert A - LL^T\rVert$$; matches the LU solve to 9e-16 |
| Cholesky, SPD 8×8    | 2e-15          | 3e-14          | Matches the LU solve to 2e-15                            |
| Cholesky, Hilbert 6×6 | 1e-17         | 1e-15          | Ill-conditioned; the solve drifts from LU by 9e-11 as the condition number grows |

### Direct 4×4 inverse

| Problem                        | Identity error | Notes                                          |
| ------------------------------ | -------------- | ---------------------------------------------- |
| Direct 4×4 inverse, general    | 2e-16          | $$\lVert A\,A^{-1} - I\rVert$$, well-conditioned |
| Direct 4×4 inverse, Hilbert    | 6e-12          | Ill-conditioned; the error tracks the condition number |

### SVD and pseudo-inverse

| Problem                          | Approximation error | Notes                                                          |
| -------------------------------- | ------------------- | -------------------------------------------------------------- |
| Kabsch rotation recovery, 3×3    | 3e-16               | $$\lVert \hat{R} - R\rVert$$; orthogonality $$\lVert \hat{R}^T\hat{R} - I\rVert$$ = 7e-16 |
| Redundant-arm pseudo-inverse, 6×7 | 2e-15              | Moore–Penrose $$\lVert JJ^+J - J\rVert$$; symmetry of $$JJ^+$$ = 5e-16 |
| Near-singular Jacobian, 6×6      | —                   | cond ≈ 5e7; truncates to rank 5 and the pseudo-inverse stays bounded ($$\lVert J^+\rVert_{max}$$ ≈ 1) |
| Overdetermined plane fit, 30×3   | 7e-16               | Agrees with the normal equations; the 4e-3 fit residual is the injected 1e-3 noise |

## Latency

Median of criterion's estimate; wall-clock and therefore machine- and build-specific (see the
[environment note](README.md#environment)).

### QR

Operation                                                        | Median time |
---------------------------------------------------------------- | ----------- |
Column-pivoted QR, 8×8 decomposition (Hilbert)                   | 0.56 µs     |
QR least-squares solve, 20×7 (Vandermonde fit)                   | 0.89 µs     |
Damped (ridge) solve, 15×8                                       | 1.4 µs      |

### LU, Cholesky, direct inverse

Operation                                                        | Median time |
---------------------------------------------------------------- | ----------- |
Direct 4×4 inverse (cofactor / adjugate)                         | 31 ns       |
LU decompose, 4×4 (Doolittle, partial pivot)                     | 46 ns       |
LU decompose, 8×8                                                | 145 ns      |
LU decompose + solve, 8×8                                        | 184 ns      |
Cholesky decompose, 4×4 (SPD)                                    | 16 ns       |
Cholesky decompose, 8×8                                          | 109 ns      |
Cholesky decompose + solve, 8×8                                  | 184 ns      |

### SVD, pseudo-inverse

Operation                                                        | Median time |
---------------------------------------------------------------- | ----------- |
One-sided Jacobi SVD, 3×3 decomposition (Kabsch shape)           | 0.34 µs     |
One-sided Jacobi SVD, 6×6 decomposition                          | 1.7 µs      |
One-sided Jacobi SVD, 8×8 decomposition                          | 4.8 µs      |
One-sided Jacobi SVD, 12×6 decomposition (tall)                  | 1.2 µs      |
Pseudo-inverse, 6×6                                              | 2.0 µs      |
Pseudo-inverse, 6×7 (redundant / wide path)                      | 2.1 µs      |
