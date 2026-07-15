# Optimization benchmarks

Results for the [`optimization`](optimization.rs) suite (`cargo bench -- optimization`).
The accuracy table reports how close each solver lands to the known solution; the latency table
reports wall-clock medians on the machine noted in [README.md](README.md).

## Accuracy

Reported below are the final **objective** (half the sum of squared residuals) and the largest **parameter error** across the recovered parameters versus the known truth. A "0.0" is an exact result to the last bit; a dash marks a pure-minimization problem with no reference parameter vector.

Levenberg-Marquardt:

| Problem                                        | Final objective | Parameter error | Notes                                                    |
| ---------------------------------------------- | --------------- | --------------- | -------------------------------------------------------- |
| Linear least squares, fit $$y = a + bt$$       | 0.0             | 0.0             | Linear residual: the exact least-squares solution        |
| Exponential decay, $$a\,e^{bt}$$ fit           | 0.0             | 0.0             | Zero-residual data; a and b recovered exactly            |
| Rosenbrock residual                            | 0.0             | 0.0             | Zero-residual; the minimum (1, 1) is reached exactly     |
| Damped sinusoids, 12 parameters                | 6e-31           | 2e-16           | All twelve parameters to ~1 ulp                          |
| Trigonometric (Moré-Garbow-Hillstrom), 6 vars  | 1e-31           | —               | Global minimum of zero reached to machine precision      |

Gauss-Newton:

| Problem                                        | Final objective | Parameter error | Notes                                                    |
| ---------------------------------------------- | --------------- | --------------- | -------------------------------------------------------- |
| Linear least squares, fit $$y = a + bt$$       | 0.0             | 0.0             | Reaches the exact solution in a single step              |
| Rosenbrock residual                            | 0.0             | 0.0             | Zero-residual; the minimum (1, 1) is reached exactly     |
| Geometric circle fit, 3 parameters             | 1e-30           | 0.0             | Center (2, -1) and radius 3 exact to machine precision   |
| Two Gaussian peaks, 6 parameters               | 0.0             | 0.0             | All six peak parameters recovered exactly                |

On these noiseless problems both solvers hit the minimizer to the last few bits; the only nonzero figures are the hardest fits (the 12-parameter sinusoid sum and the 3-parameter circle), where accumulated rounding leaves the objective around `1e-30` and the parameters within ~1 ulp. Accuracy here is bounded by floating-point rounding, not by the algorithm.

## Latency

Median of criterion's estimate; wall-clock and therefore machine- and build-specific (see the
[environment note](README.md#environment)).

Operation                                                        | Median time |
---------------------------------------------------------------- | ----------- |
Gauss-Newton, linear least squares (3×2)                         | 0.23 µs     |
Gauss-Newton, Rosenbrock (near start)                            | 0.26 µs     |
Gauss-Newton, circle fit (3 params, 40 residuals)               | 56 µs       |
Levenberg-Marquardt, exponential decay fit (2 params)            | 1.4 µs      |
Levenberg-Marquardt, Rosenbrock                                  | 6.2 µs      |
Levenberg-Marquardt, trigonometric (6 vars)                      | 45 µs       |
Levenberg-Marquardt, damped sinusoids (12 params, 60 residuals) | 14 ms       |
