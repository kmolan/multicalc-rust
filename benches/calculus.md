# Calculus benchmarks

Results for the [`calculus`](calculus.rs) suite (`cargo bench --bench calculus`). The accuracy
tables report approximation error against the analytic value; the latency tables report
wall-clock medians on the machine noted in [README.md](README.md). See that index for how the
suites fit together and how to run them.

## Accuracy

### Single-variable differentiation

Differentiation uses forward-mode autodiff (the default backend). Errors below are measured against the analytic derivative at x = 1.

| Derivative                                     | Approximation Error  | Notes                                                          |
| ---------------------------------------------- | ------ | -------------------------------------------------------------- |
| $$\mathrm{d}(Sin(x))\over\mathrm{d}x$$         | 0.0      | Exact: the Dual carries cos(x), matching the closed form       |
| $$\mathrm{d}(x^2 Sin(x))\over\mathrm{d}x$$     | 0.0      | Product rule handled exactly                                   |
| $$\mathrm{d^2}(x^2 Sin(x))\over\mathrm{d}x^2$$ | 0.0      | Second order via HyperDual, still exact                        |
| $$\mathrm{d^3}(x^2 Sin(x))\over\mathrm{d}x^3$$ | 4e-16  | Third order via Jet; a single rounding ulp, order-independent  |

### Multi-variable differentiation

Partial derivatives also use autodiff: first via Dual, second via HyperDual, third via a nested `Dual<HyperDual>`. Errors below are measured against the analytic derivative at (x, y, z) = (1, 2, 3).

| Derivative                                                               | Approximation Error  | Notes                                              |
| ------------------------------------------------------------------------ | ------ | -------------------------------------------------- |
| $$\mathrm{d^2}(x + y + z)\over\mathrm{d}x\mathrm{d}y$$                   | 0.0      | Trivial linear case, exact                         |
| $$\mathrm{d^2}(ySin(x) + xCos(y) + xye^z)\over\mathrm{d}x^2$$            | 0.0      | Pure second partial, exact                         |
| $$\mathrm{d^2}(ySin(x) + xCos(y) + xye^z)\over\mathrm{d}x\mathrm{d}y$$   | 0.0      | Mixed second partial, exact                        |
| $$\mathrm{d^3}(ySin(x) + xCos(y) + xye^z)\over\mathrm{d}x^2\mathrm{d}y$$ | 0.0      | Third-order mixed partial, exact                   |

### Iterative integration

Measured at the default 120 intervals. Boole's rule (the highest-order method here) gives the best accuracy on every integrand; Simpson's 3/8 is intermediate; the trapezoidal rule is lowest order and trails on smooth integrands. The composite rules require the interval count to be a multiple of their panel width — 4 for Boole and 3 for Simpson's 3/8 — which the default of 120 satisfies.

Booles:

| Integrand                                                                   | Approximation error | Notes                                         |
| --------------------------------------------------------------------------- | ------------------- | --------------------------------------------- |
| $$\int_0^2 2x \mathrm{d}x$$                                                 | <1e-15              | Trivial integration, exact                    |
| $$\int_0^1 (2x + yz) \mathrm{d}x$$                                          | 1e-15               | Exact for simple multivariable integrals      |
| $$\int_0^1\int_0^1\int_0^1 (yz x^2 e^x) \mathrm{d}x\mathrm{d}x\mathrm{d}x$$ | 3e-13               | High accuracy for smooth multi-fold integrals |
| $$\int_0^1\int_0^1 (x\over\sqrt{x^2 + y^2}) \mathrm{d}x$$                   | 1e-15               | High accuracy even for more complex equations |
| $$\int_0^1\int_0^1 (Sin(x) + ye^z) \mathrm{d}x\mathrm{d}y$$                 | 2e-15               | High accuracy even for complex equations      |

Simpsons:

| Integrand                                                                   | Approximation error | Notes                                          |
| --------------------------------------------------------------------------- | ------------------- | ---------------------------------------------- |
| $$\int_0^2 2x \mathrm{d}x$$                                                 | 4e-16               | Exact (interval count must be a multiple of 3) |
| $$\int_0^1 (2x + yz) \mathrm{d}x$$                                          | <1e-15              | Exact for simple multivariable integrals       |
| $$\int_0^1\int_0^1\int_0^1 (yz x^2 e^x) \mathrm{d}x\mathrm{d}x\mathrm{d}x$$ | 1e-8                | High accuracy, slightly behind Boole           |
| $$\int_0^1\int_0^1 (x\over\sqrt{x^2 + y^2}) \mathrm{d}x$$                   | 2e-11               | High accuracy for more complex equations       |
| $$\int_0^1\int_0^1 (Sin(x) + ye^z) \mathrm{d}x\mathrm{d}y$$                 | 3e-11               | High accuracy for complex equations            |

Trapezoidal:

| Integrand                                                                   | Approximation error | Notes                                             |
| --------------------------------------------------------------------------- | ------------------- | ------------------------------------------------- |
| $$\int_0^2 2x \mathrm{d}x$$                                                 | 4e-16               | Trivial integration, exact                        |
| $$\int_0^1 (2x + yz) \mathrm{d}x$$                                          | <1e-15              | Exact for simple multivariable integrals          |
| $$\int_0^1\int_0^1\int_0^1 (yz x^2 e^x) \mathrm{d}x\mathrm{d}x\mathrm{d}x$$ | 3e-4                | Lowest order; accuracy falls on smooth integrands |
| $$\int_0^1\int_0^1 (x\over\sqrt{x^2 + y^2}) \mathrm{d}x$$                   | 8e-7                | Accuracy falls for more complex equations         |
| $$\int_0^1\int_0^1 (Sin(x) + ye^z) \mathrm{d}x\mathrm{d}y$$                 | 3e-6                | Accuracy falls for more complex equations         |

### Gaussian quadrature

With gaussian quadratures, there is no one 'objective' better answer. Each quadrature rule is designed to solve a specific integrand type. For most integrands with finite limits, Gauss-Legendre is the most suitable choice. For infinite limits, Gauss-Hermite or Gauss-Laguerre is a better fit. However, all these models are only suitable for polynomial equations. For non-polynomial equations, their performance falls very fast.

Gauss-Legendre

| Integrand                                                                      | Approximation error | Notes                                            |
| ------------------------------------------------------------------------------ | ------------------- | ------------------------------------------------ |
| $$\int_0^2 4x^3 - 3x^2  \mathrm{d}x$$                                          | 2e-15               | Trivial Integration to showcase accuracy levels  |
| $$\int_0^1 (2x + yz) \mathrm{d}x$$                                             | <1e-15              | High accuracy for simple multivariable integrals |
| $$\int_0^1\int_0^1 (x^3 y + y^3 z) \mathrm{d}x\mathrm{d}y$$                    | 2e-16               | Can handle integration by parts easily           |
| $$\int_0^1\int_0^1\int_0^1 (x^3 y + y^3 z) \mathrm{d}x\mathrm{d}x\mathrm{d}y$$ | 4e-16               | High accuracy for higher order integrals         |
| $$\int_{0}^1 (Sin(x) - \sqrt{x})e^{-x} \mathrm{d}x$$                           | 6e-4                | Poor performance for non-polynomial integrands   |

Gauss-Laguerre

| Integrand                                                                                                 | Approximation error | Notes                                           |
| --------------------------------------------------------------------------------------------------------- | ------------------- | ----------------------------------------------- |
| $$\int_{0}^\infty x^2 e^{-x} \mathrm{d}x$$                                                                | 1e-15               | Trivial Integration to showcase accuracy levels |
| $$\int_{0}^\infty (4x^3 - 3x^2)e^{-x} \mathrm{d}x$$                                                       | 2e-14               | High accuracy for more complicated integrands   |
| $$\int_{0}^\infty\int_{0}^\infty\int_{0}^\infty (x^3 y + y^3 z)e^{-x} \mathrm{d}x\mathrm{d}x\mathrm{d}y$$ | 2e-14               | High accuracy for higher order integrals        |
| $$\int_{0}^\infty (Sin(x) - \sqrt{x})e^{-x} \mathrm{d}x$$                                                 | 1e-2                | Poor performance for non-polynomial integrands  |

Gauss-Hermite

| Integrand                                                                                                                     | Approximation error | Notes                                           |
| ----------------------------------------------------------------------------------------------------------------------------- | ------------------- | ----------------------------------------------- |
| $$\int_{-\infty}^\infty x^2 e^{-x^2} \mathrm{d}x$$                                                                            | 1e-16               | Trivial Integration to showcase accuracy levels |
| $$\int_{-\infty}^\infty (4x^3 - 3x^2)e^{-x^2} \mathrm{d}x$$                                                                   | 4e-16               | High accuracy for more complicated integrands   |
| $$\int_{-\infty}^\infty\int_{-\infty}^\infty\int_{-\infty}^\infty (x^3 y + y^3 z)e^{-x^2} \mathrm{d}x\mathrm{d}x\mathrm{d}y$$ | <1e-15              | High accuracy for higher order integrals        |
| $$\int_{-\infty}^\infty (Sin(x) - \sqrt{x})e^{-x^2} \mathrm{d}x$$                                                             | undefined           | √x is not real at the negative abscissae        |

## Latency

Median of criterion's estimate; wall-clock and therefore machine- and build-specific (see the
[environment note](README.md#environment)). Iterative integrals use the default 120 intervals;
Gaussian quadrature uses the listed order.

### Differentiation

Operation                                                          | Median time |
------------------------------------------------------------------ | ----------- |
single-variable, 1st derivative (Dual)                             | 1.2 ns      |
single-variable, 2nd derivative (HyperDual)                        | 1.5 ns      |
single-variable, 3rd derivative (Jet)                              | 4.0 ns      |
multi-variable, single partial $$\partial/\partial x$$ (Dual)      | 34 ns       |
multi-variable, mixed partial $$\partial^2/\partial x\partial y$$ (HyperDual) | 46 ns |
multi-variable, mixed partial $$\partial^3/\partial x^2\partial y$$ (`Dual<HyperDual>`) | 79 ns |

### Iterative integration (120 intervals)

Operation                                                        | Median time |
---------------------------------------------------------------- | ----------- |
Boole, single integral, finite limits                            | 98 ns       |
Simpson 3/8, single integral, finite limits                      | 104 ns      |
Trapezoidal, single integral, finite limits                      | 86 ns       |
Boole, double-fold single-variable                               | 180 ns      |
Boole, $$e^{-x^2}$$ over a **finite** limit $$[-5, 5]$$          | 0.55 µs     |
Boole, $$e^{-x^2}$$ over an **infinite** limit $$(-\infty,\infty)$$ | 2.2 µs   |

### Gaussian quadrature

Operation                                                        | Median time |
---------------------------------------------------------------- | ----------- |
Gauss-Legendre, order 4                                          | 6.4 ns      |
Gauss-Legendre, order 16                                         | 17 ns       |
Gauss-Hermite, order 5                                           | 4.4 ns      |
Gauss-Laguerre, order 5                                          | 4.4 ns      |

### Jacobian, Hessian & vector field

Operation                                                        | Median time |
---------------------------------------------------------------- | ----------- |
Jacobian, 2 functions × 3 variables                              | 2.7 ns      |
Hessian, 3 variables                                             | 0.15 µs     |
Curl, 3D                                                         | 6.2 ns      |
Divergence, 3D                                                   | 0.37 ns     |
Line integral, 2D (120 intervals)                                | 3.3 µs      |
Flux integral, 2D (120 intervals)                                | 3.3 µs      |

### Approximation

Operation                                                        | Median time |
---------------------------------------------------------------- | ----------- |
Linear approximation, build                                      | 2.2 ns      |
Linear approximation, `predict`                                  | 0.72 ns     |
Quadratic approximation, build                                   | 11 ns       |
Quadratic approximation, `predict`                               | 3.3 ns      |
