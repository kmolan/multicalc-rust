- [1. Single variable Differentiation](#1-single-variable-differentiation)
- [2. Multi variable Differentiation](#2-multi-variable-differentiation)
- [3. Iterative integration methods](#3-iterative-integration-methods)
- [4. Gaussian Quadrature methods](#4-gaussian-quadrature-methods)
- [5. Latency](#5-latency)

Sections 1–4 report **accuracy** (approximation error). Section 5 reports **latency** (wall-clock time per call).

## 1. Single variable Differentiation

Approximation errors evaluated at x = 1 (central difference, default step size).

| Derivative                                     | Approximation error | Notes                                                       |
| ---------------------------------------------- | ------------------- | ----------------------------------------------------------- |
| $$\mathrm{d}(Sin(x))\over\mathrm{d}x$$         | 1e-11               | Trivial case to showcase accuracy levels                    |
| $$\mathrm{d}(x^2 Sin(x))\over\mathrm{d}x$$     | 4e-11               | Can easily handle product rule with high accuracy           |
| $$\mathrm{d^2}(x^2 Sin(x))\over\mathrm{d}x^2$$ | 1e-7                | Approximation errors increase with higher order derivatives |
| $$\mathrm{d^3}(x^2 Sin(x))\over\mathrm{d}x^3$$ | 5e-5                | Approximation errors increase with higher order derivatives |

## 2. Multi variable Differentiation

Approximation errors evaluated at (x, y, z) = (1, 2, 3) (central difference, default step size).

| Derivative                                                               | Approximation error | Notes                                                       |
| ------------------------------------------------------------------------ | ------------------- | ----------------------------------------------------------- |
| $$\mathrm{d^2}(x + y + z)\over\mathrm{d}x\mathrm{d}y$$                   | <1e-15              | Trivial case to showcase accuracy levels                    |
| $$\mathrm{d^2}(ySin(x) + xCos(y) + xye^z)\over\mathrm{d}x^2$$            | 2e-7                | Can easily handle complex equations with high accuracy      |
| $$\mathrm{d^2}(ySin(x) + xCos(y) + xye^z)\over\mathrm{d}x\mathrm{d}y$$   | 2e-6                | Approximation errors increase for mixed derivatives         |
| $$\mathrm{d^3}(ySin(x) + xCos(y) + xye^z)\over\mathrm{d}x^2\mathrm{d}y$$ | 3e-3                | Approximation errors increase with higher order derivatives |


## 3. Iterative integration methods

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


## 4. Gaussian Quadrature methods

With gaussian quadratures, there is no one 'objective' better answer. Each quadrature rule is designed to solve a specific integrand type. For most integrands with finite limits, Gauss-Legendre is the most suitable choice. For infinite limits, Gauss-Hermite or Gauss-Laguerre is a better fit. However, all these models are only suitable for polynomial equations. For non-polynomial equations, their performance falls very fast.

Measured at order 5. Gauss-Hermite and Gauss-Laguerre take the **bare** integrand f(x): the tabulated weights already carry the $$e^{-x^2}$$ / $$e^{-x}$$ weighting function, so you pass only the polynomial factor (e.g. $$x^2$$, not $$x^2 e^{-x^2}$$). The integrands below are written in full weighted form.

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

## 5. Latency

Measured with the `criterion` suite in [`benches/calculus.rs`](benches/calculus.rs) (`cargo bench`). Each
figure is the median of criterion's estimate. These are wall-clock numbers and therefore machine- and
build-specific — treat the **relative** costs and scaling as the signal, not the absolute nanoseconds.
Regressions are guarded automatically by the deterministic work-count `#[test]`s (run under `cargo test`),
not by these timings.

**Environment:** 12th Gen Intel Core i7-12650H · `rustc` 1.95.0 (release, `opt-level = 3`) · criterion 0.5 ·
WSL2 (Ubuntu). Iterative integrals use the default **120** intervals; Gaussian quadrature uses the listed order.

### Differentiation

Operation                                                        | Median time |
---------------------------------------------------------------- | ----------- |
single-variable, 1st derivative                                  | 1.6 ns      |
single-variable, 2nd derivative                                  | 4.5 ns      |
single-variable, 3rd derivative                                  | 16 ns       |
multi-variable, single partial $$\partial/\partial x$$           | 31 ns       |
multi-variable, mixed partial $$\partial^2/\partial x\partial y$$ | 74 ns       |

Cost grows steeply with derivative order — each level doubles the function evaluations of repeated
finite differencing (as documented in `finite_difference.rs`).

### Iterative integration (120 intervals)

Operation                                                        | Median time |
---------------------------------------------------------------- | ----------- |
Boole, single integral, finite limits                            | 81 ns       |
Simpson 3/8, single integral, finite limits                      | 88 ns       |
Trapezoidal, single integral, finite limits                      | 74 ns       |
Boole, double-fold single-variable                               | 127 ns      |
Boole, $$e^{-x^2}$$ over a **finite** limit $$[-5, 5]$$          | 0.49 µs     |
Boole, $$e^{-x^2}$$ over an **infinite** limit $$(-\infty,\infty)$$ | 2.2 µs   |

The finite/infinite pair (same integrand, same rule) is the **finite fast path** in action: the
domain transform — `libm::tan`/`cos` per node — is paid only on the infinite domain (~4.4×), while
finite integrals are byte-for-byte the pre-transform path.

### Gaussian quadrature

Operation                                                        | Median time |
---------------------------------------------------------------- | ----------- |
Gauss-Legendre, order 4                                          | 6.4 ns      |
Gauss-Legendre, order 16                                         | 17 ns       |
Gauss-Hermite, order 5                                           | 4.4 ns      |
Gauss-Laguerre, order 5                                          | 4.4 ns      |

Roughly linear in `order` (≈0.8 ns/node) — the flat `(weight, abscissa)` table is an O(1) lookup with
no per-sample `match` or `unwrap`. For polynomial integrands, Gauss reaches full accuracy in a handful
of nodes, i.e. ~10–20× cheaper than the 120-interval iterative rules.

### Jacobian, Hessian & vector field

Operation                                                        | Median time |
---------------------------------------------------------------- | ----------- |
Jacobian, 2 functions × 3 variables                              | 37 ns       |
Hessian, 3 variables                                             | 0.37 µs     |
Curl, 3D                                                         | 20 ns       |
Divergence, 3D                                                   | 9.7 ns      |
Line integral, 2D (120 intervals)                                | 3.2 µs      |
Flux integral, 2D (120 intervals)                                | 3.2 µs      |

The 3-variable Hessian computes only 6 second derivatives (`N·(N+1)/2`), not 9, thanks to the
symmetric fill. Line/flux integrals are dominated by 120 `cos`/`sin` transform evaluations along the curve.

### Approximation

Operation                                                        | Median time |
---------------------------------------------------------------- | ----------- |
Linear approximation, build                                      | 2.9 ns      |
Linear approximation, `predict`                                  | 0.73 ns     |
Quadratic approximation, build                                   | 0.18 µs     |
Quadratic approximation, `predict`                               | 2.8 ns      |

`predict` is essentially free (sub-nanosecond for the linear form): the centered-Taylor representation
is a handful of fused multiply-adds with no allocation. The quadratic build cost is the Hessian.

