- [1. Single variable Differentiation](#1-single-variable-differentiation)
- [2. Multi variable Differentiation](#2-multi-variable-differentiation)
- [3. Iterative integration methods](#3-iterative-integration-methods)
- [4. Gaussian Quadrature methods](#4-gaussian-quadrature-methods)

## 1. Single variable Differentiation

Derivative                             | Approximation error |           Notes                                |
-------------------------------------- | ------------------- | ---------------------------------------------- |
$$\mathrm{d}(Sin(x))\over\mathrm{d}x$$ | 1e-12               | Trivial case to showcase accuracy levels       |
$$\mathrm{d}(x^2 Sin(x))\over\mathrm{d}x$$ | 1e-10 | Can easily handle product rule with high accuracy            |
$$\mathrm{d^2}(x^2 Sin(x))\over\mathrm{d}x^2$$ | 1e-7 | Approximation errors increase with higher order derivatives |
$$\mathrm{d^3}(x^2 Sin(x))\over\mathrm{d}x^3$$ | 7e-4| Approximation errors increase with higher order derivatives |

## 2. Multi variable Differentiation

Derivative                             | Approximation error |           Notes                                |
-------------------------------------- | ------------------- | ---------------------------------------------- |
$$\mathrm{d^2}(x + y + z)\over\mathrm{d}x\mathrm{d}y$$ | 1e-15 | Trivial case to showcase accuracy levels       |
$$\mathrm{d^2}(ySin(x) + xCos(y) + xye^z)\over\mathrm{d}x^2$$ | 1e-7 | Can easily handle complex equations with high accuracy |
$$\mathrm{d^2}(ySin(x) + xCos(y) + xye^z)\over\mathrm{d}x\mathrm{d}y$$ | 2e-6 | Approximation errors increase for mixed derivatives |
$$\mathrm{d^3}(ySin(x) + xCos(y) + xye^z)\over\mathrm{d}x^2\mathrm{d}y$$ | 7e-4| Approximation errors increase with higher order derivatives |


## 3. Iterative integration methods

Overall, it was found that in terms of accuracy, both Booles and Trapezoidal methods give the highest accuracy, but Booles is able to converge faster than trapezoidal, i.e. it needs fewer iterations and takes less compuatational resources. Simpsons method was found lacking in accuracy, and may be deprecated in future due to poor performance.

Booles:

Integrand                              | Approximation error |           Notes                                         |
-------------------------------------- | ------------------- | ------------------------------------------------------- |
$$\int_0^2 2x \mathrm{d}x$$            |     1e-14           | Trivial Integration to showcase accuracy levels         |
$$\int_0^1 (2x + yz) \mathrm{d}x$$     | 1e-30               | High accuracy for simple multivariable integrals        |
$$\int_0^1\int_0^1\int_0^1 (yz x^2 e^x) \mathrm{d}x\mathrm{d}x\mathrm{d}x$$ | 2e-3 | Can handle integration by parts easily|
$$\int_0^1\int_0^1 (x\over\sqrt{x^2 + y^2}) \mathrm{d}x$$ |  2e-3    | Accuracy falls for more complex equations |
$$\int_0^1\int_0^1 (Sin(x) + ye^z) \mathrm{d}x\mathrm{d}y$$ | 8e-2  | Struggles for overly complex equations |

Simpsons:

Integrand                              | Approximation error |           Notes                                         |
-------------------------------------- | ------------------- | ------------------------------------------------------- |
$$\int_0^2 2x \mathrm{d}x$$            | 5e-3                |                                                         |
$$\int_0^1 (2x + yz) \mathrm{d}x$$     | 5e-3                |                                                         |
$$\int_0^1\int_0^1\int_0^1 (yz x^2 e^x) \mathrm{d}x\mathrm{d}x\mathrm{d}x$$ | 2e-3 | Can handle integration by parts easily|
$$\int_0^1\int_0^1 (x\over\sqrt{x^2 + y^2}) \mathrm{d}x$$ |  2e-3    | Accuracy falls for more complex equations |
$$\int_0^1\int_0^1 (Sin(x) + ye^z) \mathrm{d}x\mathrm{d}y$$ | 8e-2  | Struggles for overly complex equations |


Trapezoidal:

Integrand                              | Approximation error |           Notes                                         |
-------------------------------------- | ------------------- | ------------------------------------------------------- |
$$\int_0^2 2x \mathrm{d}x$$            | 1e-14               |  Trivial Integration to showcase accuracy levels        |
$$\int_0^1 (2x + yz) \mathrm{d}x$$     | 1e-30               |   High accuracy for simple multivariable integrals      |
$$\int_0^1\int_0^1\int_0^1 (yz x^2 e^x) \mathrm{d}x\mathrm{d}x\mathrm{d}x$$ | 4e-4 | Can handle integration by parts|
$$\int_0^1\int_0^1 (x\over\sqrt{x^2 + y^2}) \mathrm{d}x$$ |  2e-1    | Struggles for complex equations  |
$$\int_0^1\int_0^1 (Sin(x) + ye^z) \mathrm{d}x\mathrm{d}y$$ | 8e-1  | Struggles for complex equations |


## 4. Gaussian Quadrature methods

With gaussian quadratures, there is no one 'objective' better answer. Each quadrature rule is designed to solve a specific integrand type. For most integrands with finite limits, Gauss-Legendre is the most suitable choice. For infinite limits, Gauss-Hermite or Gauss-Laguerre is a better fit. However, all these models are only suitable for polynomial equations. For non-polynomial equations, their performance falls very fast.

Gauss-Legendre

Integrand                              | Approximation error |           Notes                                         |
-------------------------------------- | ------------------- | ------------------------------------------------------- |
$$\int_0^2 4x^3 - 3x^2  \mathrm{d}x$$  | 1e-14               |  Trivial Integration to showcase accuracy levels        |
$$\int_0^1 (2x + yz) \mathrm{d}x$$     | 1e-30               |   High accuracy for simple multivariable integrals      |
$$\int_0^1\int_0^1 (x^3 y + y^3 z) \mathrm{d}x\mathrm{d}y$$ | 1e-30 | Can handle integration by parts easily|
$$\int_0^1\int_0^1\int_0^1 (x^3 y + y^3 z) \mathrm{d}x\mathrm{d}x\mathrm{d}y$$ | 1e-15 | High accuracy for higher order integrals|
$$\int_{0}^1 (Sin(x) - \sqrt{x})e^{-x} \mathrm{d}x$$ | 1e-2 | Poor performance for non-polynomial integrands |


Gauss-Laguerre

Integrand                              | Approximation error |           Notes                                         |
-------------------------------------- | ------------------- | ------------------------------------------------------- |
$$\int_{0}^\infty x^2 e^{-x} \mathrm{d}x$$   | 1e-30              |  Trivial Integration to showcase accuracy levels        |
$$\int_{0}^\infty (4x^3 - 3x^2)e^{-x} \mathrm{d}x$$  | 1e-12      |   High accuracy for more complicated integrands      |
$$\int_{0}^\infty\int_{0}^\infty\int_{0}^\infty (x^3 y + y^3 z)e^{-x} \mathrm{d}x\mathrm{d}x\mathrm{d}y$$ | 1e-9 | High accuracy for higher order integrals|
$$\int_{0}^\infty (Sin(x) - \sqrt{x})e^{-x} \mathrm{d}x$$ | 1e-2 | Poor performance for non-polynomial integrands |

Gauss-Hermite

Integrand                              | Approximation error |           Notes                                         |
-------------------------------------- | ------------------- | ------------------------------------------------------- |
$$\int_{-\infty}^\infty x^2 e^{-x^2} \mathrm{d}x$$   | 1e-30              |  Trivial Integration to showcase accuracy levels    |
$$\int_{-\infty}^\infty (4x^3 - 3x^2)e^{-x^2} \mathrm{d}x$$  | 1e-12      |   High accuracy for more complicated integrands      |
$$\int_{0}^\infty\int_{0}^\infty\int_{0}^\infty (x^3 y + y^3 z)e^{-x^2} \mathrm{d}x\mathrm{d}x\mathrm{d}y$$ | 1e-9 | High accuracy for higher order integrals|
$$\int_{-\infty}^\infty (Sin(x) - \sqrt{x})e^{-x^2} \mathrm{d}x$$ | 1e-1 | Poor performance for non-polynomial integrands |

