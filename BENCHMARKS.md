- [1. Derivatives for single variable equations](#1-derivatives-for-single-variable-equations)
- [2. Derivatives for single variable equations](#1-derivatives-for-single-variable-equations)
- [3. Iterative integration methods](#3-iterative-integration-methods)

## 1. Derivatives for single variable equations
Method | Derivative                             | Approximation error |           Notes                                |
------ | -------------------------------------- | ------------------- | ---------------------------------------------- |
Forward|                  |                     |                                                |
| Simpsons        | $$\int_a^b f(x) \mathrm{d}x$$                              |
| Trapezoidal     | $$\int_a^b f(x) \mathrm{d}x$$                              |
| GaussLegendre   | $$\int_a^b f(x) \mathrm{d}x$$                              |
| GaussLaguerre   | $$\int_{0}^\infty f(x) e^{-x} \mathrm{d}x$$                |
| GaussHermite    | $$\int_{-\infty}^\infty f(x) e^{-x^2} \mathrm{d}x$$        |


## 3. Iterative integration methods
Booles:

Integrand                              | Approximation error |           Notes                                         |
-------------------------------------- | ------------------- | ------------------------------------------------------- |
$$\int_0^2 2x \mathrm{d}x$$            |     1e-14           | Trivial Integration to showcase accuracy levels         |
$$\int_0^1 2x + yz \mathrm{d}x$$       | 1e-30               |                                                         |
$$\int_0^1\int_0^1\int_0^1 yz x^2 e^x \mathrm{d}x\mathrm{d}x\mathrm{d}x$$ | 0.0002 | Can handle integration by parts easily|
$$\int_0^1\int_0^1 x\over\sqrt{x^2 + y^2}\mathrm{d}x$$ |   | 0.0002    | Accuracy falls for more complex equations |
$$\int_0^1\int_0^1 Sin(x) + ye^z \mathrm{d}x\mathrm{d}y$$ | 0.08  | Struggles for overly complex equations |