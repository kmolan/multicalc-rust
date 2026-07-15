# Calculus accuracy

Differentiation, integration, and the vector-calculus operators are tested against mpmath:
derivatives against the closed-form analytic value, integrals against high-precision mpmath
quadrature.

<!-- BEGIN generated: accuracy -->
| Operation | Equation | Tolerance | Tested Against |
| --- | --- | --- | --- |
| 1st derivative at x=2 | x³ | 1e-12 | closed-form analytic (mpmath 1.3.0) |
| 2nd derivative at x=2 | x³ | 1e-12 | closed-form analytic (mpmath 1.3.0) |
| 3rd derivative at x=2 | x³ | 1e-12 | closed-form analytic (mpmath 1.3.0) |
| ∂/∂x at (1,2,3) | g = y·sin x + x·cos y + x·y·eᶻ | 1e-10 | closed-form analytic (mpmath 1.3.0) |
| ∂²/∂x∂y at (1,2,3) | g = y·sin x + x·cos y + x·y·eᶻ | 1e-10 | closed-form analytic (mpmath 1.3.0) |
| ∂³/∂x²∂y at (1,2,3) | g = y·sin x + x·cos y + x·y·eᶻ | 1e-10 | closed-form analytic (mpmath 1.3.0) |
| Jacobian at (1,2,3) | [x·y·z, x² + y²] | 1e-10 | closed-form analytic (mpmath 1.3.0) |
| Hessian at (1,2,3) | y·sin x + 2x·eʸ + z² | 1e-10 | closed-form analytic (mpmath 1.3.0) |
| Curl at (1,2,3) | [y, -x, 2z] | 1e-12 | closed-form analytic (mpmath 1.3.0) |
| Divergence at (1,2,3) | [y, -x, 2z] | 1e-12 | closed-form analytic (mpmath 1.3.0) |
| Line integral on the unit circle | [y, -x] | 5e-3 | closed-form analytic (mpmath 1.3.0) |
| Flux integral on the unit circle | [y, -x] | 5e-3 | closed-form analytic (mpmath 1.3.0) |
| Linear Taylor predict at (1.1,2.1,2.9) | x + y² + z³ | 1e-12 | closed-form analytic (mpmath 1.3.0) |
| Quadratic Taylor predict at (1.1,2.1,2.9) | x + y² + z³ | 1e-12 | closed-form analytic (mpmath 1.3.0) |
| Gauss-Legendre order 4 on [0,2] | ∫ 2x dx | 1e-12 | mpmath 1.3.0 |
| Gauss-Legendre order 4 on [0,2] | ∫ 4x³ − 3x² dx | 1e-12 | mpmath 1.3.0 |
| Gauss-Legendre order 16 on [0,2] | ∫ 4x³ − 3x² dx | 1e-12 | mpmath 1.3.0 |
| Boole's rule, 120 intervals on [0,2] | ∫ x³ dx | 1e-10 | mpmath 1.3.0 |
| Boole's rule, 120 intervals on [0,2] | ∫ 2x dx | 1e-10 | mpmath 1.3.0 |
| Simpson's rule, 120 intervals on [0,2] | ∫ 2x dx | 1e-10 | mpmath 1.3.0 |
| Trapezoidal, 120 intervals on [0,2] | ∫ 2x dx | 1e-10 | mpmath 1.3.0 |
| Boole's rule, 120 intervals on [-5,5] | ∫ e^{-x²} dx | 1e-6 | mpmath 1.3.0 |
| Gauss-Hermite order 5 on ℝ | ∫ x²e^{-x²} dx | 1e-10 | mpmath 1.3.0 |
| Gauss-Laguerre order 5 on [0,∞) | ∫ x²e^{-x} dx | 1e-10 | mpmath 1.3.0 |
| Trapezoidal, 2²⁰ intervals on [0,1] | ∫ 1/(1+x²) dx | 1e-3 | mpmath 1.3.0 |
<!-- END generated -->

The multi-fold (double/triple) integrals and non-polynomial Gauss-Legendre / Laguerre / Hermite
rows the module also exercises are illustrative and not yet fixture-backed: they need a
multi-dimensional QA harness and are pending that work.
