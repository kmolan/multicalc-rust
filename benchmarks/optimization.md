# Optimization accuracy

The least-squares solvers are tested against scipy (MINPACK). Each row fits a problem and checks
the recovered minimizer against scipy within the listed tolerance.

<!-- BEGIN generated: accuracy -->
| Operation | Equation | Tolerance | Tested Against |
| --- | --- | --- | --- |
| Rosenbrock least-squares minimizer | min ‖[10(y − x²), 1 − x]‖² | 1e-6 | SciPy/MINPACK 1.14.1 |
| Geometric circle fit, 40 points | rᵢ = √((xᵢ − cₓ)² + (yᵢ − cᵧ)²) − r | 1e-6 | SciPy/MINPACK 1.14.1 |
| Two Gaussian peaks fit, 50 samples | rᵢ = Σₖ aₖ·e^(−((tᵢ − μₖ)/σₖ)²) − yᵢ | 1e-6 | SciPy/MINPACK 1.14.1 |
| Trigonometric least-squares, 6 vars | rᵢ = n − Σⱼcos xⱼ + i(1 − cos xᵢ) − sin xᵢ | 1e-6 | SciPy/MINPACK 1.14.1 |
<!-- END generated -->
