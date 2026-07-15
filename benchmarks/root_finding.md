# Root-finding accuracy

The scalar and system root finders are tested against scipy (`scipy.optimize`). Each row solves an
equation and checks the root against scipy within the listed tolerance.

<!-- BEGIN generated: accuracy -->
| Operation | Equation | Tolerance | Tested Against |
| --- | --- | --- | --- |
| Bisection on [1,10] | −5 + x + 5e^{-x} = 0 | 1e-9 | SciPy/MINPACK 1.14.1 |
| Newton from x=5 | −5 + x + 5e^{-x} = 0 | 1e-9 | SciPy/MINPACK 1.14.1 |
| Bisection on [0,π] | E − e·sin E − M = 0 (e=0.8) | 1e-9 | SciPy/MINPACK 1.14.1 |
| Newton | E − e·sin E − M = 0 (e=0.8) | 1e-9 | SciPy/MINPACK 1.14.1 |
| Newton | 1/√f + 2·log₁₀(ε/3.7 + 2.51/(Re√f)) = 0 | 1e-9 | SciPy/MINPACK 1.14.1 |
| Damped Newton | x/√(1+x²) = 0 | 1e-9 | SciPy/MINPACK 1.14.1 |
| Newton system, 2×2 | two-link IK: tip at (pₓ, pᵧ) | 1e-9 | SciPy/MINPACK 1.14.1 |
| Newton system, 2×2 | [x²+y²−4, x·y−1] = 0 | 1e-9 | SciPy/MINPACK 1.14.1 |
| Newton system, 3×3 | [x+y+z−1, y−1.25x², z−5x·y] = 0 | 1e-8 | SciPy/MINPACK 1.14.1 |
<!-- END generated -->
