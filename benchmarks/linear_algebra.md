# Linear algebra accuracy

The linear-algebra routines are tested against numpy (LAPACK). Each case uses a random matrix `A`,
and for solves a random vector `b`, with `x` solving `Ax = b`. The table shows `det(A)` for the
LU/Cholesky solves, the residual `‖Ax − b‖` for QR least-squares, and the singular values `σ(A)`
for the SVD.

<!-- BEGIN generated: accuracy -->
| Operation | Equation | Tolerance | Tested Against |
| --- | --- | --- | --- |
| LU decompose + solve, 3×3 | det(A) | 1e-10 | numpy/LAPACK 2.1.3 |
| LU decompose + solve, 4×4 | det(A) | 1e-10 | numpy/LAPACK 2.1.3 |
| LU decompose + solve, 5×5 | det(A) | 1e-10 | numpy/LAPACK 2.1.3 |
| Cholesky decompose + solve, 2×2 | det(A) | 1e-10 | numpy/LAPACK 2.1.3 |
| Cholesky decompose + solve, 3×3 | det(A) | 1e-10 | numpy/LAPACK 2.1.3 |
| Cholesky decompose + solve, 4×4 | det(A) | 1e-10 | numpy/LAPACK 2.1.3 |
| QR least-squares, 3×2 | ‖Ax − b‖ | 1e-10 | numpy/LAPACK 2.1.3 |
| QR least-squares, 3×3 | ‖Ax − b‖ | 1e-10 | numpy/LAPACK 2.1.3 |
| QR least-squares, 4×3 | ‖Ax − b‖ | 1e-10 | numpy/LAPACK 2.1.3 |
| QR least-squares, 20×7 | ‖Ax − b‖ | 1e-10 | numpy/LAPACK 2.1.3 |
| SVD, 3×2 | σ(A) | 1e-10 | numpy/LAPACK 2.1.3 |
| SVD, 3×3 | σ(A) | 1e-10 | numpy/LAPACK 2.1.3 |
| SVD, 4×3 | σ(A) | 1e-10 | numpy/LAPACK 2.1.3 |
| SVD, 12×6 | σ(A) | 1e-10 | numpy/LAPACK 2.1.3 |
| SVD, 20×6 | σ(A) | 1e-10 | numpy/LAPACK 2.1.3 |
<!-- END generated -->
