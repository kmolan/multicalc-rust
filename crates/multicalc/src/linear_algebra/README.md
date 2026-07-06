# linear_algebra

Fixed-size, stack-allocated [`Matrix`](matrix.rs) and [`Vector`](vector.rs): dimensions are const
generics (shape mismatches are compile errors), nothing is heap-allocated, and the math never panics.

- [`Matrix::lu`](lu.rs) → [`Lu`](lu.rs) — partial-pivoting Doolittle; `solve`, `determinant`,
  `inverse`.
- [`Matrix::cholesky`](cholesky.rs) → [`Cholesky`](cholesky.rs) — faster path for symmetric
  positive-definite matrices.
- [`PivotedQr`](qr.rs) — column-pivoted Householder QR; `solve_least_squares`.
- [`Matrix::svd`](svd.rs) → [`Svd`](svd.rs) — one-sided Jacobi SVD; `singular_values`,
  `condition_number`, `pseudo_inverse`, minimum-norm `solve`.

```rust
use multicalc::linear_algebra::{Matrix, Vector};

// Solve A·x = b.
let a = Matrix::<3, 3>::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]);
let b = Vector::new([7.0, 19.0, 49.0]);
let x = a.solve(b).unwrap();                        // [1, 2, 3]

// Thin SVD of a tall matrix, then its Moore–Penrose pseudo-inverse.
let m = Matrix::<3, 2>::new([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
let svd = m.svd().unwrap();
let cond = svd.condition_number();                 // σ_max / σ_min
let m_pinv = m.pseudo_inverse().unwrap();
```

Credits: the QR factorization, damped solve, and overflow-safe norm port MINPACK's `qrfac`, `qrsolv`,
and `enorm` (Moré, Garbow, Hillstrom; public domain, netlib). LU and Cholesky follow the standard
Doolittle and Cholesky–Banachiewicz algorithms; the SVD follows Golub & Van Loan, *Matrix
Computations*, and Demmel & Veselić for high relative accuracy.
