# Linear algebra

Fixed-size, stack-allocated `Matrix` and `Vector`. Dimensions are const generics, so a shape
mismatch is a compile error and nothing is heap-allocated. Indexing (`v[i]`, `m[(r, c)]`) is
the ergonomic path and panics on out-of-range like `Vec`. Use `get` / `get_mut` /
`try_row` / `try_column` when you want `Option` instead of a panic (panicking `row` /
`column` were removed).

- `Matrix::lu` → `Lu`: partial-pivoting Doolittle LU; `solve`, `determinant`, `inverse`.
- `Matrix::cholesky` → `Cholesky`: faster path for symmetric positive-definite matrices.
- `PivotedQr`: column-pivoted Householder QR; `solve_least_squares`.
- `Matrix::svd` → `Svd`: one-sided Jacobi SVD; `singular_values`, `condition_number`,
  `pseudo_inverse`, minimum-norm `solve`.
- `Matrix::symmetric_eigendecomposition` → `SymmetricEigendecomposition`: Jacobi rotations for a
  symmetric matrix; `eigenvalues` (largest first), `eigenvectors`, `determinant`,
  `condition_number`, `is_positive_definite`, and `clamped` for raising a drifted spectrum back
  above zero.

Direct linear solves via LU and Cholesky:

```rust
use multicalc::{Matrix2D, Matrix3D, Vector};

// Solve A·x = b.
let a = Matrix3D::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]);
let b = Vector::new([7.0, 19.0, 49.0]);
let x = a.solve(b).unwrap();                        // [1, 2, 3]

let lu = a.lu().unwrap();
let det = lu.determinant();
let inv = lu.inverse();

// A symmetric positive-definite matrix has a faster Cholesky path.
let s = Matrix2D::new([[4.0, 2.0], [2.0, 3.0]]);
let s_inv = s.cholesky().unwrap().inverse();
```

The singular value decomposition (one-sided Jacobi) gives the pseudo-inverse, minimum-norm
least-squares solve, rank, and condition number for any shape:

```rust
use multicalc::{Matrix, Vector};

// Thin SVD of a tall matrix: A = U · diag(σ) · Vᵀ.
let a = Matrix::<3, 2>::new([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
let svd = a.svd().unwrap();
let sigma = svd.singular_values();          // descending, non-negative
let cond = svd.condition_number();          // σ_max / σ_min

// Moore-Penrose pseudo-inverse: tall, square, or wide (M < N) inputs.
let a_pinv = a.pseudo_inverse().unwrap();

// Minimum-norm least-squares solve of A·x = b, without forming A⁺.
let x = svd.solve(Vector::new([1.0, 2.0, 3.0]));
```

A symmetric matrix has real eigenvalues and orthonormal directions, found by rotating away the
off-diagonal entries a pair at a time:

```rust
use multicalc::Matrix2D;

// This matrix has eigenvalues 3 and -1.
let a: Matrix2D = Matrix2D::new([[1.0, 2.0], [2.0, 1.0]]);
let decomposition = a.symmetric_eigendecomposition().unwrap();

let values = decomposition.eigenvalues();       // largest first
let vectors = decomposition.eigenvectors();     // one direction per column
assert!((decomposition.condition_number() - 3.0).abs() < 1e-12);

// V · diag(λ) · Vᵀ rebuilds the matrix.
let rebuilt: Matrix2D = Matrix2D::from_fn(|row, column| {
    (0..2).map(|k| vectors[(row, k)] * values[k] * vectors[(column, k)]).sum()
});
assert!((rebuilt[(0, 0)] - a[(0, 0)]).abs() < 1e-12);

// Raising every eigenvalue to a floor and rebuilding turns a covariance that has
// drifted below zero back into one a filter can keep using.
let repaired = decomposition.clamped(0.5);
assert!((repaired[(0, 0)] - 1.75).abs() < 1e-12);
```

For an overdetermined linear least-squares fit, use the column-pivoted QR directly:

```rust
use multicalc::linear_algebra::PivotedQr;
use multicalc::{Matrix, Vector};

// Least-squares fit of y = a + b*t through (0, 1), (1, 3), (2, 5): a = 1, b = 2.
let a = Matrix::<3, 2>::new([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]);
let b = Vector::new([1.0, 3.0, 5.0]);
let x = PivotedQr::decompose(a).unwrap().solve_least_squares(b).unwrap();
```

Two matrix equations round out the module, both solved by iterations that double their reach each
pass. `solve_discrete_riccati` finds the steady-state cost-to-go behind an optimal linear feedback
law, and puts its answer back into the equation before returning it. `solve_discrete_lyapunov`
solves `Aᵀ·P·A − P + Q = 0`, which is how a closed loop is shown to settle: an answer exists only
when repeated application of `A` shrinks every direction, so not finding one is the verdict rather
than a failure. Both are `O(n³)` per pass on a fixed budget, and both belong at design time.

Errors: factorizations and solves return [`LinalgError`](error-handling.md): `Singular`,
`NotPositiveDefinite`, `Underdetermined` (a least-squares system with `M < N`), `NotSymmetric` (a
matrix that does not read the same across the diagonal), `NonFinite`, or `DidNotConverge` (a
matrix-equation solver that ran out of its budget).

Credits: the QR factorization, damped solve, and overflow-safe norm port MINPACK's `qrfac`,
`qrsolv`, and `enorm` (Moré, Garbow, Hillstrom; public domain, netlib). LU and Cholesky follow
the standard Doolittle and Cholesky–Banachiewicz algorithms; the SVD follows Golub & Van Loan,
*Matrix Computations*, and Demmel & Veselić for high relative accuracy; the symmetric
eigendecomposition follows the Jacobi method in the same Golub & Van Loan. Full demos:
[linear_algebra.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/linear_algebra.rs)
and
[svd.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/svd.rs).


---

[Back to the tutorial index](README.md)
