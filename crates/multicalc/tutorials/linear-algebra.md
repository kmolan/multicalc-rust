# Linear algebra

Fixed-size, stack-allocated `Matrix` and `Vector`. Dimensions are const generics, so a shape
mismatch is a compile error. Indexing (`v[i]`, `m[(r, c)]`) panics out of range, as `Vec` does.
`get`, `get_mut`, `try_row` and `try_column` return `Option` instead. A
[borrowed view](#borrowed-views) reshapes with no copy and no panic path.

| Entry point | Result | Method |
|---|---|---|
| `Matrix::lu_decompose` | `LuDecomposition`: `solve`, `determinant`, `inverse` | Doolittle LU, partial pivoting |
| `Matrix::cholesky` | `Cholesky`: `solve`, `determinant`, `inverse` | Cholesky–Banachiewicz, faster than LU; symmetric positive-definite only |
| `PivotedQr::decompose` | `PivotedQr`: `solve_least_squares` | Column-pivoted Householder QR |
| `Matrix::svd` | `Svd`: `singular_values`, `rank`, `numerical_rank`, `condition_number`, `pseudo_inverse`, minimum-norm `solve` | One-sided Jacobi, `M ≥ N` |
| `Matrix::symmetric_eigendecomposition` | `SymmetricEigendecomposition`: `eigenvalues` (largest first), `eigenvectors`, `determinant`, `condition_number`, `is_positive_definite`, `clamped` | Jacobi rotations |

## Borrowed views

`Matrix::view` / `view_mut` and `Vector::view` / `view_mut` return a `MatrixView` / `VectorView`
or the `Mut` form: a flat slice, an offset, and a stride per axis. Transpose, block, row, column,
diagonal and segment are index arithmetic. Only `to_matrix` / `to_vector` copies. The `Mut` views
add `try_get_mut`, `fill`, and `copy_from`.

Views have no `Index`. Each fallible call returns `Result<_, LinalgError>` with `OutOfBounds`, so
no view operation panics. The view must check the bound itself: on a strided view, an out-of-range
subscript can still land on a valid element of the parent buffer.

```rust
use multicalc::linear_algebra::{Matrix, MatrixViewMut};

let matrix = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

let block = matrix.view().try_submatrix::<2, 2>(0, 1).unwrap();
assert_eq!(block.to_matrix().into_array(), [[2.0, 3.0], [5.0, 6.0]]);

// A column has a stride of one row.
let column = matrix.view().try_column(1).unwrap();
assert_eq!(column.stride(), 3);
assert_eq!(column.dot(matrix.view().try_column(1).unwrap()), 29.0);

// Transpose into a caller buffer, with no intermediate matrix.
let mut scratch = [0.0; 6];
MatrixViewMut::<3, 2>::try_from_row_major_slice(&mut scratch)
    .unwrap()
    .copy_from(matrix.view().transposed());
assert_eq!(scratch, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

// Two disjoint mutable halves of one matrix.
let mut target = Matrix::<3, 2>::zeros();
let (mut top, mut bottom) = target.view_mut().try_split_rows_at::<1, 2>().unwrap();
*top.try_get_mut(0, 0).unwrap() = 1.0;
bottom.fill(2.0);
assert_eq!(target.into_array(), [[1.0, 0.0], [2.0, 2.0], [2.0, 2.0]]);
```

## Decompositions and solves

```rust
use multicalc::{Matrix2D, Matrix3D, Vector};

// Solve A·x = b.
let a = Matrix3D::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]);
let b = Vector::new([7.0, 19.0, 49.0]);
let x = a.solve(b).unwrap();                        // [1, 2, 3]

let factorization = a.lu_decompose().unwrap();
let det = factorization.determinant();
let inv = factorization.inverse();

// Symmetric positive-definite: Cholesky.
let s = Matrix2D::new([[4.0, 2.0], [2.0, 3.0]]);
let s_inv = s.cholesky().unwrap().inverse();
```

`Matrix::svd` accepts `M ≥ N` and returns `Underdetermined` for a wide matrix.
`Matrix::pseudo_inverse` accepts any shape and computes a wide input as `((Aᵀ)⁺)ᵀ`.

```rust
use multicalc::{Matrix, Vector};

// Thin SVD of a tall matrix: A = U · diag(σ) · Vᵀ.
let a = Matrix::<3, 2>::new([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
let svd = a.svd().unwrap();
let sigma = svd.singular_values();          // descending, non-negative
let cond = svd.condition_number();          // σ_max / σ_min
let rank = svd.numerical_rank();            // rank at the default cutoff max(M, N) · ε · σ_max

// Moore-Penrose pseudo-inverse.
let a_pinv = a.pseudo_inverse().unwrap();

// Minimum-norm least-squares solve of A·x = b, without forming A⁺.
let x = svd.solve(Vector::new([1.0, 2.0, 3.0]));
```

```rust
use multicalc::Matrix2D;

// Eigenvalues 3 and -1.
let a: Matrix2D = Matrix2D::new([[1.0, 2.0], [2.0, 1.0]]);
let decomposition = a.symmetric_eigendecomposition().unwrap();

let values = decomposition.eigenvalues();       // largest first
let vectors = decomposition.eigenvectors();     // one eigenvector per column
assert!((decomposition.condition_number() - 3.0).abs() < 1e-12);

// A = V · diag(λ) · Vᵀ.
let rebuilt: Matrix2D = Matrix2D::from_fn(|row, column| {
    (0..2).map(|k| vectors[(row, k)] * values[k] * vectors[(column, k)]).sum()
});
assert!((rebuilt[(0, 0)] - a[(0, 0)]).abs() < 1e-12);

// Clamp each eigenvalue to a floor and rebuild: repairs a covariance that lost definiteness.
let repaired = decomposition.clamped(0.5);
assert!((repaired[(0, 0)] - 1.75).abs() < 1e-12);
```

```rust
use multicalc::linear_algebra::PivotedQr;
use multicalc::{Matrix, Vector};

// Least-squares fit of y = a + b*t through (0, 1), (1, 3), (2, 5): a = 1, b = 2.
let a = Matrix::<3, 2>::new([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]);
let b = Vector::new([1.0, 3.0, 5.0]);
let x = PivotedQr::decompose(a).unwrap().solve_least_squares(b).unwrap();
```

## Matrix equations

Both solvers use doubling iterations, `O(n³)` per pass with a budget of 64 passes. Use them at
design time, not in a control loop.

- `solve_discrete_riccati`: the steady-state `P` of
  `P = Aᵀ·P·A − Aᵀ·P·B·(R + Bᵀ·P·B)⁻¹·Bᵀ·P·A + Q`, the cost-to-go of LQR. The solver checks the
  answer against the equation before it returns it.
- `solve_discrete_lyapunov`: `Aᵀ·P·A − P + Q = 0`. A solution exists only if `A` is Schur stable,
  so `DidNotConverge` is the verdict that the loop does not settle.

The `_with_budget` forms set the pass budget.

## Errors

Factorizations and solves return [`LinalgError`](error-handling.md):

- `Singular`, `IllConditioned` (LU pivot too small for a reliable solve), `NotPositiveDefinite`.
- `Matrix::inverse` normalizes a 1×1 to 4×4 matrix by its largest entry `s` before the test
  `|det(A/s)| ≤ n·ε`, so the result does not depend on the scale. It returns `Singular` when the
  inverse is not finite, and `NonFinite` for a non-finite entry.
- `Underdetermined`: a least-squares system or an SVD with `M < N`.
- `NotSymmetric`, `NonFinite`.
- `DidNotConverge`: a matrix-equation solver ran out of its budget.
- `OutOfBounds`: the only error that view operations return.

Credits: QR, the damped solve and the overflow-safe norm port MINPACK `qrfac`, `qrsolv` and
`enorm` (Moré, Garbow, Hillstrom; public domain, netlib). The SVD follows Golub & Van Loan,
*Matrix Computations*, and Demmel & Veselić. The symmetric eigendecomposition follows the Jacobi
method in Golub & Van Loan. Demos:
[linear_algebra.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/linear_algebra.rs),
[svd.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/svd.rs).

---

[Back to the tutorial index](README.md)
