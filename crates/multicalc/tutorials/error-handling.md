# Error handling

Each module family returns its own error enum. All six convert into the `CalcError` umbrella
through `From`, so a caller that spans families can hold a single type. Every enum is
`#[non_exhaustive]` and `Copy`, and implements `Display` and `core::error::Error`.

| Enum | Raised by | Variants |
| --- | --- | --- |
| `LinalgError` | [Linear algebra](linear-algebra.md), [Discretization](discretization.md) | `Singular`, `NotPositiveDefinite`, `Underdetermined`, `NonFinite`, `NotSymmetric`, `InvalidTimestep`, `OutOfBounds` |
| `DiffError` | [Derivatives](derivatives-jacobians-and-hessians.md), [Approximation](taylor-approximation.md), [Vector calculus](vector-calculus.md) | `OrderZero`, `OrderUnsupported`, `StepSizeZero`, `IndexOutOfRange`, `EmptyFunctionSet` |
| `IntegrateError` | [Integration](integration.md), [Gaussian tables](gaussian-quadrature-tables.md), [ODE](ode-integrators.md) | `IterationsZero`, `LimitsIllDefined`, `QuadratureOrderOutOfRange`, `StepSizeTooSmall`, `DidNotConverge { steps }`, `NonFinite`, `IndexOutOfRange`, `NonPositiveTimestep` |
| `SolveError` | [Optimization](least-squares-optimization.md), [Root finding](root-finding.md) | `DidNotConverge { iters, residual }`, `NonFinite`, `InvalidBracket`, `Linalg(LinalgError)`, `Diff(DiffError)` |
| `KinematicsError` | [Kinematics](kinematics.md) | `NonPositiveParameter`, `NonFinite` |
| `EstimationError` | [Estimation](estimation.md) | `NotPositiveDefinite`, `NonFinite`, `Diff(DiffError)`, `WeightsDegenerate`, `InvalidTuning` |
| `CalcError` | umbrella | `Linalg`, `Solve`, `Integrate`, `Differentiate`, `Kinematics`, `Estimation` |

`SolveError` wraps `LinalgError` and `DiffError` (a solver step can fail in either), and both
are reachable through `core::error::Error::source`. Convert up to the umbrella with `?` or
`.into()`:

```rust
use multicalc::{CalcError, Matrix, Matrix3D, Vector};

// One return type covers a function that mixes modules: each `?` converts the module's own
// error into the umbrella on its way out.
fn solve() -> Result<(), CalcError> {
    let a = Matrix::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]);
    let b = Vector::new([7.0, 19.0, 49.0]);

    let x = a.lu_decompose()?.solve(b);          // LinalgError -> CalcError
    assert!((a * x - b).norm() < 1e-9);

    // A singular matrix returns `LinalgError::Singular` here rather than panicking.
    let singular = Matrix3D::<f64>::zeros();
    assert!(singular.lu_decompose().is_err());

    Ok(())
}
# solve().unwrap();
```

This is the shape the converted demos use — see
[linear_algebra.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/linear_algebra.rs),
[root_finding.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/root_finding.rs),
and
[estimation.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/estimation.rs),
each of which returns `Result<(), CalcError>` from `main` and propagates with `?`.


---

[Back to the tutorial index](README.md)
