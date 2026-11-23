# Error handling

Each module family returns its own error enum. Each enum converts into the `CalcError` umbrella
through `From`, so a caller that spans families can use one error type. Each enum is
`#[non_exhaustive]` and `Copy`, and implements `Display` and `core::error::Error`.

| Enum | Raised by | Variants |
| --- | --- | --- |
| `LinalgError` | [Linear algebra](linear-algebra.md), [Discretization](discretization.md), [MLP inference](mlp-inference.md) | `Singular`, `IllConditioned`, `NotPositiveDefinite`, `Underdetermined`, `NonFinite`, `NotSymmetric`, `InvalidTimestep`, `DidNotConverge { iters }`, `OutOfBounds` |
| `DiffError` | [Derivatives](derivatives-jacobians-and-hessians.md), [Approximation](taylor-approximation.md), [Vector calculus](vector-calculus.md) | `OrderZero`, `OrderUnsupported`, `StepSizeZero`, `InvalidStepSize`, `IndexOutOfRange`, `EmptyFunctionSet` |
| `IntegrateError` | [Integration](integration.md), [Gaussian tables](gaussian-quadrature-tables.md), [ODE](ode-integrators.md), [Vector calculus](vector-calculus.md) | `IterationsZero`, `LimitsIllDefined`, `QuadratureOrderOutOfRange`, `StepSizeTooSmall`, `DidNotConverge { steps }`, `NonFinite`, `IndexOutOfRange`, `NonPositiveTimestep` |
| `SolveError` | [Optimization](least-squares-optimization.md), [Root finding](root-finding.md) | `DidNotConverge { iters }`, `NonFinite`, `InvalidBracket`, `Linalg(LinalgError)`, `Diff(DiffError)` |
| `PolynomialError` | [Polynomials](polynomials.md) | `NonFinite`, `LeadingCoefficientZero`, `DegreeOverflow`, `CapacityExceeded`, `Empty`, `VariableOutOfRange`, `DuplicateNode`, `TooFewSamples`, `SpanNotPositive`, `ToleranceNotPositive`, `RangeReversed`, `DidNotConverge { steps }`, `Linalg(LinalgError)` |
| `SignalError` | [Signal processing](signal-processing.md) | `NonFinite`, `NonPositiveTimestep`, `CoefficientOutOfRange`, `FrequencyOutOfRange`, `NonPositiveQualityFactor`, `NegativeThreshold`, `ThresholdsOutOfOrder`, `NonPositiveRate`, `WindowTooShort`, `WindowEvenLength`, `PolynomialOrderTooHigh`, `SectionIndexOutOfRange`, `Linalg(LinalgError)` |
| `SpatialError` | [Spatial](spatial-quaternions-and-lie-groups.md) | `NonPositiveMass`, `NonFinite`, `NotSymmetric`, `NonPositiveInertia` |
| `DynamicsError` | [Rigid-body dynamics](rigid-body-dynamics.md) | `NonFinite`, `NonPositiveInertia`, `InertiaCountMismatch`, `FloatingBaseUnsupported`, `StateShapeMismatch`, `DegenerateOrientation`, `NonPositiveArticulatedInertia`, `Kinematics(KinematicsError)`, `Linalg(LinalgError)` |
| `PlantError` | [Plant](plant.md) | `NonFinite`, `NonPositiveArmLength`, `NonPositiveTorqueRatio`, `InvalidThrustLimits`, `RotorLayoutNotIndependent`, `NonPositiveTimeConstant`, `NonPositiveTimestep`, `NonPositiveNaturalFrequency`, `NegativeDampingRatio`, `Linalg(LinalgError)` |
| `KinematicsError` | [Kinematics](kinematics.md) | `NonPositiveParameter`, `NonFinite`, `CapacityExceeded`, `JointCountMismatch`, `ParentOutOfOrder`, `AxisHasNoDirection`, `LimitsReversed`, `ToolIndexOutOfRange`, `StateShapeMismatch`, `NonPositiveWeight`, `FloatingJointNotAtRoot`, `ConfigurationCapacityExceeded`, `OrientationHasNoDirection`, `ContinuousJointHasLimits`, `StartCapacityExceeded`, `CollisionCapacityExceeded`, `Linalg(LinalgError)` |
| `ControlError` | [Control](control.md) | `NonFinite`, `NonPositiveTimestep`, `InvalidOutputLimits`, `FilterCoefficientOutOfRange` (deprecated), `NonPositiveLookaheadDistance`, `InvalidBeamCount`, `InvalidFieldOfView`, `NonPositiveRange`, `NonPositiveChassisWidth`, `NonPositiveSpeed`, `InvalidSpeedScaling`, `NegativeGoalBias`, `NonPositiveGain`, `NegativeGain`, `NotSymmetricInertia`, `NonPositiveInertia`, `UndefinedThrustDirection`, `UndefinedHeadingDirection`, `InvalidTiltLimit`, `Linalg(LinalgError)`, `Signal(SignalError)`, `Dynamics(DynamicsError)`, `Kinematics(KinematicsError)` |
| `MotionError` | [Motion](motion.md) | `NonFinite`, `CapacityExceeded`, `OutOfSync`, `PathTooShort`, `SegmentCountMismatch`, `DurationNotPositive`, `WorkspaceTooSmall`, `LimitNotPositive`, `JerkLimitRequired`, `Linalg(LinalgError)`, `Polynomial(PolynomialError)` |
| `MappingError` | [Mapping](mapping.md) | `NonFinite`, `NonPositiveResolution`, `EmptyGrid`, `GridTooLarge`, `TooFewBeams`, `InvalidFieldOfView`, `NonPositiveRange`, `InvalidRangeLimits`, `WordsPerRowTooSmall`, `GridShapeMismatch`, `WorkspaceTooSmall`, `InvalidBeliefSettings`, `RadiiNotOrdered`, `NonPositiveScaling` |
| `PlanningError` | [Planning](planning.md) | `NonFinite`, `NonPositiveParameter`, `InvalidGoalBias`, `HeuristicWeightBelowOne`, `InadmissibleHeuristic`, `BoundsReversed`, `MapTooLarge`, `WorkspaceTooSmall`, `StartOutOfBounds`, `GoalOutOfBounds`, `StartNotFree`, `GoalNotFree`, `NoPathFound`, `DidNotConverge { iterations }`, `PathCapacityExceeded { needed }`, `Motion(MotionError)` |
| `EstimationError` | [Estimation](estimation.md) | `NotPositiveDefinite`, `NonFinite`, `Diff(DiffError)`, `WeightsDegenerate`, `InvalidTuning`, `StateIndexOutOfRange` |
| `CalcError` | umbrella | `Linalg`, `Solve`, `Integrate`, `Differentiate`, `Kinematics`, `Spatial`, `Estimation`, `Signal`, `Control`, `Dynamics`, `Plant`, `Mapping`, `Motion`, `Planning`, `Polynomial` |

An enum that wraps another, for example `SolveError::Linalg(LinalgError)`, converts from it with
`From`, so `?` lifts an inner failure into the outer enum. `core::error::Error::source` returns the
wrapped error, but `ControlError::Dynamics` and `ControlError::Kinematics` return `None`. Convert
to the umbrella with `?` or `.into()`:

```rust
use multicalc::{CalcError, Matrix, Matrix3D, Vector};

// Each `?` converts the module's own error into `CalcError`.
fn solve() -> Result<(), CalcError> {
    let a = Matrix::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]);
    let b = Vector::new([7.0, 19.0, 49.0]);

    let x = a.lu_decompose()?.solve(b);          // LinalgError -> CalcError
    assert!((a * x - b).norm() < 1e-9);

    // A singular matrix returns `LinalgError::Singular`. It does not panic.
    let singular = Matrix3D::<f64>::zeros();
    assert!(singular.lu_decompose().is_err());

    Ok(())
}
# solve().unwrap();
```

These demos return `Result<(), CalcError>` from `main` and propagate with `?`:

- [linear_algebra.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/linear_algebra.rs)
- [root_finding.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/root_finding.rs)
- [estimation.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/estimation.rs)
- [error_state_estimation.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/error_state_estimation.rs)
- [attitude_filter.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/attitude_filter.rs)

---

[Back to the tutorial index](README.md)
