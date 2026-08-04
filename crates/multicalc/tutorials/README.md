# multicalc tutorials

A tour of every public module — the robotics and control layer (estimation, control, kinematics,
motion, spatial) and the calculus, autodiff, and linear-algebra core it is built on: what each does,
where to start, a snippet you can run, the errors it can return, and a link to a full demo. Read the
pages in order if you are new to the crate, or open a single one once you know your way around.

Every operation is generic over the [`Numeric`](scalars-and-automatic-differentiation.md) scalar
trait, which is implemented for `f32` and `f64` and defaults to `f64`. The math functions come
from `libm`, so the crate works without `std`. Methods like `f64::sin` need `std`; in a
`no_std` crate, call the `libm` version instead (`libm::sin(x)` in place of `x.sin()`). The
crate re-exports `libm` as `multicalc::libm`.

Every fallible call returns a `Result`, and the error is the module family's own enum; see
[Error handling](error-handling.md).

## Contents

- [Importing](importing.md)
- [Scalars and automatic differentiation](scalars-and-automatic-differentiation.md)
- [Derivatives, Jacobians, and Hessians](derivatives-jacobians-and-hessians.md)
- [Integration](integration.md)
- [Gaussian quadrature tables](gaussian-quadrature-tables.md)
- [Taylor approximation](taylor-approximation.md)
- [Linear algebra](linear-algebra.md)
- [Least-squares optimization](least-squares-optimization.md)
- [Root finding](root-finding.md)
- [Polynomials](polynomials.md)
- [Vector calculus](vector-calculus.md)
- [ODE integrators](ode-integrators.md)
- [Discretization](discretization.md)
- [Signal processing](signal-processing.md)
- [Spatial: quaternions and Lie groups](spatial-quaternions-and-lie-groups.md)
- [Rigid-body dynamics](rigid-body-dynamics.md)
- [Plant](plant.md)
- [Kinematics](kinematics.md)
- [Control](control.md)
- [Motion](motion.md)
- [Mapping](mapping.md)
- [Estimation](estimation.md)
- [Random](random.md)
- [Error handling](error-handling.md)
- [Internals](internals.md)
