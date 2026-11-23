# multicalc tutorials

One page for each public module: what it does, a runnable snippet, the errors it can return, and a
link to a full demo. The robotics layer (estimation, control, kinematics, motion, spatial) builds on
the calculus, autodiff and linear-algebra core. If you are new to the crate, read the pages in order.

Each operation is generic over the [`Numeric`](scalars-and-automatic-differentiation.md) scalar
trait. `Numeric` is implemented for `f32` and `f64`, and the default is `f64`. The math functions
come from `libm`, so the crate works without `std`. Methods such as `f64::sin` need `std`. In a
`no_std` crate, call `libm::sin(x)` in place of `x.sin()`. The crate re-exports `libm` as
`multicalc::libm`.

Each fallible call returns a `Result` with the module family's own error enum. See
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
- [Planning](planning.md)
- [MLP inference](mlp-inference.md)
- [Estimation](estimation.md)
- [Random](random.md)
- [Error handling](error-handling.md)
- [Internals](internals.md)
