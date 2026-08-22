# Scalars and automatic differentiation

The scalar number system that every calculus module is generic over: the `Numeric` trait, plus
the forward-mode automatic-differentiation numbers that also implement it.

- `Numeric`: the scalar trait, implemented for `f32` and `f64`.
- `Dual`, `HyperDual`, `Jet<T, N>`: autodiff scalars (dual numbers) carrying exact first,
  second, and arbitrary nth-order derivatives (`Dual` is `Jet<T, 2>`).
- `ScalarFn` / `ScalarFnN` / `VectorFn`: function traits whose `eval` is generic over the
  scalar, so one formula runs at `f64` or at any autodiff type.
- The `scalar_fn!` / `scalar_fn_vec!` macros build those traits from closure syntax, and `constant()`
  marks numeric constants inside the body (a bare `2.0 * x` cannot typecheck in a generic body).

One formula, differentiated exactly to any order:

```rust
use multicalc::AutoDiffSingle;
use multicalc::DerivatorSingleVariable;
use multicalc::scalar_fn;

let function = scalar_fn!(|x| x * x * x);    // f(x) = x^3, evaluable at any Numeric
let derivator = AutoDiffSingle::default();   // forward-mode autodiff, exact
let point = 2.0;

let first = derivator.differentiate(1, &function, point).unwrap();   // 12.0
let third = derivator.differentiate(3, &function, point).unwrap();   //  6.0
```

Errors: differentiation calls return [`DiffError`](error-handling.md) (for example `OrderZero`).

Credits: standard forward-mode dual numbers. Full demo:
[autodiff_scalars.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/autodiff_scalars.rs).


---

[Back to the tutorial index](README.md)
