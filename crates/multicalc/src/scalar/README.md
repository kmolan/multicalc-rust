# scalar

The scalar number system every calculus module is generic over: the [`Numeric`](numeric.rs) trait
plus the forward-mode autodiff numbers that also implement it.

- [`Numeric`](numeric.rs) — the scalar trait, implemented for `f32` and `f64`.
- [`Dual`](dual.rs), [`HyperDual`](hyper_dual.rs), [`Jet<T, N>`](jet.rs) — autodiff scalars carrying
  exact first, second, and arbitrary nth-order derivatives (`Dual` is `Jet<T, 2>`).
- [`ScalarFn`](function.rs) / `ScalarFnN` / `VectorFn` — function traits whose `eval` is generic over
  the scalar, so one formula runs at `f64` or at any autodiff type.
- The `scalar_fn!` / `scalar_fn_vec!` macros build those traits from closure syntax, and `c()` marks
  numeric constants inside the body (a bare `2.0 * x` cannot typecheck in a generic body).

```rust
use multicalc::numerical_derivative::autodiff::AutoDiffSingle;
use multicalc::numerical_derivative::derivator::DerivatorSingleVariable;
use multicalc::scalar_fn;

let f = scalar_fn!(|x| x * x * x);           // f(x) = x^3, evaluable at any Numeric
let d = AutoDiffSingle::default();           // forward-mode autodiff, exact

let first = d.get(1, &f, 2.0).unwrap();      // 12.0
let third = d.get(3, &f, 2.0).unwrap();      //  6.0
```

Credits: standard forward-mode dual numbers.
