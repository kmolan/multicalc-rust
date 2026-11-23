# Scalars and automatic differentiation

The calculus modules are generic over the `Numeric` scalar trait. The forward-mode autodiff
scalars also implement `Numeric`.

- `Numeric`: the scalar trait, implemented for `f32` and `f64`.
- `Dual`: `value + deriv·ε` with `ε² = 0`, for exact first derivatives.
- `HyperDual`: exact first and second derivatives.
- `Jet<T, N>`: a truncated Taylor series, `coeffs[k] = f⁽ᵏ⁾(x) / k!`, for exact derivatives to
  order `N - 1`. `Dual` is the order-1 case, equal to `Jet<T, 2>` in value but a separate type.
- `ScalarFn` / `ScalarFnN` / `VectorFn`: function traits with an `eval` that is generic over the
  scalar, so one formula runs at `f64` or at an autodiff type.
- `scalar_fn!` / `scalar_fn_vec!`: macros that build those traits from closure syntax. Wrap each
  numeric constant in the body in `constant()`, because a bare `2.0 * x` does not typecheck in a
  generic body.

`AutoDiffSingle::differentiate` uses `Dual` for order 1, `HyperDual` for order 2, and a `Jet` for
orders 3 to 6:

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

Errors: differentiation calls return [`DiffError`](error-handling.md): `OrderZero`, or
`OrderUnsupported` above order 6. Full demo:
[autodiff_scalars.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/autodiff_scalars.rs).


---

[Back to the tutorial index](README.md)
