# Derivatives, Jacobians, and Hessians

Derivatives of any order, total and partial — exact through forward-mode autodiff, or by finite
differences for black-box functions — plus Jacobian and Hessian matrices.

- `derivative`, `second_derivative`, `partial`: one-call functions covering the common case. See
  [the note on the two paths](importing.md#the-easy-path-and-the-configurable-one).
- `AutoDiffSingle` / `AutoDiffMulti`: exact derivatives, to any order.
- `FiniteDifferenceSingle` / `FiniteDifferenceMulti`: for functions you cannot author with
  `scalar_fn!`.
- Both implement the `DerivatorSingleVariable` / `DerivatorMultiVariable` traits
  (`differentiate`, `first_derivative`, `second_derivative`, `first_partial_derivative`).
- `Jacobian` and `Hessian` build the matrices.

For several variables, the derivative order is just the number of indices you pass:

```rust
use multicalc::AutoDiffMulti;
use multicalc::DerivatorMultiVariable;
use multicalc::scalar_fn;

// g(x, y, z) = y*sin(x) + x*cos(y) + x*y*e^z; order = number of indices passed
let g = scalar_fn!(|v: &[f64; 3]| v[1] * v[0].sin() + v[0] * v[1].cos() + v[0] * v[1] * v[2].exp());
let d = AutoDiffMulti::default();
let point = [1.0, 2.0, 3.0];

let x_index = 0;
let dx = d.first_partial_derivative(&g, x_index, &point).unwrap();

let then_by_y = [0, 1];
let mixed = d.differentiate(&g, &then_by_y, &point).unwrap();      // d(dg/dx)/dy

let twice_by_x_then_y = [0, 0, 1];
let third = d.differentiate(&g, &twice_by_x_then_y, &point).unwrap();
```

Pass a finite-difference derivator (`FiniteDifferenceSingle` / `FiniteDifferenceMulti`) instead
when the function is a black box you cannot author with `scalar_fn!`.

Write a vector-valued function with `scalar_fn_vec!` and its rows differentiate under autodiff
to give the Jacobian; a scalar field gives the Hessian:

```rust
use multicalc::Jacobian;
use multicalc::Hessian;
use multicalc::constant;
use multicalc::{scalar_fn, scalar_fn_vec};

// the vector function (x*y*z, x^2 + y^2)
let f = scalar_fn_vec!(|v: &[f64; 3]| [v[0] * v[1] * v[2], v[0] * v[0] + v[1] * v[1]]);
let jacobian_point = [1.0, 2.0, 3.0];
let jacobian: Jacobian = Jacobian::default();
let j = jacobian.evaluate(&f, &jacobian_point).unwrap();   // [[6, 3, 2], [2, 4, 0]]

// g(x, y) = y*sin(x) + 2*x*e^y
let g = scalar_fn!(|v: &[f64; 2]| v[1] * v[0].sin() + constant(2.0) * v[0] * v[1].exp());
let hessian_point = [1.0, 2.0];
let hessian: Hessian = Hessian::default();
let h = hessian.evaluate(&g, &hessian_point).unwrap();
```

With the `alloc` feature, `Jacobian::evaluate_on_heap` returns a `Vec<Vec<T>>` for inputs too large
for the stack.

Errors: these calls return [`DiffError`](error-handling.md): `OrderZero`, `OrderUnsupported`,
`StepSizeZero` (finite differences), or `IndexOutOfRange`. Full demos:
[differentiation.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/differentiation.rs)
and
[jacobian_hessian.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/jacobian_hessian.rs).


---

[Back to the tutorial index](README.md)
