# Derivatives, Jacobians, and Hessians

Total and partial derivatives, exact by forward-mode autodiff or approximate by finite
differences, plus Jacobian and Hessian matrices.

- `derivative`, `second_derivative`, `partial`: one-call functions for the common case. See
  [One-call functions and strategy objects](importing.md#one-call-functions-and-strategy-objects).
- `AutoDiffSingle` / `AutoDiffMulti`: exact derivatives, to order 6 for one variable and to order 3
  for several variables.
- `FiniteDifferenceSingle` / `FiniteDifferenceMulti`: for black-box functions that you cannot
  write with `scalar_fn!`. `from_parameters(step, mode, multiplier)` sets the step size and the
  `FiniteDifferenceMode` (`Forward`, `Backward` or `Central`).
- Both backends implement `DerivatorSingleVariable` (`differentiate`, `first_derivative`,
  `second_derivative`) and `DerivatorMultiVariable` (`differentiate`, `first_partial_derivative`,
  `second_partial_derivative`).
- `Jacobian` and `Hessian` build the matrices.

For several variables, the derivative order is the number of indices in the array:

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

`Jacobian` differentiates the rows of a `scalar_fn_vec!` function. `Hessian` takes a scalar field:

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

Errors: these calls return [`DiffError`](error-handling.md):

- `OrderZero`, or `OrderUnsupported` above the autodiff order limit.
- `StepSizeZero` or `InvalidStepSize` for a finite-difference step.
- `IndexOutOfRange` for a variable index outside the point.
- `EmptyFunctionSet` for a Jacobian of a function with no outputs.

Full demos:
[differentiation.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/differentiation.rs)
and
[jacobian_hessian.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/jacobian_hessian.rs).


---

[Back to the tutorial index](README.md)
