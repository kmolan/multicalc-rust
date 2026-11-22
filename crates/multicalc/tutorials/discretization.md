# Discretization

Turn a continuous-time linear system into its discrete-time equivalent over a step `dt`.

- `zoh(a, b, dt)`: zero-order-hold discretization of `(A, B)`, returning the discrete `(F, G)`.
- `van_loan(a, qc, dt)`: Van Loan discretization of continuous process noise, returning the
  discrete transition and process-noise covariance `(F, Q_d)`.
- `q_discrete_white_noise(dt, var)`: the filterpy-compatible discrete white-noise model.

Because the routines run through the matrix exponential, an autodiff scalar flows straight
through them: a single `Dual` recovers a derivative with respect to a parameter.

```rust
use multicalc::{q_discrete_white_noise, van_loan, zoh};
use multicalc::{Matrix, Matrix2D};
use multicalc::Dual;

let dt = 0.1;

// Zero-order hold of the double integrator: F = [[1, dt], [0, 1]], G = [[dt^2/2], [dt]].
let a = Matrix2D::new([[0.0, 1.0], [0.0, 0.0]]);
let b = Matrix::<2, 1>::new([[0.0], [1.0]]);
let (f, g) = zoh::<2, 1, 3, f64>(a, b, dt).unwrap();      // f[(0, 1)] == dt, g[(1, 0)] == dt

// Van Loan process-noise discretization of continuous white noise on velocity.
let qc = Matrix2D::new([[0.0, 0.0], [0.0, 1.0]]);
let (_f, qd) = van_loan::<2, 4, f64>(a, qc, dt).unwrap(); // qd[(1, 1)] == dt, symmetric

// Discrete white-noise model.
let q = q_discrete_white_noise::<2, f64>(dt, 2.0);        // q[(1, 1)] == 2*dt^2

// d/dx expm(x·M) at x = 0 equals M, recovered by one Dual through expm.
let m = Matrix2D::new([[0.2, 0.5], [-0.1, 0.3]]);
let ad = Matrix2D::<Dual<f64>>::from_fn(|i, j| {
  Dual::new(0.0, m[(i, j)])
})
    .expm()
    .unwrap();
// ad[(0, 1)].deriv == m[(0, 1)]
```

Errors: `zoh` and `van_loan` return [`LinalgError::InvalidTimestep`](error-handling.md) when `dt` is
negative or non-finite; errors from the matrix-exponential step are propagated unchanged. Full
demo:
[discretization.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/discretization.rs).


---

[Back to the tutorial index](README.md)
