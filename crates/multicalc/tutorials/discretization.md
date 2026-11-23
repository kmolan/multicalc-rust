# Discretization

Discrete-time equivalents of continuous-time linear systems over a step `dt`.

- `zoh(a, b, dt)`: exact zero-order hold of `ẋ = A x + B u`. It returns `(F, G)` with
  `F = expm(A·dt)` and `G = ∫₀^dt expm(A·τ) dτ · B`, from the augmented-matrix exponential. The
  const parameter `NM` must equal `N + M`. A mismatch is a compile error.
- `van_loan(a, qc, dt)`: Van Loan discretization of `(A, Q_c)`. It returns `(F, Q_d)` with
  `Q_d = ∫₀^dt expm(A·τ) Q_c expm(A·τ)ᵀ dτ`. The const parameter `N2` must equal `2·N`.
- `q_discrete_white_noise(dt, var)`: the filterpy-compatible piecewise white-noise covariance for
  a chain of `DIM ∈ {2, 3, 4}` integrators. It is a closed form, with no matrix exponential, and
  no error path.

`zoh` and `van_loan` use `Matrix::expm`, which is generic over the scalar. Thus a `Dual` through
them gives the exact derivative with respect to a parameter.

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
let ad = Matrix2D::<Dual<f64>>::from_fn(|i, j| Dual::new(0.0, m[(i, j)]))
    .expm()
    .unwrap();
// ad[(0, 1)].deriv == m[(0, 1)]
```

Errors: `zoh` and `van_loan` return [`LinalgError::InvalidTimestep`](error-handling.md) if `dt` is
negative or non-finite. They propagate errors from `expm` (`NonFinite`) unchanged. Full demo:
[discretization.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/discretization.rs).


---

[Back to the tutorial index](README.md)
