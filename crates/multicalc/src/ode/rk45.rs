//! Adaptive Dormand–Prince 5(4) with PI step control and cubic-Hermite dense output.

use crate::error::IntegrateError;
use crate::linear_algebra::Vector;
use crate::ode::tableau::*;
use crate::scalar::Numeric;

/// One accepted RK45 step, carrying the data for cubic-Hermite interpolation inside `[t0, t1]`.
#[derive(Debug, Clone, Copy)]
pub struct Step<const N: usize, T: Numeric> {
    /// Step start time.
    pub t0: T,
    /// Step end time.
    pub t1: T,
    /// State at `t0`.
    pub y0: Vector<N, T>,
    /// State at `t1`.
    pub y1: Vector<N, T>,
    /// Derivative `f(t0, y0)`.
    pub f0: Vector<N, T>,
    /// Derivative `f(t1, y1)`.
    pub f1: Vector<N, T>,
}

impl<const N: usize, T: Numeric> Step<N, T> {
    /// Cubic-Hermite interpolation of the state at `t` in `[t0, t1]`. Returns `y0` at `t0`
    /// and `y1` at `t1` exactly.
    pub fn interpolate(&self, t: T) -> Vector<N, T> {
        let h = self.t1 - self.t0;
        let s = (t - self.t0) / h; // normalized position in [0, 1]
        let s2 = s * s;
        let s3 = s2 * s;
        let three = T::from_f64(3.0);
        let h00 = T::TWO * s3 - three * s2 + T::ONE;
        let h10 = s3 - T::TWO * s2 + s;
        let h01 = -T::TWO * s3 + three * s2;
        let h11 = s3 - s2;
        self.y0.scale(h00) + self.f0.scale(h10 * h) + self.y1.scale(h01) + self.f1.scale(h11 * h)
    }
}

/// Adaptive Dormand–Prince 5(4) integrator for `y' = f(t, y)` with state `Vector<N, T>`.
///
/// Build with [`Rk45::default`] and adjust with the `with_*` methods. Solve with
/// [`solve`](Rk45::solve), stream accepted steps with [`for_each_step`](Rk45::for_each_step),
/// or sample a time grid with [`solve_on_grid`](Rk45::solve_on_grid).
pub struct Rk45<T: Numeric = f64> {
    rtol: T,
    atol: T,
    first_step: T, // 0 => auto-select on the first step
    min_step: T,   // 0 => no floor
    max_step: T,
    max_steps: usize,
}

impl<T: Numeric> Default for Rk45<T> {
    fn default() -> Self {
        Rk45 {
            rtol: T::from_f64(1e-6),
            atol: T::from_f64(1e-9),
            first_step: T::ZERO,
            min_step: T::ZERO,
            max_step: T::INFINITY,
            max_steps: 100_000,
        }
    }
}

impl<T: Numeric> Rk45<T> {
    /// Sets the relative tolerance (default `1e-6`).
    pub fn with_rtol(mut self, rtol: T) -> Self {
        self.rtol = rtol;
        self
    }
    /// Sets the absolute tolerance (default `1e-9`).
    pub fn with_atol(mut self, atol: T) -> Self {
        self.atol = atol;
        self
    }
    /// Sets the first step size; `0` (the default) auto-selects it.
    pub fn with_first_step(mut self, h: T) -> Self {
        self.first_step = h;
        self
    }
    /// Sets the minimum step magnitude; falling below it returns [`IntegrateError::StepSizeTooSmall`].
    /// `0` (the default) disables the floor.
    pub fn with_min_step(mut self, h: T) -> Self {
        self.min_step = h;
        self
    }
    /// Sets the maximum step magnitude (default unbounded).
    pub fn with_max_step(mut self, h: T) -> Self {
        self.max_step = h;
        self
    }
    /// Sets the maximum number of step attempts before [`IntegrateError::DidNotConverge`]
    /// (default `100_000`).
    pub fn with_max_steps(mut self, n: usize) -> Self {
        self.max_steps = n;
        self
    }
}

/// RMS of `err_i / (atol + rtol * max(|y0_i|, |y1_i|))` over the components.
fn error_norm<const N: usize, T: Numeric>(
    err: &Vector<N, T>,
    y0: &Vector<N, T>,
    y1: &Vector<N, T>,
    atol: T,
    rtol: T,
) -> T {
    let mut sum = T::ZERO;
    for ((e, a), b) in err.as_array().iter().zip(y0.as_array()).zip(y1.as_array()) {
        let scale = atol + rtol * a.abs().max(b.abs());
        let r = *e / scale;
        sum += r * r;
    }
    (sum / T::from_usize(N)).sqrt()
}

/// RMS of `v_i / (atol + rtol * |y_i|)` — used by the initial-step heuristic.
fn scaled_norm<const N: usize, T: Numeric>(
    v: &Vector<N, T>,
    y: &Vector<N, T>,
    atol: T,
    rtol: T,
) -> T {
    let mut sum = T::ZERO;
    for (e, a) in v.as_array().iter().zip(y.as_array()) {
        let scale = atol + rtol * a.abs();
        let r = *e / scale;
        sum += r * r;
    }
    (sum / T::from_usize(N)).sqrt()
}

impl<T: Numeric> Rk45<T> {
    /// One Dormand–Prince 5(4) trial step of size `h` from `(t, y)`.
    ///
    /// `k1` is `f(t, y)`, supplied by the caller so it can be reused from the previous
    /// accepted step (FSAL). Returns `(y5, err, k7)`: the 5th-order state `y5`, the embedded
    /// error estimate `err = y5 − y4`, and `k7 = f(t + h, y5)` (the next step's `k1`).
    fn dopri_step<const N: usize, F>(
        &self,
        f: &F,
        t: T,
        y: &Vector<N, T>,
        h: T,
        k1: Vector<N, T>,
    ) -> (Vector<N, T>, Vector<N, T>, Vector<N, T>)
    where
        F: Fn(T, &Vector<N, T>) -> Vector<N, T>,
    {
        let c2 = T::from_f64(C2);
        let c3 = T::from_f64(C3);
        let c4 = T::from_f64(C4);
        let c5 = T::from_f64(C5);
        // stage coefficients times h
        let a21 = T::from_f64(A21) * h;
        let a31 = T::from_f64(A31) * h;
        let a32 = T::from_f64(A32) * h;
        let a41 = T::from_f64(A41) * h;
        let a42 = T::from_f64(A42) * h;
        let a43 = T::from_f64(A43) * h;
        let a51 = T::from_f64(A51) * h;
        let a52 = T::from_f64(A52) * h;
        let a53 = T::from_f64(A53) * h;
        let a54 = T::from_f64(A54) * h;
        let a61 = T::from_f64(A61) * h;
        let a62 = T::from_f64(A62) * h;
        let a63 = T::from_f64(A63) * h;
        let a64 = T::from_f64(A64) * h;
        let a65 = T::from_f64(A65) * h;

        let k2 = f(t + c2 * h, &(*y + k1.scale(a21)));
        let k3 = f(t + c3 * h, &(*y + k1.scale(a31) + k2.scale(a32)));
        let k4 = f(
            t + c4 * h,
            &(*y + k1.scale(a41) + k2.scale(a42) + k3.scale(a43)),
        );
        let k5 = f(
            t + c5 * h,
            &(*y + k1.scale(a51) + k2.scale(a52) + k3.scale(a53) + k4.scale(a54)),
        );
        let k6 = f(
            t + h,
            &(*y + k1.scale(a61) + k2.scale(a62) + k3.scale(a63) + k4.scale(a64) + k5.scale(a65)),
        );

        let y5 = *y
            + (k1.scale(T::from_f64(B1))
                + k3.scale(T::from_f64(B3))
                + k4.scale(T::from_f64(B4))
                + k5.scale(T::from_f64(B5))
                + k6.scale(T::from_f64(B6)))
            .scale(h);
        let k7 = f(t + h, &y5);
        let err = (k1.scale(T::from_f64(E1))
            + k3.scale(T::from_f64(E3))
            + k4.scale(T::from_f64(E4))
            + k5.scale(T::from_f64(E5))
            + k6.scale(T::from_f64(E6))
            + k7.scale(T::from_f64(E7)))
        .scale(h);
        (y5, err, k7)
    }

    /// Picks the first step size, signed by `dir` (`+1` forward, `-1` backward).
    ///
    /// If `first_step` was set it is used directly (capped by `max_step` and `span`).
    /// Otherwise this is the Hairer–Wanner heuristic: size a tentative step from the
    /// scaled norms of `y0` and `f0`, take one explicit Euler probe to estimate the second
    /// derivative, then combine them for a step matched to the method order (5). `f0` is
    /// `f(t0, y0)` and `span` is `|tf − t0|`; the result never exceeds `max_step` or `span`.
    fn select_initial_step<const N: usize, F>(
        &self,
        f: &F,
        t0: T,
        y0: &Vector<N, T>,
        f0: &Vector<N, T>,
        dir: T,
        span: T,
    ) -> T
    where
        F: Fn(T, &Vector<N, T>) -> Vector<N, T>,
    {
        if self.first_step > T::ZERO {
            let h = self.first_step.min(self.max_step).min(span);
            return dir * h;
        }
        let d0 = scaled_norm(y0, y0, self.atol, self.rtol);
        let d1 = scaled_norm(f0, y0, self.atol, self.rtol);
        let h0 = if d0 < T::from_f64(1e-5) || d1 < T::from_f64(1e-5) {
            T::from_f64(1e-6)
        } else {
            T::from_f64(0.01) * d0 / d1
        };
        let y1 = *y0 + f0.scale(dir * h0);
        let f1 = f(t0 + dir * h0, &y1);
        let d2 = scaled_norm(&(f1 - *f0), y0, self.atol, self.rtol) / h0;
        // exponent 1/(p+1) with method order p = 5
        let h1 = if d1.max(d2) <= T::from_f64(1e-15) {
            (h0 * T::from_f64(1e-3)).max(T::from_f64(1e-6))
        } else {
            (T::from_f64(0.01) / d1.max(d2)).powf(T::ONE / T::from_f64(6.0))
        };
        let h = (T::from_f64(100.0) * h0)
            .min(h1)
            .min(span.min(self.max_step));
        dir * h
    }

    /// Integrates from `t0` to `tf`, invoking `obs` with each accepted [`Step`], and returns
    /// the final state.
    ///
    /// # Errors
    /// [`LimitsIllDefined`](IntegrateError::LimitsIllDefined) for a NaN or
    /// zero-length span; [`NonFinite`](IntegrateError::NonFinite) if `f` or the state goes
    /// non-finite; [`StepSizeTooSmall`](IntegrateError::StepSizeTooSmall) if the step drops below
    /// `min_step`; [`DidNotConverge`](IntegrateError::DidNotConverge) if `max_steps` is exhausted.
    ///
    /// ```
    /// use multicalc::ode::Rk45;
    /// use multicalc::linear_algebra::Vector;
    /// // y' = -y over [0, 2]; y(2) = e^{-2}.
    /// let yf = Rk45::default()
    ///     .solve(&|_t, y: &Vector<1, f64>| -*y, 0.0, &Vector::new([1.0]), 2.0)
    ///     .unwrap();
    /// assert!((yf[0] - (-2.0_f64).exp()).abs() < 1e-6);
    /// ```
    pub fn for_each_step<const N: usize, F, O>(
        &self,
        f: &F,
        t0: T,
        y0: &Vector<N, T>,
        tf: T,
        mut obs: O,
    ) -> Result<Vector<N, T>, IntegrateError>
    where
        F: Fn(T, &Vector<N, T>) -> Vector<N, T>,
        O: FnMut(&Step<N, T>),
    {
        if !t0.is_finite() || !tf.is_finite() || t0 == tf {
            return Err(IntegrateError::LimitsIllDefined);
        }
        let span = (tf - t0).abs();
        let dir = if tf > t0 { T::ONE } else { -T::ONE };

        let mut t = t0;
        let mut y = *y0;
        let mut k1 = f(t, &y);
        if !y.is_finite() || !k1.is_finite() {
            return Err(IntegrateError::NonFinite);
        }
        let mut h = self.select_initial_step(f, t0, &y, &k1, dir, span);
        let mut err_prev = T::from_f64(1e-4);
        let mut kahan_c = T::ZERO;
        let mut steps = 0usize;

        for _ in 0..self.max_steps {
            steps += 1;
            // Do not overshoot tf (compare signed remaining against signed h).
            let remaining = tf - t;
            if h.abs() > remaining.abs() {
                h = remaining;
            }

            let (y5, err_vec, k7) = self.dopri_step(f, t, &y, h, k1);
            if !y5.is_finite() || !err_vec.is_finite() {
                return Err(IntegrateError::NonFinite);
            }
            let err = error_norm(&err_vec, &y, &y5, self.atol, self.rtol);
            let accept = err <= T::ONE;

            if accept {
                // Kahan-compensated t += h.
                let yy = h - kahan_c;
                let tnew = t + yy;
                kahan_c = (tnew - t) - yy;
                let step = Step {
                    t0: t,
                    t1: tnew,
                    y0: y,
                    y1: y5,
                    f0: k1,
                    f1: k7,
                };
                obs(&step);
                t = tnew;
                y = y5;
                k1 = k7; // FSAL
                if (tf - t).abs() <= T::EPSILON * (T::ONE + tf.abs()) {
                    return Ok(y);
                }
            }

            // PI step-size update (uses err and the previous accepted err).
            let e = err.max(T::from_f64(1e-10));
            let factor =
                T::from_f64(0.9) * e.powf(-T::from_f64(0.17)) * err_prev.powf(T::from_f64(0.04));
            let mut factor = factor.max(T::from_f64(0.2)).min(T::from_f64(10.0));
            if !accept {
                factor = factor.min(T::ONE);
            }
            h *= factor;
            if h.abs() > self.max_step {
                h = self.max_step.copysign(h);
            }
            if accept {
                err_prev = err.max(T::from_f64(1e-4));
            }
            if self.min_step > T::ZERO && h.abs() < self.min_step {
                return Err(IntegrateError::StepSizeTooSmall);
            }
        }
        Err(IntegrateError::DidNotConverge { steps })
    }

    /// Integrates from `t0` to `tf` and returns the final state (no per-step callback).
    pub fn solve<const N: usize, F>(
        &self,
        f: &F,
        t0: T,
        y0: &Vector<N, T>,
        tf: T,
    ) -> Result<Vector<N, T>, IntegrateError>
    where
        F: Fn(T, &Vector<N, T>) -> Vector<N, T>,
    {
        self.for_each_step(f, t0, y0, tf, |_| {})
    }

    /// Samples the solution at each time in `times` (sorted in the integration direction and lying
    /// within `[t0, tf]`), writing to `out` via cubic-Hermite dense output. No allocation.
    ///
    /// # Errors
    /// [`LimitsIllDefined`](IntegrateError::LimitsIllDefined) if `times.len() !=
    /// out.len()` or a time is out of range / out of order; otherwise as
    /// [`for_each_step`](Rk45::for_each_step).
    ///
    /// ```
    /// use multicalc::ode::Rk45;
    /// use multicalc::linear_algebra::Vector;
    /// // Sample y' = -y at t = 0.5 and t = 1 by cubic-Hermite dense output.
    /// let times = [0.5, 1.0];
    /// let mut out = [Vector::<1, f64>::zeros(); 2];
    /// Rk45::default()
    ///     .solve_on_grid(&|_t, y: &Vector<1, f64>| -*y, 0.0, &Vector::new([1.0]), &times, &mut out)
    ///     .unwrap();
    /// assert!((out[0][0] - (-0.5_f64).exp()).abs() < 1e-6);
    /// assert!((out[1][0] - (-1.0_f64).exp()).abs() < 1e-6);
    /// ```
    pub fn solve_on_grid<const N: usize, F>(
        &self,
        f: &F,
        t0: T,
        y0: &Vector<N, T>,
        times: &[T],
        out: &mut [Vector<N, T>],
    ) -> Result<(), IntegrateError>
    where
        F: Fn(T, &Vector<N, T>) -> Vector<N, T>,
    {
        if times.len() != out.len() {
            return Err(IntegrateError::LimitsIllDefined);
        }
        if times.is_empty() {
            return Ok(());
        }
        let tf = times[times.len() - 1];
        let mut next = 0usize;
        let _ = self.for_each_step(f, t0, y0, tf, |step| {
            // Consume every requested time that falls in this accepted step (times are sorted).
            while next < times.len() {
                let tq = times[next];
                let in_step = if step.t1 >= step.t0 {
                    tq >= step.t0 && tq <= step.t1
                } else {
                    tq <= step.t0 && tq >= step.t1
                };
                if in_step {
                    out[next] = step.interpolate(tq);
                    next += 1;
                } else {
                    break;
                }
            }
        })?;
        if next != times.len() {
            // A requested time was out of range or out of order.
            return Err(IntegrateError::LimitsIllDefined);
        }
        Ok(())
    }
}
