//! Adaptive Dormand–Prince 5(4) with PI step control and cubic-Hermite dense output.

use crate::error::IntegrateError;
use crate::linear_algebra::Vector;
use crate::ode::tableau::*;
use crate::scalar::Numeric;

/// One accepted RK45 step, carrying the data for cubic-Hermite interpolation inside `[time_start, time_end]`.
#[derive(Debug, Clone, Copy)]
pub struct Step<const N: usize, T: Numeric = f64> {
    /// Step start time.
    pub time_start: T,
    /// Step end time.
    pub time_end: T,
    /// State at `time_start`.
    pub state_start: Vector<N, T>,
    /// State at `time_end`.
    pub state_end: Vector<N, T>,
    /// Derivative `f(time_start, state_start)`.
    pub derivative_start: Vector<N, T>,
    /// Derivative `f(time_end, state_end)`.
    pub derivative_end: Vector<N, T>,
}

impl<const N: usize, T: Numeric> Step<N, T> {
    /// Cubic-Hermite interpolation of the state at `time` in `[time_start, time_end]`. Returns `state_start` at `time_start`
    /// and `state_end` at `time_end` exactly.
    pub fn interpolate(&self, time: T) -> Vector<N, T> {
        let h = self.time_end - self.time_start;
        let frac = (time - self.time_start) / h; // normalized position in [0, 1]
        let frac2 = frac * frac;
        let frac3 = frac2 * frac;
        let three = T::from_f64(3.0);
        let h00 = T::TWO * frac3 - three * frac2 + T::ONE;
        let h10 = frac3 - T::TWO * frac2 + frac;
        let h01 = -T::TWO * frac3 + three * frac2;
        let h11 = frac3 - frac2;
        self.state_start.scale(h00)
            + self.derivative_start.scale(h10 * h)
            + self.state_end.scale(h01)
            + self.derivative_end.scale(h11 * h)
    }
}

/// Adaptive Dormand–Prince 5(4) integrator for `y' = f(time, y)` with state `Vector<N, T>`.
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
    /// ```compile_fail
    /// #![deny(unused_must_use)]
    /// use multicalc::ode::Rk45;
    /// Rk45::default().with_rtol(1e-8); //discarded builder result
    /// ```
    #[must_use]
    pub fn with_rtol(mut self, rtol: T) -> Self {
        self.rtol = rtol;
        self
    }
    /// Sets the absolute tolerance (default `1e-9`).
    #[must_use]
    pub fn with_atol(mut self, atol: T) -> Self {
        self.atol = atol;
        self
    }
    /// Sets the first step size; `0` (the default) auto-selects it.
    #[must_use]
    pub fn with_first_step(mut self, h: T) -> Self {
        self.first_step = h;
        self
    }
    /// Sets the minimum step magnitude; falling below it returns [`IntegrateError::StepSizeTooSmall`].
    /// `0` (the default) disables the floor.
    #[must_use]
    pub fn with_min_step(mut self, h: T) -> Self {
        self.min_step = h;
        self
    }
    /// Sets the maximum step magnitude (default unbounded).
    #[must_use]
    pub fn with_max_step(mut self, h: T) -> Self {
        self.max_step = h;
        self
    }
    /// Sets the maximum number of step attempts before [`IntegrateError::DidNotConverge`]
    /// (default `100_000`).
    #[must_use]
    pub fn with_max_steps(mut self, n: usize) -> Self {
        self.max_steps = n;
        self
    }
}

/// RMS of `err_i / (atol + rtol * max(|y0_i|, |y1_i|))` over the components.
/// Returns `T::ZERO` when `N == 0`.
#[must_use]
fn error_norm<const N: usize, T: Numeric>(
    err: &Vector<N, T>,
    state_start: &Vector<N, T>,
    state_end: &Vector<N, T>,
    atol: T,
    rtol: T,
) -> T {
    if N == 0 {
        return T::ZERO;
    }
    let mut sum = T::ZERO;
    for ((err_i, a), b) in err
        .as_array()
        .iter()
        .zip(state_start.as_array())
        .zip(state_end.as_array())
    {
        let scale = atol + rtol * a.abs().max(b.abs());
        let ratio = *err_i / scale;
        sum += ratio * ratio;
    }
    (sum / T::from_usize(N)).sqrt()
}

/// RMS of `v_i / (atol + rtol * |y_i|)` — used by the initial-step heuristic.
/// Returns `T::ZERO` when `N == 0`.
#[must_use]
fn scaled_norm<const N: usize, T: Numeric>(
    vector: &Vector<N, T>,
    y: &Vector<N, T>,
    atol: T,
    rtol: T,
) -> T {
    if N == 0 {
        return T::ZERO;
    }
    let mut sum = T::ZERO;
    for (err_i, a) in vector.as_array().iter().zip(y.as_array()) {
        let scale = atol + rtol * a.abs();
        let ratio = *err_i / scale;
        sum += ratio * ratio;
    }
    (sum / T::from_usize(N)).sqrt()
}

impl<T: Numeric> Rk45<T> {
    /// One Dormand–Prince 5(4) trial step of size `h` from `(time, y)`.
    ///
    /// `stage1` is `f(time, y)`, supplied by the caller so it can be reused from the previous
    /// accepted step (FSAL). Returns `(state5, err, stage7)`: the 5th-order state `state5`, the embedded
    /// error estimate `err = state5 − y4`, and `stage7 = f(time + h, state5)` (the next step's `stage1`).
    fn dopri_step<const N: usize, F>(
        &self,
        f: &F,
        time: T,
        y: &Vector<N, T>,
        h: T,
        stage1: Vector<N, T>,
    ) -> (Vector<N, T>, Vector<N, T>, Vector<N, T>)
    where
        F: Fn(T, &Vector<N, T>) -> Vector<N, T>,
    {
        let node2 = T::from_f64(NODE2);
        let node3 = T::from_f64(NODE3);
        let node4 = T::from_f64(NODE4);
        let node5 = T::from_f64(NODE5);
        // stage coefficients times h
        let a21 = T::from_f64(STAGE_A21) * h;
        let a31 = T::from_f64(STAGE_A31) * h;
        let a32 = T::from_f64(STAGE_A32) * h;
        let a41 = T::from_f64(STAGE_A41) * h;
        let a42 = T::from_f64(STAGE_A42) * h;
        let a43 = T::from_f64(STAGE_A43) * h;
        let a51 = T::from_f64(STAGE_A51) * h;
        let a52 = T::from_f64(STAGE_A52) * h;
        let a53 = T::from_f64(STAGE_A53) * h;
        let a54 = T::from_f64(STAGE_A54) * h;
        let a61 = T::from_f64(STAGE_A61) * h;
        let a62 = T::from_f64(STAGE_A62) * h;
        let a63 = T::from_f64(STAGE_A63) * h;
        let a64 = T::from_f64(STAGE_A64) * h;
        let a65 = T::from_f64(STAGE_A65) * h;

        let stage2 = f(time + node2 * h, &(*y + stage1.scale(a21)));
        let stage3 = f(
            time + node3 * h,
            &(*y + stage1.scale(a31) + stage2.scale(a32)),
        );
        let stage4 = f(
            time + node4 * h,
            &(*y + stage1.scale(a41) + stage2.scale(a42) + stage3.scale(a43)),
        );
        let stage5 = f(
            time + node5 * h,
            &(*y + stage1.scale(a51) + stage2.scale(a52) + stage3.scale(a53) + stage4.scale(a54)),
        );
        let stage6 = f(
            time + h,
            &(*y + stage1.scale(a61)
                + stage2.scale(a62)
                + stage3.scale(a63)
                + stage4.scale(a64)
                + stage5.scale(a65)),
        );

        let state5 = *y
            + (stage1.scale(T::from_f64(WEIGHT1))
                + stage3.scale(T::from_f64(WEIGHT3))
                + stage4.scale(T::from_f64(WEIGHT4))
                + stage5.scale(T::from_f64(WEIGHT5))
                + stage6.scale(T::from_f64(WEIGHT6)))
            .scale(h);
        let stage7 = f(time + h, &state5);
        let err = (stage1.scale(T::from_f64(ERROR1))
            + stage3.scale(T::from_f64(ERROR3))
            + stage4.scale(T::from_f64(ERROR4))
            + stage5.scale(T::from_f64(ERROR5))
            + stage6.scale(T::from_f64(ERROR6))
            + stage7.scale(T::from_f64(ERROR7)))
        .scale(h);
        (state5, err, stage7)
    }

    /// Picks the first step size, signed by `dir` (`+1` forward, `-1` backward).
    ///
    /// If `first_step` was set it is used directly (capped by `max_step` and `span`).
    /// Otherwise this is the Hairer–Wanner heuristic: size a tentative step from the
    /// scaled norms of `state_start` and `derivative_start`, take one explicit Euler probe to estimate the second
    /// derivative, then combine them for a step matched to the method order (5). `derivative_start` is
    /// `f(time_start, state_start)` and `span` is `|time_final − time_start|`; the result never exceeds `max_step` or `span`.
    #[must_use]
    fn select_initial_step<const N: usize, F>(
        &self,
        f: &F,
        time_start: T,
        state_start: &Vector<N, T>,
        derivative_start: &Vector<N, T>,
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
        let norm0 = scaled_norm(state_start, state_start, self.atol, self.rtol);
        let norm1 = scaled_norm(derivative_start, state_start, self.atol, self.rtol);
        let step0 = if norm0 < T::from_f64(1e-5) || norm1 < T::from_f64(1e-5) {
            T::from_f64(1e-6)
        } else {
            T::from_f64(0.01) * norm0 / norm1
        };
        let state_end = *state_start + derivative_start.scale(dir * step0);
        let derivative_end = f(time_start + dir * step0, &state_end);
        let norm2 = scaled_norm(
            &(derivative_end - *derivative_start),
            state_start,
            self.atol,
            self.rtol,
        ) / step0;
        // exponent 1/(p+1) with method order p = 5
        let step1 = if norm1.max(norm2) <= T::from_f64(1e-15) {
            (step0 * T::from_f64(1e-3)).max(T::from_f64(1e-6))
        } else {
            (T::from_f64(0.01) / norm1.max(norm2)).powf(T::ONE / T::from_f64(6.0))
        };
        let h = (T::from_f64(100.0) * step0)
            .min(step1)
            .min(span.min(self.max_step));
        dir * h
    }

    /// Integrates from `time_start` to `time_final`, invoking `obs` with each accepted [`Step`], and returns
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
    /// let rate_of_change = |_t, y: &Vector<1, f64>| -*y;
    /// let start_time = 0.0;
    /// let start_state = Vector::new([1.0]);
    /// let end_time = 2.0;
    ///
    /// let final_state = Rk45::default()
    ///     .solve(&rate_of_change, start_time, &start_state, end_time)
    ///     .unwrap();
    /// assert!((final_state[0] - (-2.0_f64).exp()).abs() < 1e-6);
    /// ```
    pub fn for_each_step<const N: usize, F, O>(
        &self,
        f: &F,
        time_start: T,
        state_start: &Vector<N, T>,
        time_final: T,
        mut obs: O,
    ) -> Result<Vector<N, T>, IntegrateError>
    where
        F: Fn(T, &Vector<N, T>) -> Vector<N, T>,
        O: FnMut(&Step<N, T>),
    {
        if !time_start.is_finite() || !time_final.is_finite() || time_start == time_final {
            return Err(IntegrateError::LimitsIllDefined);
        }
        let span = (time_final - time_start).abs();
        let dir = if time_final > time_start {
            T::ONE
        } else {
            -T::ONE
        };

        let mut time = time_start;
        let mut y = *state_start;
        let mut stage1 = f(time, &y);
        if !y.is_finite() || !stage1.is_finite() {
            return Err(IntegrateError::NonFinite);
        }
        let mut h = self.select_initial_step(f, time_start, &y, &stage1, dir, span);
        let mut err_prev = T::from_f64(1e-4);
        let mut kahan_c = T::ZERO;
        let mut steps = 0usize;

        for _ in 0..self.max_steps {
            steps += 1;
            // Do not overshoot time_final (compare signed remaining against signed h).
            let remaining = time_final - time;
            if h.abs() > remaining.abs() {
                h = remaining;
            }

            let (state5, err_vec, stage7) = self.dopri_step(f, time, &y, h, stage1);
            if !state5.is_finite() || !err_vec.is_finite() {
                return Err(IntegrateError::NonFinite);
            }
            let err = error_norm(&err_vec, &y, &state5, self.atol, self.rtol);
            let accept = err <= T::ONE;

            if accept {
                // Kahan-compensated time += h.
                let delta = h - kahan_c;
                let tnew = time + delta;
                kahan_c = (tnew - time) - delta;
                let step = Step {
                    time_start: time,
                    time_end: tnew,
                    state_start: y,
                    state_end: state5,
                    derivative_start: stage1,
                    derivative_end: stage7,
                };
                obs(&step);
                time = tnew;
                y = state5;
                stage1 = stage7; // FSAL
                if (time_final - time).abs() <= T::EPSILON * (T::ONE + time_final.abs()) {
                    return Ok(y);
                }
            }

            // PI step-size update (uses err and the previous accepted err).
            let err_i = err.max(T::from_f64(1e-10));
            let factor = T::from_f64(0.9)
                * err_i.powf(-T::from_f64(0.17))
                * err_prev.powf(T::from_f64(0.04));
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

    /// Integrates from `time_start` to `time_final` and returns the final state (no per-step callback).
    pub fn solve<const N: usize, F>(
        &self,
        f: &F,
        time_start: T,
        state_start: &Vector<N, T>,
        time_final: T,
    ) -> Result<Vector<N, T>, IntegrateError>
    where
        F: Fn(T, &Vector<N, T>) -> Vector<N, T>,
    {
        self.for_each_step(f, time_start, state_start, time_final, |_| {})
    }

    /// Samples the solution at each time in `times` (sorted in the integration direction and lying
    /// within `[time_start, time_final]`), writing to `out` via cubic-Hermite dense output. No allocation.
    ///
    /// # Errors
    /// [`LimitsIllDefined`](IntegrateError::LimitsIllDefined) if `times.len() !=
    /// out.len()` or a time is out of range / out of order; otherwise as
    /// [`for_each_step`](Rk45::for_each_step).
    ///
    /// ```
    /// use multicalc::ode::Rk45;
    /// use multicalc::linear_algebra::Vector;
    /// // Sample y' = -y at time = 0.5 and time = 1 by cubic-Hermite dense output.
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
        time_start: T,
        state_start: &Vector<N, T>,
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
        let time_final = times[times.len() - 1];
        let mut next = 0usize;
        let _ = self.for_each_step(f, time_start, state_start, time_final, |step| {
            // Consume every requested time that falls in this accepted step (times are sorted).
            while next < times.len() {
                let time_query = times[next];
                let in_step = if step.time_end >= step.time_start {
                    time_query >= step.time_start && time_query <= step.time_end
                } else {
                    time_query <= step.time_start && time_query >= step.time_end
                };
                if in_step {
                    out[next] = step.interpolate(time_query);
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
