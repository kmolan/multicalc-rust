//! Classic fixed-step fourth-order Runge–Kutta.

use crate::error::IntegrateError;
use crate::linear_algebra::Vector;
use crate::scalar::Numeric;

/// The classic fixed-step RK4 integrator for `state' = f(time, state)` with state `Vector<N, T>`.
pub struct Rk4;

impl Rk4 {
    /// Advances the state one step of size `timestep` from `(time, state)`.
    ///
    /// ```
    /// use multicalc::ode::Rk4;
    /// use multicalc::linear_algebra::Vector;
    /// // state' = state, state(0) = 1  ->  state(timestep) ≈ e^{timestep}
    /// let rate_of_change = |_time, state: &Vector<1, f64>| *state;
    /// let start_time = 0.0;
    /// let start_state = Vector::new([1.0]);
    /// let timestep = 0.1;
    ///
    /// let next = Rk4::step(&rate_of_change, start_time, &start_state, timestep);
    /// assert!((next[0] - 0.1_f64.exp()).abs() < 1e-6);
    /// ```
    pub fn step<const N: usize, T, F>(
        f: &F,
        time: T,
        state: &Vector<N, T>,
        timestep: T,
    ) -> Vector<N, T>
    where
        T: Numeric,
        F: Fn(T, &Vector<N, T>) -> Vector<N, T>,
    {
        let half = T::HALF * timestep;
        let stage1 = f(time, state);
        let stage2 = f(time + half, &(*state + stage1.scale(half)));
        let stage3 = f(time + half, &(*state + stage2.scale(half)));
        let stage4 = f(time + timestep, &(*state + stage3.scale(timestep)));
        let sixth = timestep / T::from_f64(6.0);
        *state + (stage1 + stage2.scale(T::TWO) + stage3.scale(T::TWO) + stage4).scale(sixth)
    }

    /// Integrates `steps` fixed steps of size `timestep` from `(time_start, state_start)`, invoking `observer`
    /// with each node (the initial node included) and returning the final state.
    ///
    /// ```
    /// use multicalc::ode::Rk4;
    /// use multicalc::linear_algebra::Vector;
    /// // state' = -state over [0, 1] in 100 steps; endpoint ≈ e^{-1}.
    /// let rate_of_change = |_time, state: &Vector<1, f64>| -*state;
    /// let start_time = 0.0;
    /// let start_state = Vector::new([1.0]);
    /// let timestep = 0.01;
    /// let step_count = 100;
    ///
    /// let mut last = 0.0;
    /// let final_state = Rk4::integrate(
    ///     &rate_of_change,
    ///     start_time,
    ///     &start_state,
    ///     timestep,
    ///     step_count,
    ///     |_time, state| last = state[0],
    /// );
    /// assert!((final_state[0] - (-1.0_f64).exp()).abs() < 1e-6);
    /// assert_eq!(last, final_state[0]);
    /// ```
    pub fn integrate<const N: usize, T, F, O>(
        f: &F,
        time_start: T,
        state_start: &Vector<N, T>,
        timestep: T,
        steps: usize,
        mut observer: O,
    ) -> Vector<N, T>
    where
        T: Numeric,
        F: Fn(T, &Vector<N, T>) -> Vector<N, T>,
        O: FnMut(T, &Vector<N, T>),
    {
        let mut time = time_start;
        let mut state = *state_start;
        observer(time, &state);
        for _ in 0..steps {
            state = Self::step(f, time, &state, timestep);
            time += timestep;
            observer(time, &state);
        }
        state
    }

    /// [`step`](Self::step), checked: returns [`IntegrateError::NonFinite`] if the new state is
    /// not finite. A non-finite stage always reaches the new state, so one check covers all four.
    ///
    /// ```
    /// use multicalc::error::IntegrateError;
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::ode::Rk4;
    ///
    /// let rate_of_change = |_time, state: &Vector<1, f64>| *state;
    /// let next = Rk4::try_step(&rate_of_change, 0.0, &Vector::new([1.0]), 0.1).unwrap();
    /// assert!((next[0] - 0.1_f64.exp()).abs() < 1e-6);
    ///
    /// let blown_up = Vector::new([f64::INFINITY]);
    /// assert_eq!(
    ///     Rk4::try_step(&rate_of_change, 0.0, &blown_up, 0.1),
    ///     Err(IntegrateError::NonFinite)
    /// );
    /// ```
    pub fn try_step<const N: usize, T, F>(
        f: &F,
        time: T,
        state: &Vector<N, T>,
        timestep: T,
    ) -> Result<Vector<N, T>, IntegrateError>
    where
        T: Numeric,
        F: Fn(T, &Vector<N, T>) -> Vector<N, T>,
    {
        let next = Self::step(f, time, state, timestep);
        if next.is_finite() {
            Ok(next)
        } else {
            Err(IntegrateError::NonFinite)
        }
    }

    /// [`integrate`](Self::integrate), checked: stops at the first state that is not finite and
    /// returns [`IntegrateError::NonFinite`]. `observer` sees only finite states.
    ///
    /// ```
    /// use multicalc::error::IntegrateError;
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::ode::Rk4;
    ///
    /// // x' = x², x(0) = 1 has the solution 1/(1 − t), which blows up at t = 1.
    /// let rate_of_change = |_time, state: &Vector<1, f64>| Vector::new([state[0] * state[0]]);
    /// let start = Vector::new([1.0]);
    /// let outcome = Rk4::try_integrate(&rate_of_change, 0.0, &start, 0.01, 200, |_, _| {});
    /// assert_eq!(outcome, Err(IntegrateError::NonFinite));
    /// ```
    pub fn try_integrate<const N: usize, T, F, O>(
        f: &F,
        time_start: T,
        state_start: &Vector<N, T>,
        timestep: T,
        steps: usize,
        mut observer: O,
    ) -> Result<Vector<N, T>, IntegrateError>
    where
        T: Numeric,
        F: Fn(T, &Vector<N, T>) -> Vector<N, T>,
        O: FnMut(T, &Vector<N, T>),
    {
        if !state_start.is_finite() {
            return Err(IntegrateError::NonFinite);
        }
        let mut time = time_start;
        let mut state = *state_start;
        observer(time, &state);
        for _ in 0..steps {
            state = Self::try_step(f, time, &state, timestep)?;
            time += timestep;
            observer(time, &state);
        }
        Ok(state)
    }
}
