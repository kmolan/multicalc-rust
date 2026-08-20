//! Classic fixed-step fourth-order Runge–Kutta.

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
}
