//! Classic fixed-step fourth-order Runge–Kutta.

use crate::linear_algebra::Vector;
use crate::scalar::Numeric;

/// The classic fixed-step RK4 integrator for `y' = f(t, y)` with state `Vector<N, T>`.
pub struct Rk4;

impl Rk4 {
    /// Advances the state one step of size `dt` from `(t, y)`.
    ///
    /// ```
    /// use multicalc::ode::Rk4;
    /// use multicalc::linear_algebra::Vector;
    /// // y' = y, y(0) = 1  ->  y(timestep) ≈ e^{timestep}
    /// let rate_of_change = |_t, y: &Vector<1, f64>| *y;
    /// let start_time = 0.0;
    /// let start_state = Vector::new([1.0]);
    /// let timestep = 0.1;
    ///
    /// let next = Rk4::step(&rate_of_change, start_time, &start_state, timestep);
    /// assert!((next[0] - 0.1_f64.exp()).abs() < 1e-6);
    /// ```
    pub fn step<const N: usize, T, F>(f: &F, t: T, y: &Vector<N, T>, dt: T) -> Vector<N, T>
    where
        T: Numeric,
        F: Fn(T, &Vector<N, T>) -> Vector<N, T>,
    {
        let half = T::HALF * dt;
        let k1 = f(t, y);
        let k2 = f(t + half, &(*y + k1.scale(half)));
        let k3 = f(t + half, &(*y + k2.scale(half)));
        let k4 = f(t + dt, &(*y + k3.scale(dt)));
        let sixth = dt / T::from_f64(6.0);
        *y + (k1 + k2.scale(T::TWO) + k3.scale(T::TWO) + k4).scale(sixth)
    }

    /// Integrates `steps` fixed steps of size `dt` from `(t0, y0)`, invoking `observer`
    /// with each node (the initial node included) and returning the final state.
    ///
    /// ```
    /// use multicalc::ode::Rk4;
    /// use multicalc::linear_algebra::Vector;
    /// // y' = -y over [0, 1] in 100 steps; endpoint ≈ e^{-1}.
    /// let rate_of_change = |_t, y: &Vector<1, f64>| -*y;
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
    ///     |_t, y| last = y[0],
    /// );
    /// assert!((final_state[0] - (-1.0_f64).exp()).abs() < 1e-6);
    /// assert_eq!(last, final_state[0]);
    /// ```
    pub fn integrate<const N: usize, T, F, O>(
        f: &F,
        t0: T,
        y0: &Vector<N, T>,
        dt: T,
        steps: usize,
        mut observer: O,
    ) -> Vector<N, T>
    where
        T: Numeric,
        F: Fn(T, &Vector<N, T>) -> Vector<N, T>,
        O: FnMut(T, &Vector<N, T>),
    {
        let mut t = t0;
        let mut y = *y0;
        observer(t, &y);
        for _ in 0..steps {
            y = Self::step(f, t, &y, dt);
            t += dt;
            observer(t, &y);
        }
        y
    }
}
