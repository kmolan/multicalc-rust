//! Classic fixed-step fourth-order Runge–Kutta.

use crate::error::IntegrateError;
use crate::linear_algebra::Vector;
use crate::scalar::Numeric;

/// The classic fixed-step RK4 integrator for `y' = f(t, y)` with state `Vector<N, T>`.
pub struct Rk4;

impl Rk4 {
    /// Advances the state one step of size `dt` from `(t, y)`.
    ///
    /// Returns [`IntegrateError::NonFinite`] if the state or any stage derivative
    /// is non-finite (matching [`Rk45`](crate::ode::Rk45)'s NaN policy).
    ///
    /// ```
    /// use multicalc::ode::Rk4;
    /// use multicalc::linear_algebra::Vector;
    /// // y' = y, y(0) = 1  ->  y(dt) ≈ e^{dt}
    /// let y1 = Rk4::step(&|_t, y: &Vector<1, f64>| *y, 0.0, &Vector::new([1.0]), 0.1).unwrap();
    /// assert!((y1[0] - 0.1_f64.exp()).abs() < 1e-6);
    /// ```
    pub fn step<const N: usize, T, F>(
        f: &F,
        t: T,
        y: &Vector<N, T>,
        dt: T,
    ) -> Result<Vector<N, T>, IntegrateError>
    where
        T: Numeric,
        F: Fn(T, &Vector<N, T>) -> Vector<N, T>,
    {
        if !y.is_finite() || !t.is_finite() || !dt.is_finite() {
            return Err(IntegrateError::NonFinite);
        }
        let half = T::HALF * dt;
        let k1 = f(t, y);
        if !k1.is_finite() {
            return Err(IntegrateError::NonFinite);
        }
        let k2 = f(t + half, &(*y + k1.scale(half)));
        if !k2.is_finite() {
            return Err(IntegrateError::NonFinite);
        }
        let k3 = f(t + half, &(*y + k2.scale(half)));
        if !k3.is_finite() {
            return Err(IntegrateError::NonFinite);
        }
        let k4 = f(t + dt, &(*y + k3.scale(dt)));
        if !k4.is_finite() {
            return Err(IntegrateError::NonFinite);
        }
        let sixth = dt / T::from_f64(6.0);
        let next = *y + (k1 + k2.scale(T::TWO) + k3.scale(T::TWO) + k4).scale(sixth);
        if !next.is_finite() {
            return Err(IntegrateError::NonFinite);
        }
        Ok(next)
    }

    /// Integrates `steps` fixed steps of size `dt` from `(t0, y0)`, invoking `observer`
    /// with each node (the initial node included) and returning the final state.
    ///
    /// Returns [`IntegrateError::NonFinite`] if any accepted state or stage goes non-finite.
    ///
    /// ```
    /// use multicalc::ode::Rk4;
    /// use multicalc::linear_algebra::Vector;
    /// let mut last = 0.0;
    /// // y' = -y over [0, 1] in 100 steps; endpoint ≈ e^{-1}.
    /// let yf = Rk4::integrate(&|_t, y: &Vector<1, f64>| -*y, 0.0,
    ///     &Vector::new([1.0]), 0.01, 100, |_t, y| last = y[0]).unwrap();
    /// assert!((yf[0] - (-1.0_f64).exp()).abs() < 1e-6);
    /// assert_eq!(last, yf[0]);
    /// ```
    pub fn integrate<const N: usize, T, F, O>(
        f: &F,
        t0: T,
        y0: &Vector<N, T>,
        dt: T,
        steps: usize,
        mut observer: O,
    ) -> Result<Vector<N, T>, IntegrateError>
    where
        T: Numeric,
        F: Fn(T, &Vector<N, T>) -> Vector<N, T>,
        O: FnMut(T, &Vector<N, T>),
    {
        if !y0.is_finite() || !t0.is_finite() || !dt.is_finite() {
            return Err(IntegrateError::NonFinite);
        }
        let mut t = t0;
        let mut y = *y0;
        observer(t, &y);
        for _ in 0..steps {
            y = Self::step(f, t, &y, dt)?;
            t += dt;
            if !t.is_finite() {
                return Err(IntegrateError::NonFinite);
            }
            observer(t, &y);
        }
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_algebra::Vector;

    #[test]
    fn step_rejects_non_finite_rhs() {
        let res = Rk4::step(
            &|_t, _y: &Vector<1, f64>| Vector::new([f64::NAN]),
            0.0,
            &Vector::new([1.0]),
            0.1,
        );
        assert_eq!(res.unwrap_err(), IntegrateError::NonFinite);
    }

    #[test]
    fn integrate_rejects_diverging_blow_up() {
        // y' = y^2 blows up in finite time from y(0)=1; a large step hits Inf/NaN.
        let res = Rk4::integrate(
            &|_t, y: &Vector<1, f64>| Vector::new([y[0] * y[0]]),
            0.0,
            &Vector::new([1.0]),
            1.0,
            20,
            |_t, _y| {},
        );
        assert_eq!(res.unwrap_err(), IntegrateError::NonFinite);
    }
}
