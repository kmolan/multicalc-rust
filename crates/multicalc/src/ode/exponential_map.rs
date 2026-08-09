//! Turning an orientation forward in time by the turn it makes over the step.

use crate::error::IntegrateError;
use crate::linear_algebra::Vector3D;
use crate::scalar::Numeric;
use crate::spatial::SO3;

/// Steps an orientation forward by composing on the turn it makes over the step, so the answer
/// stays a true rotation instead of drifting off and needing to be scaled back.
pub struct ExponentialMap;

impl ExponentialMap {
    /// Turns an orientation forward one step of size `dt` at a steady turn rate.
    ///
    /// `angular_rate` is how fast the body is turning, about its own axes, in radians per second.
    /// The step works out the whole turn made over the step and composes it onto the orientation,
    /// so what comes back is still a true rotation to within rounding — there is nothing to scale
    /// back afterwards, which is what happens instead when four loose orientation numbers are
    /// stepped and drift off unit length.
    ///
    /// The turn rate is taken as steady across the step, so the error shrinks in proportion to the
    /// step size. That is first order: coarser than [`Rk4`](crate::ode::Rk4), which shrinks with
    /// the fourth power. Use [`ExponentialMap::attitude_step_with_angular_acceleration`] when the
    /// rate is changing and the extra accuracy is wanted for the same one exponential.
    ///
    /// Behavior: this is infallible and does not validate its input. A non-finite `angular_rate`
    /// or a non-finite/negative `dt` produces a non-finite result rather than an error; callers
    /// that need validated input can use [`ExponentialMap::integrate_attitude`] instead, or
    /// validate upstream - this stays a raw, panic-free primitive for hot per-tick call sites.
    ///
    /// ```
    /// use multicalc::ode::ExponentialMap;
    /// use multicalc::spatial::SO3;
    /// use multicalc::linear_algebra::Vector;
    /// // Turning at 2 rad/s about its own z axis, stepped twice by an eighth of a turn each time.
    /// let turning_about_z = Vector::new([0.0, 0.0, 2.0]);
    /// let timestep = core::f64::consts::FRAC_PI_8;
    /// let mut orientation = SO3::<f64>::identity();
    /// for _ in 0..2 {
    ///     orientation = ExponentialMap::attitude_step(orientation, turning_about_z, timestep);
    /// }
    ///
    /// // A quarter turn in all: the x axis has swung onto the y axis.
    /// let swung = orientation.act(Vector::new([1.0, 0.0, 0.0]));
    /// assert!(swung[0].abs() < 1e-12);
    /// assert!((swung[1] - 1.0).abs() < 1e-12);
    /// // And the orientation is still a true rotation.
    /// assert!((orientation.quaternion().norm() - 1.0).abs() < 1e-15);
    /// ```
    #[inline]
    #[must_use]
    pub fn attitude_step<T: Numeric>(
        orientation: SO3<T>,
        angular_rate: Vector3D<T>,
        dt: T,
    ) -> SO3<T> {
        orientation * SO3::exp(angular_rate * dt)
    }

    /// Turns an orientation forward one step of size `dt` when the turn rate is itself changing.
    ///
    /// `angular_rate` is how fast the body is turning and `angular_acceleration` how fast that is
    /// changing, both about the body's own axes. The step uses the rate half way through rather
    /// than the rate at the start, which costs one extra add and no extra exponential, and shrinks
    /// the error with the square of the step size instead of in proportion to it. The result is
    /// still a true rotation to within rounding.
    ///
    /// Behavior: this is infallible and does not validate its input; see the "Behavior" note on
    /// [`ExponentialMap::attitude_step`] for how non-finite or negative input is handled.
    ///
    /// ```
    /// use multicalc::ode::ExponentialMap;
    /// use multicalc::spatial::SO3;
    /// use multicalc::linear_algebra::Vector;
    /// // Starting from rest and picking up spin about z at 4 rad/s^2 for half a second: the turn
    /// // made is 0.5 * 4 * 0.5^2 = 0.5 rad.
    /// let at_rest = Vector::new([0.0, 0.0, 0.0]);
    /// let picking_up_spin = Vector::new([0.0, 0.0, 4.0]);
    /// let timestep = 0.5;
    ///
    /// let orientation = ExponentialMap::attitude_step_with_angular_acceleration(
    ///     SO3::<f64>::identity(), at_rest, picking_up_spin, timestep,
    /// );
    /// assert!((orientation.log()[2] - 0.5).abs() < 1e-12);
    /// ```
    #[inline]
    #[must_use]
    pub fn attitude_step_with_angular_acceleration<T: Numeric>(
        orientation: SO3<T>,
        angular_rate: Vector3D<T>,
        angular_acceleration: Vector3D<T>,
        dt: T,
    ) -> SO3<T> {
        let half_way_rate = angular_rate + angular_acceleration * (dt * T::HALF);
        orientation * SO3::exp(half_way_rate * dt)
    }

    /// Runs `steps` fixed steps of size `dt`, asking `angular_rate_at` how fast the body is turning
    /// as it goes, and hands each node to `observer` — the starting one included — before returning
    /// where the body ends up facing.
    ///
    /// Each step asks for the turn rate twice, once at the start and once half a step in, and turns
    /// the orientation forward with the half-way rate. That makes the error shrink with the square
    /// of the step size, without the caller having to work out how fast the turn rate is changing.
    /// The orientation stays a true rotation to within rounding the whole way through.
    ///
    /// # Errors
    ///
    /// [`IntegrateError::NonPositiveTimestep`] if `dt` is not strictly positive, or
    /// [`IntegrateError::NonFinite`] if `dt` or a rate returned by `angular_rate_at` is not
    /// finite.
    ///
    /// ```
    /// use multicalc::ode::ExponentialMap;
    /// use multicalc::spatial::SO3;
    /// use multicalc::linear_algebra::{Vector, Vector3D};
    /// // A steady quarter turn about z, split into a hundred steps.
    /// let steady = |_time: f64, _orientation: SO3<f64>| Vector::new([0.0, 0.0, 1.0]);
    /// let steps = 100;
    /// let timestep = core::f64::consts::FRAC_PI_2 / steps as f64;
    ///
    /// let mut nodes = 0;
    /// let facing = ExponentialMap::integrate_attitude(
    ///     &steady, 0.0, SO3::identity(), timestep, steps, |_time, _orientation| nodes += 1,
    /// ).unwrap();
    /// assert_eq!(nodes, steps + 1);
    ///
    /// let swung: Vector3D<f64> = facing.act(Vector::new([1.0, 0.0, 0.0]));
    /// assert!((swung[1] - 1.0).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn integrate_attitude<T, F, O>(
        angular_rate_at: &F,
        t0: T,
        start_orientation: SO3<T>,
        dt: T,
        steps: usize,
        mut observer: O,
    ) -> Result<SO3<T>, IntegrateError>
    where
        T: Numeric,
        F: Fn(T, SO3<T>) -> Vector3D<T>,
        O: FnMut(T, SO3<T>),
    {
        if !dt.is_finite() {
            return Err(IntegrateError::NonFinite);
        }
        if dt <= T::ZERO {
            return Err(IntegrateError::NonPositiveTimestep);
        }
        let half = dt * T::HALF;
        let mut time = t0;
        let mut orientation = start_orientation;
        observer(time, orientation);
        for _ in 0..steps {
            let rate_at_start = angular_rate_at(time, orientation);
            if !rate_at_start.is_finite() {
                return Err(IntegrateError::NonFinite);
            }
            let half_way = Self::attitude_step(orientation, rate_at_start, half);
            let half_way_rate = angular_rate_at(time + half, half_way);
            if !half_way_rate.is_finite() {
                return Err(IntegrateError::NonFinite);
            }
            orientation = Self::attitude_step(orientation, half_way_rate, dt);
            time += dt;
            observer(time, orientation);
        }
        Ok(orientation)
    }
}
