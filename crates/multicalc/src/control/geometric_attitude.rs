//! Attitude control for a rigid body, worked directly on rotations rather than on angles.

use crate::control::LimitedOutput;
use crate::error::ControlError;
use crate::linear_algebra::{Matrix3D, Vector3D};
use crate::scalar::Numeric;
use crate::spatial::SO3;

/// Turns the difference between where a body is pointing and where it should be pointing into the
/// torque that closes the gap.
///
/// Working on rotations directly, rather than on three angles, means there is no orientation the
/// law breaks down at and no wrap-around to handle. Give it two gains — one on how far off the
/// pointing is, one on how far off the turn rate is — and the body's resistance to spinning. Every
/// call is a fixed handful of small matrix products: bounded work, no allocation, and cheap enough
/// for the fastest loop in a flight stack.
///
/// The law also cancels the body's own gyroscopic torque and follows the reference's turn rate, so
/// it tracks a moving target rather than only holding still.
///
/// [`update`](Self::update) adds an integral term and per-axis torque limits to the law. The type
/// is `Copy`, so a copy also copies the integral.
///
/// Every operation is generic over [`Numeric`](crate::Numeric).
///
/// ```
/// use multicalc::control::GeometricAttitudeController;
/// use multicalc::linear_algebra::{Matrix, Vector};
/// use multicalc::SO3;
///
/// let inertia = Matrix::<3, 3>::from_diagonal([0.02, 0.02, 0.04]);
/// let controller = GeometricAttitudeController::new(6.0, 1.2, inertia).unwrap();
///
/// // Already pointing the right way and not turning: nothing to do.
/// let level = SO3::<f64>::identity();
/// let still = Vector::new([0.0, 0.0, 0.0]);
/// let torque = controller.torque(level, still, level, still, still);
/// assert!(torque.norm() < 1e-15);
///
/// // Tipped a little about x, and still: the torque pushes back the other way.
/// let tipped = SO3::exp(Vector::new([0.1, 0.0, 0.0]));
/// let torque = controller.torque(tipped, still, level, still, still);
/// assert!(torque[0] < 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeometricAttitudeController<T: Numeric = f64> {
    attitude_gain: T,
    rate_gain: T,
    inertia: Matrix3D<T>,
    integral_gain: T,
    integral_limit: Vector3D<T>,
    torque_limit: Vector3D<T>,
    integral: Vector3D<T>,
    timestep: T,
}

impl<T: Numeric> GeometricAttitudeController<T> {
    /// Builds a controller from the two gains and the body's resistance to spinning.
    ///
    /// `attitude_gain` sets how hard the pointing error is pushed on, `rate_gain` how hard the turn
    /// rate error is; both have to be strictly positive. `inertia` is measured about the body's
    /// balance point, in the body's own frame.
    ///
    /// Returns [`ControlError::NonFinite`] if any value is not finite,
    /// [`ControlError::NonPositiveGain`] if either gain is zero or negative,
    /// [`ControlError::NotSymmetricInertia`] if the inertia does not read the same across the
    /// diagonal, or [`ControlError::NonPositiveInertia`] if it is not positive definite.
    pub fn new(attitude_gain: T, rate_gain: T, inertia: Matrix3D<T>) -> Result<Self, ControlError> {
        if !attitude_gain.is_finite() || !rate_gain.is_finite() || !inertia.is_finite() {
            return Err(ControlError::NonFinite);
        }
        if attitude_gain <= T::ZERO || rate_gain <= T::ZERO {
            return Err(ControlError::NonPositiveGain);
        }
        if !inertia.is_symmetric() {
            return Err(ControlError::NotSymmetricInertia);
        }
        if inertia.cholesky().is_err() {
            return Err(ControlError::NonPositiveInertia);
        }
        Ok(Self {
            attitude_gain,
            rate_gain,
            inertia,
            integral_gain: T::ZERO,
            integral_limit: Vector3D::from_fn(|_| T::INFINITY),
            torque_limit: Vector3D::from_fn(|_| T::INFINITY),
            integral: Vector3D::zeros(),
            timestep: T::ZERO,
        })
    }

    /// The attitude error `e = 2·sgn(w)·q_v` of `q = q_dᶜ ⊗ q_b`, with `sgn(0) = +1`.
    ///
    /// `|e| = 2·sin(θ/2)` for the shorter rotation angle `θ ∈ [0, π]`, so `e` agrees with the
    /// rotation vector to first order. `e` is largest at a half turn, where the sign picks one of
    /// the two equal paths.
    pub fn attitude_error(attitude: SO3<T>, desired_attitude: SO3<T>) -> Vector3D<T> {
        let error_quaternion = desired_attitude.quaternion().conjugate() * attitude.quaternion();
        let doubled = if error_quaternion.w() >= T::ZERO {
            T::TWO
        } else {
            -T::TWO
        };
        error_quaternion.vec().scale(doubled)
    }

    /// Returns the torque to apply to the body, in the body's own frame.
    ///
    /// `attitude` and `body_rate` are where the body is pointing and how fast it is turning;
    /// `desired_attitude`, `desired_body_rate`, and `desired_body_rate_derivative` are the same
    /// three things for the target, with the two rates given in the target's own frame.
    pub fn torque(
        &self,
        attitude: SO3<T>,
        body_rate: Vector3D<T>,
        desired_attitude: SO3<T>,
        desired_body_rate: Vector3D<T>,
        desired_body_rate_derivative: Vector3D<T>,
    ) -> Vector3D<T> {
        let relative = attitude.to_matrix().transpose() * desired_attitude.to_matrix();
        let attitude_error = Self::attitude_error(attitude, desired_attitude);

        // The target's turn rate and its change, both read in the body's own frame.
        let carried_rate = relative * desired_body_rate;
        let carried_rate_change = relative * desired_body_rate_derivative;
        let rate_error = body_rate - carried_rate;

        // The body's own spin fights any change of axis; adding it back cancels it.
        let spin_resistance = body_rate.cross(self.inertia * body_rate);
        // Following the target's turn takes torque of its own, which the first part accounts for
        // as the frame the target's rate is read in keeps moving.
        let following = self.inertia * (body_rate.cross(carried_rate) - carried_rate_change);

        attitude_error.scale(-self.attitude_gain) - rate_error.scale(self.rate_gain)
            + spin_resistance
            - following
    }

    /// Adds `−k_I·∫e dt` to [`update`](Self::update), with each axis of `∫e dt` clamped to
    /// `±integral_limit[axis]`. `timestep` is the period of `update`.
    ///
    /// Returns [`ControlError::NonFinite`] if `integral_gain` or `timestep` is not finite or a
    /// limit is NaN, [`ControlError::NonPositiveGain`] if `integral_gain <= 0`,
    /// [`ControlError::NonPositiveTimestep`] if `timestep <= 0`, or
    /// [`ControlError::InvalidOutputLimits`] if a limit is negative.
    pub fn with_integral(
        mut self,
        integral_gain: T,
        integral_limit: Vector3D<T>,
        timestep: T,
    ) -> Result<Self, ControlError> {
        if !integral_gain.is_finite()
            || !timestep.is_finite()
            || (0..3).any(|axis| integral_limit[axis].is_nan())
        {
            return Err(ControlError::NonFinite);
        }
        if integral_gain <= T::ZERO {
            return Err(ControlError::NonPositiveGain);
        }
        if timestep <= T::ZERO {
            return Err(ControlError::NonPositiveTimestep);
        }
        if (0..3).any(|axis| integral_limit[axis] < T::ZERO) {
            return Err(ControlError::InvalidOutputLimits);
        }
        self.integral_gain = integral_gain;
        self.integral_limit = integral_limit;
        self.timestep = timestep;
        Ok(self)
    }

    /// Sets the per-axis limit `|τ[axis]| ≤ torque_limit[axis]` for [`update`](Self::update). An
    /// infinite limit means no limit on that axis. Defaults to no limits.
    ///
    /// Returns [`ControlError::NonFinite`] if a limit is NaN, or
    /// [`ControlError::InvalidOutputLimits`] if a limit is negative.
    pub fn with_torque_limits(mut self, torque_limit: Vector3D<T>) -> Result<Self, ControlError> {
        if (0..3).any(|axis| torque_limit[axis].is_nan()) {
            return Err(ControlError::NonFinite);
        }
        if (0..3).any(|axis| torque_limit[axis] < T::ZERO) {
            return Err(ControlError::InvalidOutputLimits);
        }
        self.torque_limit = torque_limit;
        Ok(self)
    }

    /// [`torque`](Self::torque) plus `−k_I·∫e dt`, clamped per axis to the torque limits.
    ///
    /// Anti-windup is by conditional integration, as in [`Pid`](crate::control::Pid): a limited
    /// axis holds its integral while the error drives it further into the limit. Without
    /// [`with_integral`](Self::with_integral) and
    /// [`with_torque_limits`](Self::with_torque_limits), the output equals `torque` and no axis is
    /// limited.
    ///
    /// ```
    /// use multicalc::control::GeometricAttitudeController;
    /// use multicalc::linear_algebra::{Matrix, Vector};
    /// use multicalc::SO3;
    ///
    /// let inertia = Matrix::<3, 3>::from_diagonal([0.02, 0.02, 0.04]);
    /// let mut controller = GeometricAttitudeController::new(6.0, 1.2, inertia)
    ///     .unwrap()
    ///     .with_integral(4.0, Vector::new([0.05, 0.05, 0.05]), 0.002)
    ///     .unwrap()
    ///     .with_torque_limits(Vector::new([0.15, 0.15, 0.075]))
    ///     .unwrap();
    ///
    /// let inverted = SO3::exp(Vector::new([3.0, 0.0, 0.0]));
    /// let still = Vector::new([0.0, 0.0, 0.0]);
    /// let command = controller.update(inverted, still, SO3::identity(), still, still);
    /// assert_eq!(command.limited(), [true, false, false]);
    /// assert!(command.output()[0].abs() <= 0.15);
    /// ```
    pub fn update(
        &mut self,
        attitude: SO3<T>,
        body_rate: Vector3D<T>,
        desired_attitude: SO3<T>,
        desired_body_rate: Vector3D<T>,
        desired_body_rate_derivative: Vector3D<T>,
    ) -> LimitedOutput<3, T> {
        let error = Self::attitude_error(attitude, desired_attitude);
        let proportional_derivative = self.torque(
            attitude,
            body_rate,
            desired_attitude,
            desired_body_rate,
            desired_body_rate_derivative,
        );
        let candidate = Vector3D::from_fn(|axis| {
            (self.integral[axis] + error[axis] * self.timestep)
                .max(-self.integral_limit[axis])
                .min(self.integral_limit[axis])
        });
        let unlimited = proportional_derivative - candidate.scale(self.integral_gain);
        let command = LimitedOutput::clamp(unlimited, -self.torque_limit, self.torque_limit);
        for axis in 0..3 {
            // The integral enters as −k_I·∫e, so a positive error lowers the torque.
            let deeper = (unlimited[axis] > self.torque_limit[axis] && error[axis] < T::ZERO)
                || (unlimited[axis] < -self.torque_limit[axis] && error[axis] > T::ZERO);
            if !deeper {
                self.integral[axis] = candidate[axis];
            }
        }
        command
    }

    /// Clears the integral.
    pub fn reset(&mut self) {
        self.integral = Vector3D::zeros();
    }

    /// Returns `∫e dt`, the integral state of [`update`](Self::update).
    #[inline]
    pub fn integral(&self) -> Vector3D<T> {
        self.integral
    }

    /// Returns the gain on `∫e dt`.
    #[inline]
    #[must_use]
    pub fn integral_gain(&self) -> T {
        self.integral_gain
    }

    /// Returns the gain on how far off the pointing is.
    #[inline]
    #[must_use]
    pub fn attitude_gain(&self) -> T {
        self.attitude_gain
    }

    /// Returns the gain on how far off the turn rate is.
    #[inline]
    #[must_use]
    pub fn rate_gain(&self) -> T {
        self.rate_gain
    }

    /// Returns the body's resistance to spinning.
    #[inline]
    pub fn inertia(&self) -> Matrix3D<T> {
        self.inertia
    }
}
