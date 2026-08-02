//! Attitude control for a rigid body, worked directly on rotations rather than on angles.

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
        })
    }

    /// Returns how far the body is from pointing the way it should, as a small rotation.
    ///
    /// The vector says which way the body is turned away from where it should be: it points along
    /// the axis of that turn, and its length grows with how far off it is. The torque pushes back
    /// the other way. The length shrinks again past a quarter turn and reaches zero at a half turn,
    /// which
    /// is a balance point the law cannot push away from — a body started exactly upside down needs
    /// a nudge before this closes the gap.
    pub fn attitude_error(attitude: SO3<T>, desired_attitude: SO3<T>) -> Vector3D<T> {
        let relative = attitude.to_matrix().transpose() * desired_attitude.to_matrix();
        SO3::vee(relative.transpose() - relative).scale(T::HALF)
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
        let attitude_error = SO3::vee(relative.transpose() - relative).scale(T::HALF);

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
