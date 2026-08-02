//! A single body free to move in all six directions, and how it responds to the forces on it.

use crate::error::DynamicsError;
use crate::linear_algebra::{Matrix3D, Vector3D};
use crate::scalar::Numeric;
use crate::spatial::SpatialInertia;

/// How many numbers it takes to say where a free body is and how it is moving.
/// index  0  1  2   3   4   5   6   7  8  9   10 11 12
///        px py pz  qw  qx  qy  qz  vx vy vz  ωx ωy ωz
/// p — position in world frame
/// q — orientation in body frame
/// v — linear velocity in world frame
/// ω — angular velocity in body frame
pub const STATE_DIMENSION: usize = 13;

/// A body free to move in all six directions, and what the forces on it do to it.
///
/// Give it the body's mass and how that mass is spread out, plus the pull of gravity, and it
/// answers the question an integrator keeps asking: given where the body is pointing, how fast it
/// is turning, and what is pushing on it, how quickly is its motion changing? The work needed
/// every time — inverting the body's resistance to spinning — is done once here, so each later
/// call is a fixed handful of small products with nothing that can fail.
///
/// Straight-line motion is measured in world axes; turning, and the forces applied to the body,
/// are in the body's own axes.
///
/// ```
/// use multicalc::dynamics::RigidBody;
/// use multicalc::linear_algebra::Vector;
/// use multicalc::spatial::SpatialInertia;
///
/// let inertia = SpatialInertia::from_diagonal_inertia(
///     0.8_f64,
///     Vector::new([0.0, 0.0, 0.0]),
///     Vector::new([0.005, 0.007, 0.009]),
/// )
/// .unwrap();
/// let body = RigidBody::new(inertia, Vector::new([0.0, 0.0, -9.81])).unwrap();
///
/// assert_eq!(body.inertia().mass(), 0.8);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigidBody<T: Numeric = f64> {
    inertia: SpatialInertia<T>,
    inverse_rotational_inertia: Matrix3D<T>,
    gravity: Vector3D<T>,
}

impl<T: Numeric> RigidBody<T> {
    /// Builds a body from how its mass is spread out and the pull of gravity.
    ///
    /// `gravity` is the acceleration gravity gives the body, in world axes — on Earth with the
    /// world's z axis pointing up, that is `[0, 0, -9.81]`.
    ///
    /// Returns [`DynamicsError::NonFinite`] if any value is not finite,
    /// [`DynamicsError::NonPositiveInertia`] if the body's resistance to spinning is not positive
    /// definite, or [`DynamicsError::Linalg`] if it cannot be inverted.
    pub fn new(inertia: SpatialInertia<T>, gravity: Vector3D<T>) -> Result<Self, DynamicsError> {
        if !inertia.is_finite() || !gravity.is_finite() {
            return Err(DynamicsError::NonFinite);
        }
        let rotational_inertia = inertia.rotational_inertia();
        if rotational_inertia.cholesky().is_err() {
            return Err(DynamicsError::NonPositiveInertia);
        }
        let inverse_rotational_inertia = rotational_inertia.inverse()?;
        Ok(RigidBody {
            inertia,
            inverse_rotational_inertia,
            gravity,
        })
    }

    /// How the body's mass is spread out.
    #[inline]
    #[must_use]
    pub fn inertia(self) -> SpatialInertia<T> {
        self.inertia
    }

    /// The acceleration gravity gives the body, in world axes.
    #[inline]
    pub fn gravity(self) -> Vector3D<T> {
        self.gravity
    }
}
