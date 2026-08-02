//! A single body free to move in all six directions, and how it responds to the forces on it.

use crate::error::DynamicsError;
use crate::linear_algebra::{Matrix3D, Vector3D};
use crate::scalar::Numeric;
use crate::spatial::{SO3, SpatialInertia, Wrench};

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

    /// How quickly the body's motion changes under a given push and turn.
    ///
    /// `orientation` is which way the body is facing and `angular_rate` how fast it is turning,
    /// in the body's own axes. `applied_wrench` is everything pushing and turning the body apart
    /// from gravity — rotor thrust, drag, a gust — with both parts read in the body's own axes and
    /// the turning part taken about the body frame's origin. A force given in world axes is turned
    /// into body axes first with `orientation.inverse().act(force)`.
    ///
    /// The answer comes back as how fast the body frame's origin picks up speed, in world axes,
    /// and how fast the spin changes, in body axes.
    ///
    /// ```
    /// use multicalc::dynamics::RigidBody;
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::{SO3, SpatialInertia, Wrench};
    ///
    /// let inertia = SpatialInertia::from_diagonal_inertia(
    ///     2.0_f64,
    ///     Vector::new([0.0, 0.0, 0.0]),
    ///     Vector::new([1.0, 1.0, 1.0]),
    /// )
    /// .unwrap();
    /// let body = RigidBody::new(inertia, Vector::new([0.0, 0.0, -9.81])).unwrap();
    ///
    /// // Level, not turning, nothing pushing on it: it falls.
    /// let level = SO3::identity();
    /// let still = Vector::new([0.0, 0.0, 0.0]);
    /// let falling = body.accelerations(level, still, Wrench::zeros());
    /// assert!((falling.linear() - Vector::new([0.0, 0.0, -9.81])).norm() < 1e-12);
    /// assert!(falling.angular().norm() < 1e-12);
    ///
    /// // Pushing up with exactly its own weight holds it still.
    /// let hover = Wrench::new(Vector::new([0.0, 0.0, 19.62]), Vector::new([0.0, 0.0, 0.0]));
    /// let held = body.accelerations(level, still, hover);
    /// assert!(held.linear().norm() < 1e-12);
    /// ```
    #[must_use]
    pub fn accelerations(
        self,
        orientation: SO3<T>,
        angular_rate: Vector3D<T>,
        applied_wrench: Wrench<T>,
    ) -> RigidBodyAcceleration<T> {
        let mass = self.inertia.mass();
        let balance_point = self.inertia.center_of_mass();
        let rotational_inertia = self.inertia.rotational_inertia();
        let force = applied_wrench.force();

        // The turn, taken about the point the body balances on rather than about its origin.
        let turn_about_balance_point = applied_wrench.torque() - balance_point.cross(force);
        // A spinning body resists having its axis moved, which eats part of the turn applied.
        let spin_resistance = angular_rate.cross(rotational_inertia * angular_rate);
        let angular =
            self.inverse_rotational_inertia * (turn_about_balance_point - spin_resistance);

        // Gravity acts through the balance point, so it never turns the body.
        let balance_point_acceleration = orientation.act(force).scale(T::ONE / mass) + self.gravity;
        // The origin swings around the balance point as the body turns, so it picks up speed the
        // balance point does not.
        let swing =
            angular.cross(balance_point) + angular_rate.cross(angular_rate.cross(balance_point));
        let linear = balance_point_acceleration - orientation.act(swing);

        RigidBodyAcceleration { linear, angular }
    }
}

/// How quickly a body's motion is changing.
///
/// The straight-line part is how fast the body frame's origin is picking up speed, in world axes.
/// The turning part is how fast its spin is changing, in the body's own axes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigidBodyAcceleration<T: Numeric = f64> {
    linear: Vector3D<T>,
    angular: Vector3D<T>,
}

impl<T: Numeric> RigidBodyAcceleration<T> {
    /// A value from its straight-line and turning parts.
    #[inline]
    #[must_use]
    pub fn new(linear: Vector3D<T>, angular: Vector3D<T>) -> Self {
        RigidBodyAcceleration { linear, angular }
    }

    /// How fast the body frame's origin is picking up speed, in world axes.
    #[inline]
    pub fn linear(self) -> Vector3D<T> {
        self.linear
    }

    /// How fast the body's spin is changing, in the body's own axes.
    #[inline]
    pub fn angular(self) -> Vector3D<T> {
        self.angular
    }
}
