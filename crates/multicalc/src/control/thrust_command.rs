//! Works out how a flying body should point, and how hard it should push, to get a wanted
//! acceleration.

use crate::error::ControlError;
use crate::linear_algebra::{Matrix, Vector, Vector3D};
use crate::scalar::Numeric;
use crate::spatial::SO3;

/// How a flying body should point and how hard it should push to produce a wanted acceleration.
///
/// A body whose rotors only push one way cannot accelerate sideways without tipping first, so a
/// wanted acceleration is really two commands: an attitude to reach, and a push to apply once
/// there. This carries both. Feed the attitude to
/// [`GeometricAttitudeController`](crate::control::GeometricAttitudeController) and the push to
/// whatever drives the motors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThrustCommand<T: Numeric = f64> {
    attitude: SO3<T>,
    thrust_acceleration: T,
}

impl<T: Numeric> ThrustCommand<T> {
    /// Returns the way the body should point.
    #[inline]
    #[must_use]
    pub fn attitude(&self) -> SO3<T> {
        self.attitude
    }

    /// Returns how hard to push, as an acceleration — the same number whatever the body weighs.
    #[inline]
    #[must_use]
    pub fn thrust_acceleration(&self) -> T {
        self.thrust_acceleration
    }

    /// Returns how hard to push as a force, for a body of the given mass.
    #[inline]
    #[must_use]
    pub fn thrust_force(&self, mass: T) -> T {
        self.thrust_acceleration * mass
    }
}

/// Works out how to point a flying body, and how hard to push, to get a wanted acceleration.
///
/// `acceleration_command` is the acceleration wanted of the body, not counting gravity — the output
/// of a position loop such as [`Lqr`](crate::control::Lqr). `desired_heading` is which way the body
/// should face, as an angle in the level plane measured from the world's +x axis, positive turning
/// toward +y. `gravity` is the strength of gravity, which the rotors have to cover on top of
/// whatever is being asked for. The world's z axis points up.
///
/// Returns [`ControlError::NonFinite`] if any argument is not finite,
/// [`ControlError::UndefinedThrustDirection`] if the wanted acceleration cancels gravity exactly,
/// leaving no direction to push in, or [`ControlError::UndefinedHeadingDirection`] if the push
/// would be straight along the wanted heading, which leaves the heading with nothing to set it by.
///
/// ```
/// use multicalc::control::thrust_command_from_acceleration;
/// use multicalc::linear_algebra::Vector;
///
/// let gravity = 9.81_f64;
/// let facing_along_x = 0.0;
///
/// // Asked to hold still, the body stays level and pushes just hard enough to hold itself up.
/// let hold =
///     thrust_command_from_acceleration(Vector::new([0.0, 0.0, 0.0]), facing_along_x, gravity)
///         .unwrap();
/// assert!((hold.thrust_acceleration() - gravity).abs() < 1e-12);
/// assert!(hold.attitude().log().norm() < 1e-12);
///
/// // Asked to speed up along x, it tips that way and pushes a little harder.
/// let go = thrust_command_from_acceleration(Vector::new([2.0, 0.0, 0.0]), facing_along_x, gravity)
///     .unwrap();
/// assert!(go.thrust_acceleration() > gravity);
/// let body_up = go.attitude().act(Vector::new([0.0, 0.0, 1.0]));
/// assert!(body_up[0] > 0.0);
///
/// // A body of known mass turns that push into a force.
/// let mass = 0.9;
/// assert!((go.thrust_force(mass) - mass * go.thrust_acceleration()).abs() < 1e-12);
///
/// // Asked to fall freely, there is no direction to push in.
/// let free_fall = Vector::new([0.0, 0.0, -gravity]);
/// assert!(thrust_command_from_acceleration(free_fall, facing_along_x, gravity).is_err());
/// ```
pub fn thrust_command_from_acceleration<T: Numeric>(
    acceleration_command: Vector3D<T>,
    desired_heading: T,
    gravity: T,
) -> Result<ThrustCommand<T>, ControlError> {
    if !acceleration_command.is_finite() || !desired_heading.is_finite() || !gravity.is_finite() {
        return Err(ControlError::NonFinite);
    }

    // Gravity pulls the body down the whole time, so the rotors cover that as well as whatever
    // acceleration is being asked for.
    let push = acceleration_command + Vector::new([T::ZERO, T::ZERO, gravity]);
    let thrust_acceleration = push.norm();
    let up_axis = push
        .try_normalized()
        .ok_or(ControlError::UndefinedThrustDirection)?;

    // Which way the body's nose ends up facing once it has tipped: the wanted heading, taken level,
    // then leaned over to sit at right angles to the push.
    let heading_direction = Vector::new([desired_heading.cos(), desired_heading.sin(), T::ZERO]);
    let side_axis = up_axis
        .cross(heading_direction)
        .try_normalized()
        .ok_or(ControlError::UndefinedHeadingDirection)?;
    let forward_axis = side_axis.cross(up_axis);

    // The three axes, written as the columns of the rotation they describe.
    let rotation = Matrix::new([
        [forward_axis[0], side_axis[0], up_axis[0]],
        [forward_axis[1], side_axis[1], up_axis[1]],
        [forward_axis[2], side_axis[2], up_axis[2]],
    ]);
    // The axes are built at right angles and the right way round, so this only refuses a value
    // that slipped through as not finite.
    let attitude = SO3::try_from_matrix(rotation).ok_or(ControlError::NonFinite)?;

    Ok(ThrustCommand {
        attitude,
        thrust_acceleration,
    })
}
