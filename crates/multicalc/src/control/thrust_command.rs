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
    tilt_bound: bool,
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

    /// `true` when the attitude asks for less tilt than the wanted acceleration would otherwise
    /// need, because
    /// [`thrust_command_from_acceleration_with_tilt_limit`] capped it at its `max_tilt` and scaled
    /// the sideways part of the push down to fit, keeping the vertical part unchanged.
    ///
    /// A command built by [`thrust_command_from_acceleration`], which has no tilt limit to hit, is
    /// never bound.
    #[inline]
    #[must_use]
    pub fn tilt_bound(&self) -> bool {
        self.tilt_bound
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
    thrust_command_from_push(push, desired_heading, false)
}

/// Works out how to point a flying body, and how hard to push, to get a wanted acceleration —
/// the same as [`thrust_command_from_acceleration`], but with the tilt capped.
///
/// A position loop with a distant setpoint, or one that has wound up, can ask for a lateral
/// acceleration large enough to tip the body past where its vertical thrust still holds it up, so
/// it descends while accelerating sideways instead of climbing or holding height. `max_tilt` caps
/// how far from level the attitude is allowed to lean, as an angle in `(0, π/2)` radians measured
/// from the world's +z axis. When the wanted acceleration would tip the body past that, the
/// sideways part of the push is scaled down to fit the cap while the vertical part — and so the
/// climb or descent it commands — is left exactly as it was; [`ThrustCommand::tilt_bound`] then
/// reports `true`. A wanted acceleration already inside the cap comes back unchanged, same as
/// [`thrust_command_from_acceleration`] would give it, with [`ThrustCommand::tilt_bound`]
/// reporting `false`.
///
/// A commanded descent that is already faster than free-fall cannot be produced at any tilt under
/// a quarter turn, so the vertical part cannot be left untouched the way it is above: instead the
/// push is leaned over to sit right at `max_tilt` while its *magnitude* is left as it was, which
/// still climbs (or falls) as much as the cap allows rather than as much as gravity alone would
/// give it. With no sideways push to take a direction from (a commanded descent straight down),
/// it leans toward `desired_heading` instead of leaving the direction undefined.
///
/// Returns [`ControlError::NonFinite`] if any argument, including `max_tilt`, is not finite,
/// [`ControlError::InvalidTiltLimit`] if `max_tilt` is not strictly positive or reaches a quarter
/// turn (π/2) or beyond, [`ControlError::UndefinedThrustDirection`] if the wanted acceleration
/// cancels gravity exactly, leaving no direction to push in, or
/// [`ControlError::UndefinedHeadingDirection`] if the push would be straight along the wanted
/// heading, which leaves the heading with nothing to set it by.
///
/// ```
/// use multicalc::control::thrust_command_from_acceleration_with_tilt_limit;
/// use multicalc::linear_algebra::Vector;
///
/// let gravity = 9.81_f64;
/// let facing_along_x = 0.0;
/// let max_tilt = 0.5; // a bit under a third of a turn
///
/// // A modest sideways request stays inside the cap and comes back unbound.
/// let modest = thrust_command_from_acceleration_with_tilt_limit(
///     Vector::new([1.0, 0.0, 0.0]),
///     facing_along_x,
///     gravity,
///     max_tilt,
/// )
/// .unwrap();
/// assert!(!modest.tilt_bound());
///
/// // A far larger sideways request is capped at max_tilt, and says so.
/// let huge = thrust_command_from_acceleration_with_tilt_limit(
///     Vector::new([50.0, 0.0, 0.0]),
///     facing_along_x,
///     gravity,
///     max_tilt,
/// )
/// .unwrap();
/// assert!(huge.tilt_bound());
/// let up_axis = huge.attitude().act(Vector::new([0.0, 0.0, 1.0]));
/// assert!((up_axis[2].acos() - max_tilt).abs() < 1e-9);
///
/// // The vertical part of the push — what holds the body up — is left untouched by the cap.
/// assert!((huge.thrust_acceleration() * up_axis[2] - gravity).abs() < 1e-9);
/// ```
pub fn thrust_command_from_acceleration_with_tilt_limit<T: Numeric>(
    acceleration_command: Vector3D<T>,
    desired_heading: T,
    gravity: T,
    max_tilt: T,
) -> Result<ThrustCommand<T>, ControlError> {
    if !acceleration_command.is_finite()
        || !desired_heading.is_finite()
        || !gravity.is_finite()
        || !max_tilt.is_finite()
    {
        return Err(ControlError::NonFinite);
    }
    let quarter_turn = T::PI * T::HALF;
    if max_tilt <= T::ZERO || max_tilt >= quarter_turn {
        return Err(ControlError::InvalidTiltLimit);
    }

    let push = acceleration_command + Vector::new([T::ZERO, T::ZERO, gravity]);
    let vertical = push[2];
    let horizontal = Vector::new([push[0], push[1], T::ZERO]).norm();

    let mut tilt_bound = false;
    let push = if vertical > T::ZERO && horizontal.atan2(vertical) > max_tilt {
        // Still points somewhat upward: hold the vertical part fixed, so the commanded climb or
        // descent rate survives, and shrink the sideways part down to the cap.
        tilt_bound = true;
        let horizontal_limit = vertical * max_tilt.tan();
        let scale = horizontal_limit / horizontal;
        Vector::new([push[0] * scale, push[1] * scale, vertical])
    } else if vertical < T::ZERO {
        // The requested descent alone is already faster than free-fall, so no tilt within the cap
        // (necessarily under a quarter turn) can produce it: holding the vertical part fixed the
        // way the branch above does is not an option here, since that would need a lean of more
        // than max_tilt no matter how far the sideways part is shrunk. The best this can do is
        // hold the *magnitude* of the push fixed and lean it over to sit right at the cap, which
        // still gives as much climb as the cap allows rather than none. With no sideways part to
        // take a direction from, lean toward the desired heading instead of leaving it undefined.
        tilt_bound = true;
        let magnitude = push.norm();
        let capped_horizontal = magnitude * max_tilt.sin();
        let capped_vertical = magnitude * max_tilt.cos();
        if horizontal > T::ZERO {
            let scale = capped_horizontal / horizontal;
            Vector::new([push[0] * scale, push[1] * scale, capped_vertical])
        } else {
            Vector::new([
                desired_heading.cos() * capped_horizontal,
                desired_heading.sin() * capped_horizontal,
                capped_vertical,
            ])
        }
    } else {
        push
    };

    thrust_command_from_push(push, desired_heading, tilt_bound)
}

/// Shared tail of both entry points: turns an already-settled push (gravity accounted for, and
/// tilt-capped if that was asked for) into the attitude and thrust that produce it.
fn thrust_command_from_push<T: Numeric>(
    push: Vector3D<T>,
    desired_heading: T,
    tilt_bound: bool,
) -> Result<ThrustCommand<T>, ControlError> {
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
        tilt_bound,
    })
}
