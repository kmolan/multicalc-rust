//! Desired motion for one tick, joint-space or Cartesian, and the per-joint terms a model-based
//! law works out against it.
#![deny(clippy::indexing_slicing)]

use crate::dynamics::ArticulatedBody;
use crate::kinematics::JointKind;
use crate::linear_algebra::Vector;
use crate::scalar::Numeric;
use crate::spatial::{SE3, Twist};

/// Desired joint-space motion for one tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointReference<const MAX_CONFIG: usize, T: Numeric = f64> {
    /// Desired joint positions.
    pub position: Vector<MAX_CONFIG, T>,
    /// Desired joint rates.
    pub velocity: Vector<MAX_CONFIG, T>,
    /// Desired joint accelerations.
    pub acceleration: Vector<MAX_CONFIG, T>,
}

impl<const MAX_CONFIG: usize, T: Numeric> JointReference<MAX_CONFIG, T> {
    /// A reference from its three vectors.
    #[inline]
    #[must_use]
    pub fn new(
        position: Vector<MAX_CONFIG, T>,
        velocity: Vector<MAX_CONFIG, T>,
        acceleration: Vector<MAX_CONFIG, T>,
    ) -> Self {
        Self {
            position,
            velocity,
            acceleration,
        }
    }

    /// A reference standing still at `position`: zero rate and zero acceleration.
    #[inline]
    #[must_use]
    pub fn at_rest(position: Vector<MAX_CONFIG, T>) -> Self {
        Self {
            position,
            velocity: Vector::zeros(),
            acceleration: Vector::zeros(),
        }
    }
}

/// Desired tool motion for one tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CartesianReference<T: Numeric = f64> {
    /// Desired tool pose, in world.
    pub pose: SE3<T>,
    /// Desired tool twist, in the controller's configured frame.
    pub twist: Twist<T>,
}

impl<T: Numeric> CartesianReference<T> {
    /// A reference from a pose and a twist.
    #[inline]
    #[must_use]
    pub fn new(pose: SE3<T>, twist: Twist<T>) -> Self {
        Self { pose, twist }
    }

    /// A reference standing still at `pose`: zero twist.
    #[inline]
    #[must_use]
    pub fn at_rest(pose: SE3<T>) -> Self {
        Self {
            pose,
            twist: Twist::zeros(),
        }
    }
}

/// The per-joint position error `desired − measured`, wrapped where the joint kind needs it.
///
/// Plain difference for `Revolute`/`Prismatic`, [`Numeric::wrap_to_pi`] of the difference for
/// `Continuous`, zero for `Fixed`. `ArticulatedBody::new` rejects a floating base, so a joint's
/// configuration offset and velocity offset coincide on every model this sees.
pub(crate) fn joint_position_error<const MAX_JOINTS: usize, const MAX_CONFIG: usize, T: Numeric>(
    body: &ArticulatedBody<MAX_JOINTS, MAX_CONFIG, T>,
    desired: &Vector<MAX_CONFIG, T>,
    measured: &Vector<MAX_CONFIG, T>,
) -> Vector<MAX_CONFIG, T> {
    let mut error = Vector::<MAX_CONFIG, T>::zeros();
    for index in 0..body.len() {
        let Some(joint) = body.tree().joint(index) else {
            continue;
        };
        let Some(offset) = body.tree().velocity_offset(index) else {
            continue;
        };
        let Some(desired_reading) = desired.get(offset) else {
            continue;
        };
        let Some(measured_reading) = measured.get(offset) else {
            continue;
        };
        let difference = *desired_reading - *measured_reading;
        let wrapped = match joint.kind() {
            JointKind::Revolute | JointKind::Prismatic => difference,
            JointKind::Continuous => difference.wrap_to_pi(),
            JointKind::Fixed | JointKind::Floating => continue,
        };
        let Some(entry) = error.get_mut(offset) else {
            continue;
        };
        *entry = wrapped;
    }
    error
}

/// `sign(rate)`, zero at zero on every scalar type.
///
/// Not [`Numeric::signum`]: the `f32`/`f64` impls override it to the primitive, where `0.0.signum()`
/// is `1.0` — which would break a standing joint away under its full friction loss.
fn coulomb_direction<T: Numeric>(rate: T) -> T {
    if rate > T::ZERO {
        T::ONE
    } else if rate < T::ZERO {
        -T::ONE
    } else {
        T::ZERO
    }
}

/// The Coulomb feedforward correction: `friction_loss_i · (sign(q̇_d,i) − sign(q̇_i))` per movable
/// joint, zero elsewhere.
///
/// The model terms inside [`ArticulatedBody::inverse_dynamics`] evaluate Coulomb friction at the
/// rate they are handed. Adding this moves that term to the desired rate, where it depends only on
/// a planned reference and so cannot flip sign on measurement noise about zero. Viscous damping is
/// deliberately left at the measured rate: it is linear in `q̇` and cancels the model exactly.
pub(crate) fn coulomb_feedforward_correction<
    const MAX_JOINTS: usize,
    const MAX_CONFIG: usize,
    T: Numeric,
>(
    body: &ArticulatedBody<MAX_JOINTS, MAX_CONFIG, T>,
    desired_velocity: &Vector<MAX_CONFIG, T>,
    measured_velocity: &Vector<MAX_CONFIG, T>,
) -> Vector<MAX_CONFIG, T> {
    let mut correction = Vector::<MAX_CONFIG, T>::zeros();
    for index in 0..body.len() {
        let Some(joint) = body.tree().joint(index) else {
            continue;
        };
        match joint.kind() {
            JointKind::Revolute | JointKind::Continuous | JointKind::Prismatic => {}
            JointKind::Fixed | JointKind::Floating => continue,
        }
        let Some(offset) = body.tree().velocity_offset(index) else {
            continue;
        };
        let Some(desired_rate) = desired_velocity.get(offset) else {
            continue;
        };
        let Some(measured_rate) = measured_velocity.get(offset) else {
            continue;
        };
        let Some(entry) = correction.get_mut(offset) else {
            continue;
        };
        *entry = joint.friction_loss()
            * (coulomb_direction(*desired_rate) - coulomb_direction(*measured_rate));
    }
    correction
}
