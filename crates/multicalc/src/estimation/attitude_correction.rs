//! Common helpers for madgwick_filter and mahony_filter

use crate::linear_algebra::Vector3D;
use crate::ode::ExponentialMap;
use crate::scalar::Numeric;
use crate::spatial::SO3;

/// Which way an accelerometer points when the body is still, in world axes: straight up.
#[inline]
pub(super) fn upward_reference<T: Numeric>() -> Vector3D<T> {
    Vector3D::new([T::ZERO, T::ZERO, T::ONE])
}

/// Which way a magnetometer points once it is flattened level, in world axes: north.
#[inline]
pub(super) fn north_reference<T: Numeric>() -> Vector3D<T> {
    Vector3D::new([T::ONE, T::ZERO, T::ZERO])
}

/// True when every reading and the timestep is a real, finite number.
#[inline]
pub(super) fn readings_are_finite<T: Numeric>(
    gyroscope_reading: Vector3D<T>,
    accelerometer_reading: Vector3D<T>,
    magnetometer_reading: Option<Vector3D<T>>,
    timestep: T,
) -> bool {
    gyroscope_reading.is_finite()
        && accelerometer_reading.is_finite()
        && timestep.is_finite()
        && magnetometer_reading.is_none_or(Vector3D::is_finite)
}

/// True when every number in the orientation is a real, finite one.
#[inline]
pub(super) fn orientation_is_finite<T: Numeric>(orientation: SO3<T>) -> bool {
    let [w, x, y, z] = orientation.quaternion().as_array();
    w.is_finite() && x.is_finite() && y.is_finite() && z.is_finite()
}

/// The small turn that would bring the estimated facing onto what the sensors say.
///
/// Each sensor gives a direction the body can see and a direction the world is known to have. The
/// turn from one to the other is their cross product, and the two add. A reading that is all
/// zeros, or that will not normalize, adds nothing: a body in free fall has no usable down, and a
/// dropped magnetometer sample must not stop the loop.
///
/// The magnetometer's world direction is worked out afresh each call rather than taken as given.
/// The measured field is turned into world axes, its upward part is kept as it is, and everything
/// left over is laid along north. That way the magnetometer only ever moves the heading, and a
/// caller who does not know how steeply the local field dips cannot get a lasting lean out of it.
#[inline]
pub(super) fn correction<T: Numeric>(
    orientation: SO3<T>,
    accelerometer_reading: Vector3D<T>,
    magnetometer_reading: Option<Vector3D<T>>,
    upward_reference: Vector3D<T>,
    north_reference: Vector3D<T>,
) -> Vector3D<T> {
    let mut total = Vector3D::zeros();

    if let Some(measured) = accelerometer_reading.try_normalized() {
        total += measured.cross(orientation.inverse().act(upward_reference));
    }

    if let Some(measured) = magnetometer_reading.and_then(Vector3D::try_normalized) {
        let in_world = orientation.act(measured);
        let vertical = in_world.dot(upward_reference);
        let across = T::ONE - vertical * vertical;
        // Rounding can push this a hair below zero when the field is almost straight up or down.
        let horizontal = if across > T::ZERO {
            across.sqrt()
        } else {
            T::ZERO
        };
        let reference = north_reference * horizontal + upward_reference * vertical;
        total += measured.cross(orientation.inverse().act(reference));
    }

    total
}

/// Turns the facing forward one tick and pulls it back onto unit length.
///
/// The step itself already gives a true rotation; the pull back is there because these filters are
/// expected to run for hours at a kilohertz in single precision with nothing else watching for
/// drift. It costs one square root a tick.
#[inline]
pub(super) fn stepped_orientation<T: Numeric>(
    orientation: SO3<T>,
    corrected_rate: Vector3D<T>,
    timestep: T,
) -> SO3<T> {
    ExponentialMap::attitude_step(orientation, corrected_rate, timestep).normalized()
}
