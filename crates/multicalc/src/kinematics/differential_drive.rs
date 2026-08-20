//! Differential-drive geometry: wheel motion to body motion and back.

use crate::error::KinematicsError;
use crate::linear_algebra::{Vector, Vector3D};
use crate::scalar::Numeric;
use crate::spatial::SE2;

/// Left/right wheel motion to linear/angular body motion. Unit-agnostic: velocities in, twist out;
/// rotations in, arc out.
#[inline]
#[must_use]
fn to_body<T: Numeric>(wheel_radius: T, wheelbase: T, left: T, right: T) -> (T, T) {
    (
        wheel_radius * (right + left) * T::HALF,
        wheel_radius * (right - left) / wheelbase,
    )
}

/// The inverse of [`to_body`].
#[inline]
#[must_use]
fn to_wheels<T: Numeric>(wheel_radius: T, wheelbase: T, linear: T, angular: T) -> (T, T) {
    let half_span = angular * wheelbase * T::HALF;
    (
        (linear - half_span) / wheel_radius,
        (linear + half_span) / wheel_radius,
    )
}

/// Wheel angular velocities [rad/s]. Positive drives the body forward.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WheelVelocities<T: Numeric = f64> {
    left: T,
    right: T,
}

/// Wheel angular displacements over one tick `[rad]` — what an encoder reports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WheelRotations<T: Numeric = f64> {
    left: T,
    right: T,
}

/// The body twist a differential drive can realise: forward speed `[m/s]` and yaw rate `[rad/s]`.
///
/// The se(2) twist restricted to two degrees of freedom. There is no lateral field: a
/// differential-drive body cannot slide sideways.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyTwist<T: Numeric = f64> {
    linear: T,
    angular: T,
}

/// The arc a body traces over one tick: arc length `[m]` and heading change `[rad]`.
///
/// Arc length, not displacement — the straight-line distance covered is the chord, which is shorter
/// whenever the heading changes. These are the exponential coordinates of the relative pose.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyArc<T: Numeric = f64> {
    linear: T,
    angular: T,
}

impl<T: Numeric> WheelVelocities<T> {
    /// Velocities of the left and right wheel, in `[rad/s]`.
    #[inline]
    #[must_use]
    pub fn new(left: T, right: T) -> Self {
        WheelVelocities { left, right }
    }

    /// The left wheel velocity.
    #[inline]
    #[must_use]
    pub fn left(self) -> T {
        self.left
    }

    /// The right wheel velocity.
    #[inline]
    #[must_use]
    pub fn right(self) -> T {
        self.right
    }

    /// Both wheels stopped.
    #[inline]
    #[must_use]
    pub fn zeros() -> Self {
        WheelVelocities {
            left: T::ZERO,
            right: T::ZERO,
        }
    }
}

impl<T: Numeric> WheelRotations<T> {
    /// Rotations of the left and right wheel, in `[rad]`.
    #[inline]
    #[must_use]
    pub fn new(left: T, right: T) -> Self {
        WheelRotations { left, right }
    }

    /// The left wheel rotation.
    #[inline]
    #[must_use]
    pub fn left(self) -> T {
        self.left
    }

    /// The right wheel rotation.
    #[inline]
    #[must_use]
    pub fn right(self) -> T {
        self.right
    }

    /// Neither wheel turned.
    #[inline]
    #[must_use]
    pub fn zeros() -> Self {
        WheelRotations {
            left: T::ZERO,
            right: T::ZERO,
        }
    }
}

impl<T: Numeric> BodyTwist<T> {
    /// A twist from a forward speed `[m/s]` and a yaw rate `[rad/s]`.
    #[inline]
    #[must_use]
    pub fn new(linear: T, angular: T) -> Self {
        BodyTwist { linear, angular }
    }

    /// The forward speed.
    #[inline]
    #[must_use]
    pub fn linear(self) -> T {
        self.linear
    }

    /// The yaw rate.
    #[inline]
    #[must_use]
    pub fn angular(self) -> T {
        self.angular
    }

    /// The body at rest.
    #[inline]
    #[must_use]
    pub fn zeros() -> Self {
        BodyTwist {
            linear: T::ZERO,
            angular: T::ZERO,
        }
    }

    /// The se(2) tangent `[v, 0, ω]` in the crate-wide `[v; ω]` ordering.
    #[inline]
    pub fn to_tangent(self) -> Vector3D<T> {
        Vector::new([self.linear, T::ZERO, self.angular])
    }

    /// Projects an se(2) tangent onto the motions a differential drive can produce, discarding the
    /// lateral component.
    ///
    /// Lossy: `BodyTwist::project_tangent(twist).to_tangent()` equals `twist` only when the
    /// lateral is zero. `tangent_slip` reports what is discarded.
    #[inline]
    #[must_use]
    pub fn project_tangent(twist: Vector3D<T>) -> Self {
        let [linear, _, angular] = *twist.as_array();
        BodyTwist { linear, angular }
    }

    /// The lateral component of `twist`, which [`project_tangent`](Self::project_tangent) discards.
    /// Zero for any motion a differential drive can produce.
    #[inline]
    #[must_use]
    pub fn tangent_slip(twist: Vector3D<T>) -> T {
        let [_, lateral, _] = *twist.as_array();
        lateral
    }

    /// The arc traced over `timestep` by holding this twist constant.
    #[inline]
    #[must_use]
    pub fn integrate_over(self, timestep: T) -> BodyArc<T> {
        BodyArc {
            linear: self.linear * timestep,
            angular: self.angular * timestep,
        }
    }
}

impl<T: Numeric> BodyArc<T> {
    /// An arc from an arc length `[m]` and a heading change `[rad]`.
    #[inline]
    #[must_use]
    pub fn new(linear: T, angular: T) -> Self {
        BodyArc { linear, angular }
    }

    /// The arc length.
    #[inline]
    #[must_use]
    pub fn linear(self) -> T {
        self.linear
    }

    /// The heading change.
    #[inline]
    #[must_use]
    pub fn angular(self) -> T {
        self.angular
    }

    /// The body did not move.
    #[inline]
    #[must_use]
    pub fn zeros() -> Self {
        BodyArc {
            linear: T::ZERO,
            angular: T::ZERO,
        }
    }
}

/// Differential-drive geometry.
///
/// `wheelbase` is the track width: the lateral distance between the two wheel contact points, not a
/// front-to-rear axle distance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DifferentialDrive<T: Numeric = f64> {
    wheel_radius: T,
    wheelbase: T,
}

impl<T: Numeric> DifferentialDrive<T> {
    /// Geometry from a wheel radius and a track width, both in metres.
    ///
    /// This is the only fallible operation in the module: with the geometry checked once here,
    /// every map below is total.
    ///
    /// # Errors
    /// [`NonFinite`](KinematicsError::NonFinite) if either parameter is infinite or NaN;
    /// [`NonPositiveParameter`](KinematicsError::NonPositiveParameter) if either is not strictly
    /// positive.
    ///
    /// ```
    /// use multicalc::kinematics::{DifferentialDrive, WheelVelocities};
    /// let wheel_radius = 0.036_f64;   // 36 mm wheels
    /// let track_width = 0.235;        // 235 mm apart
    /// let drive = DifferentialDrive::new(wheel_radius, track_width).unwrap();
    ///
    /// let wheels = WheelVelocities::new(1.0, 2.0);
    /// let back = drive.inverse(drive.forward(wheels));
    /// assert!((back.left() - 1.0).abs() < 1e-15);
    /// assert!((back.right() - 2.0).abs() < 1e-15);
    /// ```
    pub fn new(wheel_radius: T, wheelbase: T) -> Result<Self, KinematicsError> {
        // Finiteness first: NaN fails `<= 0`, so the sign test alone would accept it.
        if !wheel_radius.is_finite() || !wheelbase.is_finite() {
            return Err(KinematicsError::NonFinite);
        }
        if wheel_radius <= T::ZERO || wheelbase <= T::ZERO {
            return Err(KinematicsError::NonPositiveParameter);
        }
        Ok(DifferentialDrive {
            wheel_radius,
            wheelbase,
        })
    }

    /// The wheel radius.
    #[inline]
    #[must_use]
    pub fn wheel_radius(self) -> T {
        self.wheel_radius
    }

    /// The track width.
    #[inline]
    #[must_use]
    pub fn wheelbase(self) -> T {
        self.wheelbase
    }

    /// The body twist produced by wheel velocities.
    #[inline]
    #[must_use]
    pub fn forward(self, w: WheelVelocities<T>) -> BodyTwist<T> {
        let (linear, angular) = to_body(self.wheel_radius, self.wheelbase, w.left(), w.right());
        BodyTwist::new(linear, angular)
    }

    /// The wheel velocities that produce a body twist.
    #[inline]
    #[must_use]
    pub fn inverse(self, command: BodyTwist<T>) -> WheelVelocities<T> {
        let (left, right) = to_wheels(
            self.wheel_radius,
            self.wheelbase,
            command.linear(),
            command.angular(),
        );
        WheelVelocities::new(left, right)
    }

    /// The arc traced by wheel rotations over one tick.
    #[inline]
    #[must_use]
    pub fn forward_arc(self, rotations: WheelRotations<T>) -> BodyArc<T> {
        let (linear, angular) = to_body(
            self.wheel_radius,
            self.wheelbase,
            rotations.left(),
            rotations.right(),
        );
        BodyArc::new(linear, angular)
    }

    /// The wheel rotations that trace an arc over one tick.
    #[inline]
    #[must_use]
    pub fn inverse_arc(self, arc: BodyArc<T>) -> WheelRotations<T> {
        let (left, right) = to_wheels(
            self.wheel_radius,
            self.wheelbase,
            arc.linear(),
            arc.angular(),
        );
        WheelRotations::new(left, right)
    }

    /// Wheel rotations from the distance each wheel travelled, in metres.
    #[inline]
    #[must_use]
    pub fn wheel_rotations_from_travel(self, left_m: T, right_m: T) -> WheelRotations<T> {
        WheelRotations::new(left_m / self.wheel_radius, right_m / self.wheel_radius)
    }

    /// The distance each wheel travelled, in metres, from its rotation.
    #[inline]
    #[must_use]
    pub fn wheel_travel(self, rotations: WheelRotations<T>) -> (T, T) {
        (
            rotations.left() * self.wheel_radius,
            rotations.right() * self.wheel_radius,
        )
    }

    /// The pose after one tick of wheel motion, along the exact constant-twist arc.
    #[inline]
    #[must_use]
    pub fn odometry_step(self, pose: SE2<T>, rotations: WheelRotations<T>) -> SE2<T> {
        crate::kinematics::odometry::integrate(pose, self.forward_arc(rotations))
    }
}
