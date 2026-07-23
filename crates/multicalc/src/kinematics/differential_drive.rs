//! Differential-drive geometry: wheel motion to body motion and back.

use crate::error::KinematicsError;
use crate::linear_algebra::Vector;
use crate::scalar::Numeric;
use crate::spatial::SE2;

/// Left/right wheel motion to linear/angular body motion. Unit-agnostic: velocities in, twist out;
/// rotations in, arc out.
#[inline]
fn to_body<T: Numeric>(r: T, b: T, left: T, right: T) -> (T, T) {
    (r * (right + left) * T::HALF, r * (right - left) / b)
}

/// The inverse of [`to_body`].
#[inline]
fn to_wheels<T: Numeric>(r: T, b: T, linear: T, angular: T) -> (T, T) {
    let half_span = angular * b * T::HALF;
    ((linear - half_span) / r, (linear + half_span) / r)
}

/// Wheel angular velocities [rad/s]. Positive drives the body forward.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WheelVelocities<T: Numeric> {
    left: T,
    right: T,
}

/// Wheel angular displacements over one tick `[rad]` — what an encoder reports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WheelRotations<T: Numeric> {
    left: T,
    right: T,
}

/// The body twist a differential drive can realise: forward speed `[m/s]` and yaw rate `[rad/s]`.
///
/// The se(2) twist restricted to two degrees of freedom. There is no lateral field: a
/// differential-drive body cannot slide sideways.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyTwist<T: Numeric> {
    linear: T,
    angular: T,
}

/// The arc a body traces over one tick: arc length `[m]` and heading change `[rad]`.
///
/// Arc length, not displacement — the straight-line distance covered is the chord, which is shorter
/// whenever the heading changes. These are the exponential coordinates of the relative pose.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyArc<T: Numeric> {
    linear: T,
    angular: T,
}

impl<T: Numeric> WheelVelocities<T> {
    /// Velocities of the left and right wheel, in `[rad/s]`.
    #[inline]
    pub fn new(left: T, right: T) -> Self {
        WheelVelocities { left, right }
    }

    /// The left wheel velocity.
    #[inline]
    pub fn left(self) -> T {
        self.left
    }

    /// The right wheel velocity.
    #[inline]
    pub fn right(self) -> T {
        self.right
    }

    /// Both wheels stopped.
    #[inline]
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
    pub fn new(left: T, right: T) -> Self {
        WheelRotations { left, right }
    }

    /// The left wheel rotation.
    #[inline]
    pub fn left(self) -> T {
        self.left
    }

    /// The right wheel rotation.
    #[inline]
    pub fn right(self) -> T {
        self.right
    }

    /// Neither wheel turned.
    #[inline]
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
    pub fn new(linear: T, angular: T) -> Self {
        BodyTwist { linear, angular }
    }

    /// The forward speed.
    #[inline]
    pub fn linear(self) -> T {
        self.linear
    }

    /// The yaw rate.
    #[inline]
    pub fn angular(self) -> T {
        self.angular
    }

    /// The body at rest.
    #[inline]
    pub fn zeros() -> Self {
        BodyTwist {
            linear: T::ZERO,
            angular: T::ZERO,
        }
    }

    /// The se(2) tangent `[v, 0, ω]` in the crate-wide `[v; ω]` ordering.
    #[inline]
    pub fn to_tangent(self) -> Vector<3, T> {
        Vector::new([self.linear, T::ZERO, self.angular])
    }

    /// Projects an se(2) tangent onto the motions a differential drive can produce, discarding the
    /// lateral component.
    ///
    /// Lossy: `BodyTwist::project_tangent(xi).to_tangent()` equals `xi` only when the
    /// lateral is zero. `tangent_slip` reports what is discarded.
    #[inline]
    pub fn project_tangent(xi: Vector<3, T>) -> Self {
        let [linear, _, angular] = *xi.as_array();
        BodyTwist { linear, angular }
    }

    /// The lateral component of `xi`, which [`project_tangent`](Self::project_tangent) discards.
    /// Zero for any motion a differential drive can produce.
    #[inline]
    pub fn tangent_slip(xi: Vector<3, T>) -> T {
        let [_, lateral, _] = *xi.as_array();
        lateral
    }

    /// The arc traced over `dt` by holding this twist constant.
    #[inline]
    pub fn integrate_over(self, dt: T) -> BodyArc<T> {
        BodyArc {
            linear: self.linear * dt,
            angular: self.angular * dt,
        }
    }
}

impl<T: Numeric> BodyArc<T> {
    /// An arc from an arc length `[m]` and a heading change `[rad]`.
    #[inline]
    pub fn new(linear: T, angular: T) -> Self {
        BodyArc { linear, angular }
    }

    /// The arc length.
    #[inline]
    pub fn linear(self) -> T {
        self.linear
    }

    /// The heading change.
    #[inline]
    pub fn angular(self) -> T {
        self.angular
    }

    /// The body did not move.
    #[inline]
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
pub struct DifferentialDrive<T: Numeric> {
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
    /// let dd = DifferentialDrive::new(0.036_f64, 0.235).unwrap();
    /// let wheels = WheelVelocities::new(1.0, 2.0);
    /// let back = dd.inverse(dd.forward(wheels));
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
    pub fn wheel_radius(self) -> T {
        self.wheel_radius
    }

    /// The track width.
    #[inline]
    pub fn wheelbase(self) -> T {
        self.wheelbase
    }

    /// The body twist produced by wheel velocities.
    #[inline]
    pub fn forward(self, w: WheelVelocities<T>) -> BodyTwist<T> {
        let (linear, angular) = to_body(self.wheel_radius, self.wheelbase, w.left(), w.right());
        BodyTwist::new(linear, angular)
    }

    /// The wheel velocities that produce a body twist.
    #[inline]
    pub fn inverse(self, c: BodyTwist<T>) -> WheelVelocities<T> {
        let (left, right) = to_wheels(self.wheel_radius, self.wheelbase, c.linear(), c.angular());
        WheelVelocities::new(left, right)
    }

    /// The arc traced by wheel rotations over one tick.
    #[inline]
    pub fn forward_arc(self, d: WheelRotations<T>) -> BodyArc<T> {
        let (linear, angular) = to_body(self.wheel_radius, self.wheelbase, d.left(), d.right());
        BodyArc::new(linear, angular)
    }

    /// The wheel rotations that trace an arc over one tick.
    #[inline]
    pub fn inverse_arc(self, d: BodyArc<T>) -> WheelRotations<T> {
        let (left, right) = to_wheels(self.wheel_radius, self.wheelbase, d.linear(), d.angular());
        WheelRotations::new(left, right)
    }

    /// Wheel rotations from the distance each wheel travelled, in metres.
    #[inline]
    pub fn wheel_rotations_from_travel(self, left_m: T, right_m: T) -> WheelRotations<T> {
        WheelRotations::new(left_m / self.wheel_radius, right_m / self.wheel_radius)
    }

    /// The distance each wheel travelled, in metres, from its rotation.
    #[inline]
    pub fn wheel_travel(self, d: WheelRotations<T>) -> (T, T) {
        (d.left() * self.wheel_radius, d.right() * self.wheel_radius)
    }

    /// The pose after one tick of wheel motion, along the exact constant-twist arc.
    #[inline]
    pub fn odometry_step(self, pose: SE2<T>, d: WheelRotations<T>) -> SE2<T> {
        crate::kinematics::odometry::integrate(pose, self.forward_arc(d))
    }
}
