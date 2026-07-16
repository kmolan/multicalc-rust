//! Differential-drive geometry: wheel motion to chassis motion and back.

use crate::error::KinematicsError;
use crate::linear_algebra::Vector;
use crate::scalar::Numeric;
use crate::spatial::SE2;

/// Left/right wheel motion to linear/angular chassis motion. Unit-agnostic: rates in, rates out;
/// deltas in, deltas out.
#[inline]
fn to_chassis<T: Numeric>(r: T, b: T, left: T, right: T) -> (T, T) {
    (r * (right + left) * T::HALF, r * (right - left) / b)
}

/// The inverse of [`to_chassis`].
#[inline]
fn to_wheels<T: Numeric>(r: T, b: T, linear: T, angular: T) -> (T, T) {
    let half_span = angular * b * T::HALF;
    ((linear - half_span) / r, (linear + half_span) / r)
}

/// Wheel angular rates [rad/s]. Positive drives the chassis forward.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WheelRates<T: Numeric> {
    left: T,
    right: T,
}

/// Wheel angular deltas over one tick [rad] — what an encoder reports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WheelDeltas<T: Numeric> {
    left: T,
    right: T,
}

/// Chassis rate: forward speed [m/s] and yaw rate [rad/s].
///
/// There is no lateral field: a differential-drive chassis cannot slide sideways.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChassisRate<T: Numeric> {
    linear: T,
    angular: T,
}

/// Chassis delta over one tick: arc length [m] and heading change [rad].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChassisDelta<T: Numeric> {
    linear: T,
    angular: T,
}

impl<T: Numeric> WheelRates<T> {
    /// Rates from the left and right wheel, in [rad/s].
    #[inline]
    pub fn new(left: T, right: T) -> Self {
        WheelRates { left, right }
    }

    /// The left wheel rate.
    #[inline]
    pub fn left(self) -> T {
        self.left
    }

    /// The right wheel rate.
    #[inline]
    pub fn right(self) -> T {
        self.right
    }

    /// Both wheels stopped.
    #[inline]
    pub fn zeros() -> Self {
        WheelRates {
            left: T::ZERO,
            right: T::ZERO,
        }
    }
}

impl<T: Numeric> WheelDeltas<T> {
    /// Deltas from the left and right wheel, in [rad].
    #[inline]
    pub fn new(left: T, right: T) -> Self {
        WheelDeltas { left, right }
    }

    /// The left wheel delta.
    #[inline]
    pub fn left(self) -> T {
        self.left
    }

    /// The right wheel delta.
    #[inline]
    pub fn right(self) -> T {
        self.right
    }

    /// Neither wheel turned.
    #[inline]
    pub fn zeros() -> Self {
        WheelDeltas {
            left: T::ZERO,
            right: T::ZERO,
        }
    }
}

impl<T: Numeric> ChassisRate<T> {
    /// A rate from a forward speed [m/s] and a yaw rate [rad/s].
    #[inline]
    pub fn new(linear: T, angular: T) -> Self {
        ChassisRate { linear, angular }
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

    /// The chassis at rest.
    #[inline]
    pub fn zeros() -> Self {
        ChassisRate {
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
    /// Lossy: `ChassisRate::project_tangent(xi).to_tangent()` equals `xi` only when `xi[1]` is
    /// zero. [`tangent_slip`](Self::tangent_slip) reports what is discarded.
    #[inline]
    pub fn project_tangent(xi: Vector<3, T>) -> Self {
        ChassisRate {
            linear: xi[0],
            angular: xi[2],
        }
    }

    /// The lateral component of `xi`, which [`project_tangent`](Self::project_tangent) discards.
    /// Zero for any motion a differential drive can produce.
    #[inline]
    pub fn tangent_slip(xi: Vector<3, T>) -> T {
        xi[1]
    }

    /// The delta accrued over `dt` at this constant rate.
    #[inline]
    pub fn integrate_over(self, dt: T) -> ChassisDelta<T> {
        ChassisDelta {
            linear: self.linear * dt,
            angular: self.angular * dt,
        }
    }
}

impl<T: Numeric> ChassisDelta<T> {
    /// A delta from an arc length [m] and a heading change [rad].
    #[inline]
    pub fn new(linear: T, angular: T) -> Self {
        ChassisDelta { linear, angular }
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

    /// The chassis did not move.
    #[inline]
    pub fn zeros() -> Self {
        ChassisDelta {
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
pub struct DiffDrive<T: Numeric> {
    wheel_radius: T,
    wheelbase: T,
}

impl<T: Numeric> DiffDrive<T> {
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
    /// use multicalc::kinematics::{DiffDrive, WheelRates};
    /// let dd = DiffDrive::new(0.036_f64, 0.235).unwrap();
    /// let wheels = WheelRates::new(1.0, 2.0);
    /// let back = dd.inverse(dd.forward(wheels));
    /// assert!((back.left() - 1.0).abs() < 1e-15);
    /// assert!((back.right() - 2.0).abs() < 1e-15);
    /// ```
    pub fn new(wheel_radius: T, wheelbase: T) -> Result<Self, KinematicsError> {
        // Finiteness first: NaN passes `<= 0`, so it would slip through the sign test as positive.
        if !wheel_radius.is_finite() || !wheelbase.is_finite() {
            return Err(KinematicsError::NonFinite);
        }
        if wheel_radius <= T::ZERO || wheelbase <= T::ZERO {
            return Err(KinematicsError::NonPositiveParameter);
        }
        Ok(DiffDrive {
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

    /// Chassis rate from wheel rates.
    #[inline]
    pub fn forward(self, w: WheelRates<T>) -> ChassisRate<T> {
        let (linear, angular) = to_chassis(self.wheel_radius, self.wheelbase, w.left(), w.right());
        ChassisRate::new(linear, angular)
    }

    /// Wheel rates from a chassis rate.
    #[inline]
    pub fn inverse(self, c: ChassisRate<T>) -> WheelRates<T> {
        let (left, right) = to_wheels(self.wheel_radius, self.wheelbase, c.linear(), c.angular());
        WheelRates::new(left, right)
    }

    /// Chassis delta from wheel deltas over one tick.
    #[inline]
    pub fn forward_delta(self, d: WheelDeltas<T>) -> ChassisDelta<T> {
        let (linear, angular) = to_chassis(self.wheel_radius, self.wheelbase, d.left(), d.right());
        ChassisDelta::new(linear, angular)
    }

    /// Wheel deltas from a chassis delta over one tick.
    #[inline]
    pub fn inverse_delta(self, d: ChassisDelta<T>) -> WheelDeltas<T> {
        let (left, right) = to_wheels(self.wheel_radius, self.wheelbase, d.linear(), d.angular());
        WheelDeltas::new(left, right)
    }

    /// Wheel deltas from the distance each wheel travelled, in metres.
    #[inline]
    pub fn wheel_deltas_from_travel(self, left_m: T, right_m: T) -> WheelDeltas<T> {
        WheelDeltas::new(left_m / self.wheel_radius, right_m / self.wheel_radius)
    }

    /// The distance each wheel travelled, in metres, from its delta.
    #[inline]
    pub fn wheel_travel(self, d: WheelDeltas<T>) -> (T, T) {
        (d.left() * self.wheel_radius, d.right() * self.wheel_radius)
    }

    /// The pose after one tick of wheel motion, along the exact constant-twist arc.
    #[inline]
    pub fn odometry_step(self, pose: SE2<T>, d: WheelDeltas<T>) -> SE2<T> {
        crate::kinematics::odometry::integrate(pose, self.forward_delta(d))
    }
}
