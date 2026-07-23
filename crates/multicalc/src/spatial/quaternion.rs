//! Quaternions for 3D rotation.
//!
//! [`Quaternion`] is a single quaternion type in the Hamilton convention, following the model of
//! Eigen, glam, and ROS `tf2`: contains both the raw quaternion algebra and the rotation
//! helpers. A quaternion represents a rotation only when it has unit norm. The rotation
//! *constructors* ([`Quaternion::from_axis_angle`], [`Quaternion::from_scaled_axis`],
//! [`Quaternion::from_euler_zyx`], [`Quaternion::try_from_rotation_matrix`]) return unit output;
//! the rotation *queries* ([`Quaternion::to_rotation_matrix`], [`Quaternion::slerp`],
//! [`Quaternion::transform_point`], [`Quaternion::to_euler_zyx`], [`Quaternion::to_axis_angle`],
//! [`Quaternion::to_scaled_axis`]) assume unit input — call [`Quaternion::normalized`] first if a
//! quaternion has drifted.
//!
//! Conventions (pinned crate-wide): the Hamilton product (matches Eigen/ROS/Sophus/Pinocchio,
//! not JPL), storage scalar-first as `[w, x, y, z]`, ZYX intrinsic Euler angles (yaw-pitch-roll),
//! and the shortest-path rule (a quaternion with a negative scalar part is negated before an
//! angle is measured or an interpolation taken).
//!
//! Renormalization policy: composition (`*`) is the exact Hamilton product and does not
//! renormalize, so a long chain of multiplications drifts off the unit sphere by rounding — call
//! [`Quaternion::normalized`] periodically (for example once per control tick). [`Quaternion::slerp`]
//! renormalizes its own result, and the rotation constructors return unit output by construction.

use core::ops::{Add, Mul, Neg, Sub};

use crate::linear_algebra::{Matrix, Vector};
use crate::scalar::Numeric;

use crate::spatial::{small_angle, small_angle_sq};

/// A quaternion `w + x·i + y·j + z·k`, stored scalar-first.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion<T: Numeric> {
    w: T,
    x: T,
    y: T,
    z: T,
}

impl<T: Numeric> Quaternion<T> {
    /// A quaternion from its four components, in `[w, x, y, z]` order.
    #[inline]
    pub fn new(w: T, x: T, y: T, z: T) -> Self {
        Quaternion { w, x, y, z }
    }

    /// The multiplicative identity `1 + 0i + 0j + 0k` (the zero rotation).
    ///
    /// ```
    /// use multicalc::spatial::Quaternion;
    /// assert_eq!(Quaternion::<f64>::identity().as_array(), [1.0, 0.0, 0.0, 0.0]);
    /// ```
    #[inline]
    pub fn identity() -> Self {
        Quaternion {
            w: T::ONE,
            x: T::ZERO,
            y: T::ZERO,
            z: T::ZERO,
        }
    }

    /// Builds a quaternion from a `[w, x, y, z]` array.
    #[inline]
    pub fn from_array(a: [T; 4]) -> Self {
        let [w, x, y, z] = a;
        Quaternion { w, x, y, z }
    }

    /// Builds a quaternion from a scalar part and a vector part.
    #[inline]
    pub fn from_scalar_vector(w: T, v: Vector<3, T>) -> Self {
        let [x, y, z] = *v.as_array();
        Quaternion { w, x, y, z }
    }

    /// The scalar (real) component.
    #[inline]
    pub fn w(self) -> T {
        self.w
    }

    /// The `i` component.
    #[inline]
    pub fn x(self) -> T {
        self.x
    }

    /// The `j` component.
    #[inline]
    pub fn y(self) -> T {
        self.y
    }

    /// The `k` component.
    #[inline]
    pub fn z(self) -> T {
        self.z
    }

    /// The vector (imaginary) part `[x, y, z]`.
    #[inline]
    pub fn vec(self) -> Vector<3, T> {
        Vector::new([self.x, self.y, self.z])
    }

    /// The components as a `[w, x, y, z]` array.
    #[inline]
    pub fn as_array(self) -> [T; 4] {
        [self.w, self.x, self.y, self.z]
    }

    /// The conjugate `w − x·i − y·j − z·k`.
    #[inline]
    pub fn conjugate(self) -> Self {
        Quaternion {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// The squared norm `w² + x² + y² + z²`.
    #[inline]
    pub fn norm_squared(self) -> T {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// The Euclidean norm.
    #[inline]
    pub fn norm(self) -> T {
        self.norm_squared().sqrt()
    }

    /// The four-component dot product.
    #[inline]
    pub fn dot(self, r: Self) -> T {
        self.w * r.w + self.x * r.x + self.y * r.y + self.z * r.z
    }

    /// The inverse `conjugate / norm²`. For a unit quaternion this equals the conjugate. Yields
    /// `inf`/`NaN` for a zero quaternion, exactly as plain-float division does elsewhere.
    #[inline]
    pub fn inverse(self) -> Self {
        self.conjugate() * self.norm_squared().recip()
    }

    /// This quaternion scaled to unit norm. Yields `NaN` components for a zero quaternion, as
    /// plain-float division does; use [`Quaternion::try_normalized`] for a checked version.
    #[inline]
    pub fn normalized(self) -> Self {
        self * self.norm().recip()
    }

    /// This quaternion scaled to unit norm, or `None` if the norm is non-finite or underflows.
    #[inline]
    pub fn try_normalized(self) -> Option<Self> {
        let n = self.norm();
        if !n.is_finite() || n <= T::EPSILON {
            None
        } else {
            Some(self * n.recip())
        }
    }

    /// The quaternion exponential `exp(w)·(cos‖v‖, sin‖v‖/‖v‖ · v)`, where `v` is the vector
    /// part. Near `‖v‖ = 0` the `cos‖v‖` and `sin‖v‖/‖v‖` factors use Taylor series in `‖v‖²`,
    /// so no `sqrt` is taken there and the AD derivative stays finite at `v = 0`.
    #[inline]
    pub fn exp(self) -> Self {
        let vn_sq = self.x * self.x + self.y * self.y + self.z * self.z;
        let ew = self.w.exp();
        let (cos_v, sinc_v) = if vn_sq < small_angle_sq::<T>() {
            // Taylor in ‖v‖²; avoids `sqrt(0)`, whose derivative is NaN.
            (T::ONE - vn_sq / T::TWO, T::ONE - vn_sq / T::from_f64(6.0))
        } else {
            let vn = vn_sq.sqrt();
            (vn.cos(), vn.sin() / vn)
        };
        let s = ew * sinc_v;
        Quaternion {
            w: ew * cos_v,
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    /// The quaternion natural logarithm, the inverse of [`Quaternion::exp`]: `(ln‖q‖, θ/‖v‖ · v)`
    /// with `θ = atan2(‖v‖, w)`. The `θ/‖v‖` factor tends to `1/‖q‖` as the vector part vanishes.
    ///
    /// Inputs near the negative real axis (`w < 0` with a vanishing vector part, i.e. a rotation
    /// near 2π) are not supported: the true log's vector part has magnitude ≈ π there but its
    /// direction is ill-defined, so the branch cut is left unhandled. For rotation logarithms use
    /// [`Quaternion::to_scaled_axis`], which resolves this region via the shortest-path sign fix.
    #[inline]
    pub fn ln(self) -> Self {
        let n = self.norm();
        let vn = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        // Restrict the small-vector fallback to w > 0; near the negative real axis θ ≈ π, not 0,
        // so 1/‖q‖ would be the wrong coefficient (see the doc note above).
        let coeff = if vn < small_angle::<T>() && self.w > T::ZERO {
            n.recip()
        } else {
            vn.atan2(self.w) / vn
        };
        Quaternion {
            w: n.ln(),
            x: self.x * coeff,
            y: self.y * coeff,
            z: self.z * coeff,
        }
    }

    /// The rotation of `angle` radians about `axis`, as a unit quaternion. The axis is normalized
    /// internally; a zero-length axis yields the identity rotation.
    ///
    /// ```
    /// use multicalc::spatial::Quaternion;
    /// use multicalc::linear_algebra::Vector;
    /// let q = Quaternion::from_axis_angle(Vector::new([0.0, 0.0, 1.0]),
    ///                                     core::f64::consts::FRAC_PI_2);
    /// let p = q.transform_point(Vector::new([1.0, 0.0, 0.0]));
    /// let [px, py, pz] = *p.as_array();
    /// assert!((px - 0.0).abs() < 1e-12);
    /// assert!((py - 1.0).abs() < 1e-12);
    /// assert!((pz - 0.0).abs() < 1e-12);
    /// ```
    #[inline]
    pub fn from_axis_angle(axis: Vector<3, T>, angle: T) -> Self {
        let an = axis.dot(axis).sqrt();
        if an <= T::EPSILON {
            return Self::identity();
        }
        let half = angle * T::HALF;
        let s = half.sin() / an;
        let [ax, ay, az] = *axis.as_array();
        Quaternion {
            w: half.cos(),
            x: ax * s,
            y: ay * s,
            z: az * s,
        }
    }

    /// The rotation whose axis-angle is the rotation vector `φ = θ·n̂` (the so(3) exponential
    /// map), as a unit quaternion. Near `θ = 0` the `cos(θ/2)` and `sin(θ/2)/θ` factors use
    /// Taylor series in `θ²`, so no `sqrt` is taken there and the AD derivative stays finite at
    /// `φ = 0` (a robot at rest).
    #[inline]
    pub fn from_scaled_axis(rotvec: Vector<3, T>) -> Self {
        let theta_sq = rotvec.dot(rotvec);
        let (w, scale) = if theta_sq < small_angle_sq::<T>() {
            // Taylor in θ²; avoids `sqrt(0)`, whose derivative is NaN.
            (
                T::ONE - theta_sq / T::from_f64(8.0),
                T::HALF - theta_sq / T::from_f64(48.0),
            )
        } else {
            let theta = theta_sq.sqrt();
            let half = theta * T::HALF;
            (half.cos(), half.sin() / theta)
        };
        let [rx, ry, rz] = *rotvec.as_array();
        Quaternion {
            w,
            x: rx * scale,
            y: ry * scale,
            z: rz * scale,
        }
    }

    /// The rotation from ZYX intrinsic Euler angles: `R = Rz(yaw)·Ry(pitch)·Rx(roll)`. The result
    /// is a unit quaternion.
    #[inline]
    pub fn from_euler_zyx(roll: T, pitch: T, yaw: T) -> Self {
        let (cr, sr) = ((roll * T::HALF).cos(), (roll * T::HALF).sin());
        let (cp, sp) = ((pitch * T::HALF).cos(), (pitch * T::HALF).sin());
        let (cy, sy) = ((yaw * T::HALF).cos(), (yaw * T::HALF).sin());
        Quaternion {
            w: cr * cp * cy + sr * sp * sy,
            x: sr * cp * cy - cr * sp * sy,
            y: cr * sp * cy + sr * cp * sy,
            z: cr * cp * sy - sr * sp * cy,
        }
    }

    /// Builds a unit quaternion from a rotation matrix by Shepperd's method (the largest of the
    /// trace and the three diagonal terms is the pivot, for numerical stability). A proper
    /// (orthonormal, determinant +1) rotation is assumed; `None` guards only against a degenerate
    /// pivot that would divide by zero.
    #[inline]
    pub fn try_from_rotation_matrix(m: Matrix<3, 3, T>) -> Option<Self> {
        let quarter = T::from_f64(0.25);
        let [[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]] = m.into_array();
        let trace = m00 + m11 + m22;
        let q = if trace > T::ZERO {
            let s = (trace + T::ONE).sqrt() * T::TWO; // s = 4·w
            Quaternion::new(
                quarter * s,
                (m21 - m12) / s,
                (m02 - m20) / s,
                (m10 - m01) / s,
            )
        } else if m00 > m11 && m00 > m22 {
            let s = (T::ONE + m00 - m11 - m22).sqrt() * T::TWO; // s = 4·x
            Quaternion::new(
                (m21 - m12) / s,
                quarter * s,
                (m01 + m10) / s,
                (m02 + m20) / s,
            )
        } else if m11 > m22 {
            let s = (T::ONE + m11 - m00 - m22).sqrt() * T::TWO; // s = 4·y
            Quaternion::new(
                (m02 - m20) / s,
                (m01 + m10) / s,
                quarter * s,
                (m12 + m21) / s,
            )
        } else {
            let s = (T::ONE + m22 - m00 - m11).sqrt() * T::TWO; // s = 4·z
            Quaternion::new(
                (m10 - m01) / s,
                (m02 + m20) / s,
                (m12 + m21) / s,
                quarter * s,
            )
        };
        q.try_normalized()
    }

    /// The 3×3 rotation matrix. Assumes a unit quaternion; call [`Quaternion::normalized`] first
    /// if it may have drifted.
    #[inline]
    pub fn to_rotation_matrix(self) -> Matrix<3, 3, T> {
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);
        let two = T::TWO;
        Matrix::new([
            [
                T::ONE - two * (y * y + z * z),
                two * (x * y - w * z),
                two * (x * z + w * y),
            ],
            [
                two * (x * y + w * z),
                T::ONE - two * (x * x + z * z),
                two * (y * z - w * x),
            ],
            [
                two * (x * z - w * y),
                two * (y * z + w * x),
                T::ONE - two * (x * x + y * y),
            ],
        ])
    }

    /// The rotation as a `(unit axis, angle)` pair, with `angle` in `[0, π]`. Assumes a unit
    /// quaternion. A near-zero rotation returns the x-axis and a zero angle.
    #[inline]
    pub fn to_axis_angle(self) -> (Vector<3, T>, T) {
        let q = if self.w < T::ZERO { -self } else { self };
        let vn = (q.x * q.x + q.y * q.y + q.z * q.z).sqrt();
        if vn <= T::EPSILON {
            return (Vector::new([T::ONE, T::ZERO, T::ZERO]), T::ZERO);
        }
        let inv = vn.recip();
        (
            Vector::new([q.x * inv, q.y * inv, q.z * inv]),
            T::TWO * vn.atan2(q.w),
        )
    }

    /// The rotation vector `φ = θ·n̂` (the so(3) logarithm), the inverse of
    /// [`Quaternion::from_scaled_axis`]. Assumes a unit quaternion; the shortest-path sign fix is
    /// applied so `‖φ‖ ≤ π`.
    #[inline]
    pub fn to_scaled_axis(self) -> Vector<3, T> {
        let q = if self.w < T::ZERO { -self } else { self };
        let vn = (q.x * q.x + q.y * q.y + q.z * q.z).sqrt();
        let coeff = if vn < small_angle::<T>() {
            T::TWO
        } else {
            (T::TWO * vn.atan2(q.w)) / vn
        };
        Vector::new([q.x * coeff, q.y * coeff, q.z * coeff])
    }

    /// The ZYX intrinsic Euler angles `(roll, pitch, yaw)`, the inverse of
    /// [`Quaternion::from_euler_zyx`]. Assumes a unit quaternion. At the gimbal-lock poles
    /// (`pitch = ±π/2`) the roll/yaw split is not unique; this returns `pitch = ±π/2` and a
    /// consistent roll/yaw.
    #[inline]
    pub fn to_euler_zyx(self) -> (T, T, T) {
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);
        let two = T::TWO;
        let sinp = two * (w * y - z * x);
        if sinp.abs() >= T::ONE - small_angle_sq::<T>() {
            // Gimbal lock: at pitch = ±π/2 the standard roll/yaw formulas divide 0/0. Snap pitch
            // to the pole, fix roll = 0, and fold the whole rotation into yaw (sign set by the
            // pole). Reconstruction still matches; only the roll/yaw split is not unique here.
            let pitch = (T::PI * T::HALF).copysign(sinp);
            let a = two * x.atan2(w);
            let yaw = if sinp > T::ZERO { -a } else { a };
            return (T::ZERO, pitch, yaw);
        }
        let roll = (two * (w * x + y * z)).atan2(T::ONE - two * (x * x + y * y));
        let pitch = sinp.asin();
        let yaw = (two * (w * z + x * y)).atan2(T::ONE - two * (y * y + z * z));
        (roll, pitch, yaw)
    }

    /// Rotates a point by the sandwich product `q · (0, v) · q⁻¹`. Assumes a unit quaternion.
    #[inline]
    pub fn transform_point(self, v: Vector<3, T>) -> Vector<3, T> {
        let [x, y, z] = *v.as_array();
        let p = Quaternion {
            w: T::ZERO,
            x,
            y,
            z,
        };
        let r = self * p * self.conjugate();
        Vector::new([r.x, r.y, r.z])
    }

    /// Spherical linear interpolation from `self` (`t = 0`) to `other` (`t = 1`). Assumes unit
    /// quaternions. Takes the shortest path (the `other` quaternion is negated when the dot
    /// product is negative) and falls back to normalized linear interpolation when the endpoints
    /// are nearly parallel. The result is renormalized.
    #[inline]
    pub fn slerp(self, other: Self, t: T) -> Self {
        let mut d = self.dot(other);
        let mut q2 = other;
        if d < T::ZERO {
            d = -d;
            q2 = -q2;
        }
        if d > T::ONE - T::EPSILON {
            // Endpoints nearly identical: the great-arc formula divides by ~0, so lerp instead.
            return (self * (T::ONE - t) + q2 * t).normalized();
        }
        let theta = d.acos();
        let sin_theta = theta.sin();
        let s0 = ((T::ONE - t) * theta).sin() / sin_theta;
        let s1 = (t * theta).sin() / sin_theta;
        (self * s0 + q2 * s1).normalized()
    }
}

impl<T: Numeric> Add for Quaternion<T> {
    type Output = Self;
    #[inline]
    fn add(self, r: Self) -> Self {
        Quaternion {
            w: self.w + r.w,
            x: self.x + r.x,
            y: self.y + r.y,
            z: self.z + r.z,
        }
    }
}

impl<T: Numeric> Sub for Quaternion<T> {
    type Output = Self;
    #[inline]
    fn sub(self, r: Self) -> Self {
        Quaternion {
            w: self.w - r.w,
            x: self.x - r.x,
            y: self.y - r.y,
            z: self.z - r.z,
        }
    }
}

impl<T: Numeric> Neg for Quaternion<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Quaternion {
            w: -self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

/// Scalar multiplication.
impl<T: Numeric> Mul<T> for Quaternion<T> {
    type Output = Self;
    #[inline]
    fn mul(self, s: T) -> Self {
        Quaternion {
            w: self.w * s,
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }
}

/// The Hamilton product `(w1 + v1)(w2 + v2) = w1·w2 − v1·v2 + w1·v2 + w2·v1 + v1×v2`. This is the
/// exact algebra product and does not renormalize.
impl<T: Numeric> Mul for Quaternion<T> {
    type Output = Self;
    #[inline]
    fn mul(self, r: Self) -> Self {
        Quaternion {
            w: self.w * r.w - self.x * r.x - self.y * r.y - self.z * r.z,
            x: self.w * r.x + self.x * r.w + self.y * r.z - self.z * r.y,
            y: self.w * r.y - self.x * r.z + self.y * r.w + self.z * r.x,
            z: self.w * r.z + self.x * r.y - self.y * r.x + self.z * r.w,
        }
    }
}
