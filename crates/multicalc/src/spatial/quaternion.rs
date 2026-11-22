//! Quaternions for 3D rotation.
//!
//! [`Quaternion`] is a single quaternion type in the Hamilton convention, following the model of
//! Eigen, glam, and ROS `tf2`: contains both the raw quaternion algebra and the rotation
//! helpers. A quaternion represents a rotation only when it has unit norm. The rotation
//! *constructors* ([`Quaternion::from_axis_angle`], [`Quaternion::from_scaled_axis`],
//! [`Quaternion::from_euler_zyx`], [`Quaternion::from_two_vectors`],
//! [`Quaternion::try_from_rotation_matrix`]) return unit output;
//! the rotation *queries* ([`Quaternion::to_rotation_matrix`], [`Quaternion::slerp`],
//! [`Quaternion::transform_point`], [`Quaternion::inverse_transform_point`],
//! [`Quaternion::rotation_angle_to`], [`Quaternion::to_euler_zyx`], [`Quaternion::to_axis_angle`],
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

use crate::linear_algebra::{Matrix, Matrix3D, Vector, Vector3D};
use crate::scalar::Numeric;

use crate::spatial::{small_angle, small_angle_sq};

/// A quaternion `w + x·i + y·j + z·k`, stored scalar-first.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion<T: Numeric = f64> {
    w: T,
    x: T,
    y: T,
    z: T,
}

/// The unit vector along `v`, or `None` when `v` carries no direction: a non-finite component,
/// or all components zero.
///
/// Components are divided by the largest of them in magnitude before the norm is taken, which
/// keeps the direction of inputs whose squared norm is not representable. Normalizing straight
/// through `Vector::try_normalized` does not: `norm()` squares first, so `[1e200, 0, 0]` overflows
/// to an infinite norm and comes back as the zero vector, and `[1e-200, 0, 0]` underflows to a
/// zero norm and comes back as `None`, though both name a perfectly good axis.
#[inline]
#[must_use]
fn unit_direction<T: Numeric>(vector: Vector<3, T>) -> Option<Vector<3, T>> {
    if !vector.is_finite() {
        return None;
    }

    let [x, y, z] = *vector.as_array();
    let scale = x.abs().max(y.abs()).max(z.abs());

    if scale == T::ZERO {
        return None;
    }

    // Every component is now at most 1 in magnitude and at least one is exactly 1, so the squared
    // norm sits in `[1, 3]` and the division below is exact in the exponent.
    (vector / scale).try_normalized()
}

impl<T: Numeric> Quaternion<T> {
    /// A quaternion from its four components, in `[w, x, y, z]` order.
    #[inline]
    #[must_use]
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
    #[must_use]
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
    #[must_use]
    pub fn from_array(a: [T; 4]) -> Self {
        let [w, x, y, z] = a;
        Quaternion { w, x, y, z }
    }

    /// Builds a quaternion from a scalar part and a vector part.
    #[inline]
    #[must_use]
    pub fn from_scalar_vector(w: T, vector: Vector3D<T>) -> Self {
        let [x, y, z] = *vector.as_array();
        Quaternion { w, x, y, z }
    }

    /// The scalar (real) component.
    #[inline]
    #[must_use]
    pub fn w(self) -> T {
        self.w
    }

    /// The `i` component.
    #[inline]
    #[must_use]
    pub fn x(self) -> T {
        self.x
    }

    /// The `j` component.
    #[inline]
    #[must_use]
    pub fn y(self) -> T {
        self.y
    }

    /// The `k` component.
    #[inline]
    #[must_use]
    pub fn z(self) -> T {
        self.z
    }

    /// The vector (imaginary) part `[x, y, z]`.
    #[inline]
    pub fn vec(self) -> Vector3D<T> {
        Vector::new([self.x, self.y, self.z])
    }

    /// The components as a `[w, x, y, z]` array.
    #[inline]
    #[must_use]
    pub fn as_array(self) -> [T; 4] {
        [self.w, self.x, self.y, self.z]
    }

    /// The conjugate `w − x·i − y·j − z·k`.
    #[inline]
    #[must_use]
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
    #[must_use]
    pub fn norm_squared(self) -> T {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// The Euclidean norm.
    #[inline]
    #[must_use]
    pub fn norm(self) -> T {
        self.norm_squared().sqrt()
    }

    /// The four-component dot product.
    #[inline]
    #[must_use]
    pub fn dot(self, rhs: Self) -> T {
        self.w * rhs.w + self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// The inverse `conjugate / norm²`. For a unit quaternion this equals the conjugate. Yields
    /// `inf`/`NaN` for a zero quaternion, exactly as plain-float division does elsewhere.
    #[inline]
    #[must_use]
    pub fn inverse(self) -> Self {
        self.conjugate() * self.norm_squared().recip()
    }

    /// This quaternion scaled to unit norm. Yields `NaN` components for a zero quaternion, as
    /// plain-float division does; use [`Quaternion::try_normalized`] for a checked version.
    #[inline]
    #[must_use]
    pub fn normalized(self) -> Self {
        self * self.norm().recip()
    }

    /// This quaternion scaled to unit norm, or `None` if the norm is non-finite or underflows.
    #[inline]
    #[must_use]
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
    #[must_use]
    pub fn exp(self) -> Self {
        let vn_sq = self.x * self.x + self.y * self.y + self.z * self.z;
        let exp_scalar = self.w.exp();
        let (cos_v, sinc_v) = if vn_sq < small_angle_sq::<T>() {
            // Taylor in ‖v‖²; avoids `sqrt(0)`, whose derivative is NaN.
            (T::ONE - vn_sq / T::TWO, T::ONE - vn_sq / T::from_f64(6.0))
        } else {
            let vector_norm = vn_sq.sqrt();
            (vector_norm.cos(), vector_norm.sin() / vector_norm)
        };
        let scale = exp_scalar * sinc_v;
        Quaternion {
            w: exp_scalar * cos_v,
            x: self.x * scale,
            y: self.y * scale,
            z: self.z * scale,
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
    #[must_use]
    pub fn log(self) -> Self {
        let n = self.norm();
        let vector_norm = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        // Restrict the small-vector fallback to w > 0; near the negative real axis θ ≈ π, not 0,
        // so 1/‖q‖ would be the wrong coefficient (see the doc note above).
        let coeff = if vector_norm < small_angle::<T>() && self.w > T::ZERO {
            n.recip()
        } else {
            vector_norm.atan2(self.w) / vector_norm
        };
        Quaternion {
            w: n.log(),
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
    /// let axis = Vector::new([0.0, 0.0, 1.0]);
    /// let angle = core::f64::consts::FRAC_PI_2;
    /// let rotation = Quaternion::from_axis_angle(axis, angle);
    ///
    /// let point = Vector::new([1.0, 0.0, 0.0]);
    /// let rotated = rotation.transform_point(point);
    /// assert!((rotated[0] - 0.0).abs() < 1e-12);
    /// assert!((rotated[1] - 1.0).abs() < 1e-12);
    /// assert!((rotated[2] - 0.0).abs() < 1e-12);
    /// ```
    #[inline]
    #[must_use]
    pub fn from_axis_angle(axis: Vector3D<T>, angle: T) -> Self {
        let axis_norm = axis.dot(axis).sqrt();
        if axis_norm <= T::EPSILON {
            return Self::identity();
        }
        let half = angle * T::HALF;
        let scale = half.sin() / axis_norm;
        let [x, y, z] = *axis.as_array();
        Quaternion {
            w: half.cos(),
            x: x * scale,
            y: y * scale,
            z: z * scale,
        }
    }

    /// The rotation whose axis-angle is the rotation vector `φ = θ·n̂` (the so(3) exponential
    /// map), as a unit quaternion. Near `θ = 0` the `cos(θ/2)` and `sin(θ/2)/θ` factors use
    /// Taylor series in `θ²`, so no `sqrt` is taken there and the AD derivative stays finite at
    /// `φ = 0` (a robot at rest).
    #[inline]
    #[must_use]
    pub fn from_scaled_axis(rotvec: Vector3D<T>) -> Self {
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
        let [x, y, z] = *rotvec.as_array();
        Quaternion {
            w,
            x: x * scale,
            y: y * scale,
            z: z * scale,
        }
    }

    /// The rotation from ZYX intrinsic Euler angles: `R = Rz(yaw)·Ry(pitch)·Rx(roll)`. The result
    /// is a unit quaternion.
    #[inline]
    #[must_use]
    pub fn from_euler_zyx(roll: T, pitch: T, yaw: T) -> Self {
        let (cos_roll, sin_roll) = ((roll * T::HALF).cos(), (roll * T::HALF).sin());
        let (cos_pitch, sin_pitch) = ((pitch * T::HALF).cos(), (pitch * T::HALF).sin());
        let (cos_yaw, sin_yaw) = ((yaw * T::HALF).cos(), (yaw * T::HALF).sin());
        Quaternion {
            w: cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw,
            x: sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw,
            y: cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw,
            z: cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw,
        }
    }

    /// The shortest rotation taking the direction `from` onto the direction `to`, as a unit
    /// quaternion. Only the directions matter: both inputs are normalized internally, over the
    /// whole finite range rather than only where the squared norm is representable. An input that
    /// is non-finite or all zeros carries no direction, and yields the identity rotation.
    ///
    /// When the two directions are exactly opposite the rotation is by pi about an axis the inputs
    /// do not determine: any perpendicular of `from` maps it onto `to`, and one is picked here.
    ///
    /// Near-opposite inputs stay accurate to a few eps, with no band of degraded accuracy around
    /// pi. Two things buy that. The half-angle is taken from `norm(a + b)` rather than from the
    /// `sqrt(2(1 + a.b))` form, which loses relative accuracy to cancellation exactly there. And
    /// the axis is `a x (a + b)`, which is perpendicular to `a` whatever error `a + b` carries, so
    /// `from` is always rotated within the correct plane; the `eps/delta` relative error of
    /// `a + b` at a separation `delta` from pi is then scaled back down by the `delta` it is
    /// rotated through, leaving an error flat in eps rather than growing as pi is approached.
    ///
    /// ```
    /// use multicalc::spatial::Quaternion;
    /// use multicalc::linear_algebra::Vector;
    ///
    /// let from = Vector::new([1.0, 0.0, 0.0]);
    /// let target = Vector::new([0.0, 2.0, 0.0]);
    /// let rotated = Quaternion::from_two_vectors(from, target).transform_point(from);
    ///
    /// assert!((rotated - target.normalized()).norm() < 1e-12);
    /// ```
    #[inline]
    #[must_use]
    pub fn from_two_vectors(from: Vector<3, T>, target: Vector<3, T>) -> Self {
        let (Some(a), Some(b)) = (unit_direction(from), unit_direction(target)) else {
            return Self::identity();
        };
        // `norm(a + b) = 2.cos(theta/2)` for unit inputs, theta the angle between them. Getting the
        // half-angle from the sum rather than from `sqrt(2(1 + a.b))` matters near `theta = pi`: there
        // `1 + a.b` is the difference of two nearly equal numbers, so it loses all relative
        // accuracy at `eps/delta^2` for a separation `delta` from `pi`, while the components of `a + b`
        // each keep theirs to `eps/delta`.
        let h = a + b;
        let h_sq = h.norm_squared();
        // Only the exactly degenerate case needs the fallback, not a band around it. The axis below
        // is `a x h`, which is perpendicular to `a` however inaccurate `h` is, so `a` always lands
        // in the right plane and the leftover error is `delta` times the `eps/delta` relative error
        // of `h`, i.e. flat in `eps`. That holds until `h` is exactly zero and there is no axis at
        // all. An `h_sq` that has underflowed to zero from a nonzero `h` lands here too, and is
        // just as well served: `h_sq = delta^2`, so that needs `delta < 1e-154`, far below the
        // point where a pi turn is the right answer anyway.
        if h_sq == T::ZERO {
            // Antiparallel, where the inputs pin down no axis. Build the pi rotation by crossing
            // `a` with the principal axis it is least aligned with, which keeps that cross
            // product's norm at 0.816 or above.
            let [x, y, z] = *a.as_array();
            let principal = if x.abs() <= y.abs() && x.abs() <= z.abs() {
                Vector::new([T::ONE, T::ZERO, T::ZERO])
            } else if y.abs() <= z.abs() {
                Vector::new([T::ZERO, T::ONE, T::ZERO])
            } else {
                Vector::new([T::ZERO, T::ZERO, T::ONE])
            };

            return Self::from_scalar_vector(T::ZERO, a.cross(principal).normalized());
        }
        // The intended quaternion is `(s/2, (a x b)/s)` for `s = norm(a + b)`, since `a x (a + b)`
        // is `a x b` with magnitude `sin(theta)` and `sin(theta)/s = sin(theta/2)`. Scaling it by
        // `s` clears both divisions and leaves `(h_sq/2, a x h)`, whose norm is exactly `s`, so one
        // normalization recovers the unit result and replaces the `sqrt` and `recip` it drops.
        // Dividing by `s` instead leaves the norm off by `(eps/delta)^2` just above the cutoff,
        // where `s` is small enough for its own rounding to show; this way the norm is unit to eps
        // over the whole range.
        Self::from_scalar_vector(h_sq * T::HALF, a.cross(h)).normalized()
    }

    /// Builds a unit quaternion from a rotation matrix by Shepperd's method (the largest of the
    /// trace and the three diagonal terms is the pivot, for numerical stability). A proper
    /// (orthonormal, determinant +1) rotation is assumed; `None` guards only against a degenerate
    /// pivot that would divide by zero.
    #[inline]
    #[must_use]
    pub fn try_from_rotation_matrix(matrix: Matrix3D<T>) -> Option<Self> {
        let quarter = T::from_f64(0.25);
        let [[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]] = matrix.into_array();
        let trace = m00 + m11 + m22;
        let quaternion = if trace > T::ZERO {
            let scale = (trace + T::ONE).sqrt() * T::TWO; // s = 4·w
            Quaternion::new(
                quarter * scale,
                (m21 - m12) / scale,
                (m02 - m20) / scale,
                (m10 - m01) / scale,
            )
        } else if m00 > m11 && m00 > m22 {
            let scale = (T::ONE + m00 - m11 - m22).sqrt() * T::TWO; // s = 4·x
            Quaternion::new(
                (m21 - m12) / scale,
                quarter * scale,
                (m01 + m10) / scale,
                (m02 + m20) / scale,
            )
        } else if m11 > m22 {
            let scale = (T::ONE + m11 - m00 - m22).sqrt() * T::TWO; // s = 4·y
            Quaternion::new(
                (m02 - m20) / scale,
                (m01 + m10) / scale,
                quarter * scale,
                (m12 + m21) / scale,
            )
        } else {
            let scale = (T::ONE + m22 - m00 - m11).sqrt() * T::TWO; // s = 4·z
            Quaternion::new(
                (m10 - m01) / scale,
                (m02 + m20) / scale,
                (m12 + m21) / scale,
                quarter * scale,
            )
        };
        quaternion.try_normalized()
    }

    /// The 3×3 rotation matrix. Assumes a unit quaternion; call [`Quaternion::normalized`] first
    /// if it may have drifted.
    #[inline]
    pub fn to_rotation_matrix(self) -> Matrix3D<T> {
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
    pub fn to_axis_angle(self) -> (Vector3D<T>, T) {
        let quaternion = if self.w < T::ZERO { -self } else { self };
        let vector_norm = (quaternion.x * quaternion.x
            + quaternion.y * quaternion.y
            + quaternion.z * quaternion.z)
            .sqrt();
        if vector_norm <= T::EPSILON {
            return (Vector::new([T::ONE, T::ZERO, T::ZERO]), T::ZERO);
        }
        let inv = vector_norm.recip();
        (
            Vector::new([quaternion.x * inv, quaternion.y * inv, quaternion.z * inv]),
            T::TWO * vector_norm.atan2(quaternion.w),
        )
    }

    /// The rotation vector `φ = θ·n̂` (the so(3) logarithm), the inverse of
    /// [`Quaternion::from_scaled_axis`]. Assumes a unit quaternion; the shortest-path sign fix is
    /// applied so `‖φ‖ ≤ π`.
    #[inline]
    pub fn to_scaled_axis(self) -> Vector3D<T> {
        let quaternion = if self.w < T::ZERO { -self } else { self };
        let vector_norm = (quaternion.x * quaternion.x
            + quaternion.y * quaternion.y
            + quaternion.z * quaternion.z)
            .sqrt();
        let coeff = if vector_norm < small_angle::<T>() {
            T::TWO
        } else {
            (T::TWO * vector_norm.atan2(quaternion.w)) / vector_norm
        };
        Vector::new([
            quaternion.x * coeff,
            quaternion.y * coeff,
            quaternion.z * coeff,
        ])
    }

    /// The ZYX intrinsic Euler angles `(roll, pitch, yaw)`, the inverse of
    /// [`Quaternion::from_euler_zyx`]. Assumes a unit quaternion. At the gimbal-lock poles
    /// (`pitch = ±π/2`) the roll/yaw split is not unique; this returns `pitch = ±π/2` and a
    /// consistent roll/yaw.
    #[inline]
    #[must_use]
    pub fn to_euler_zyx(self) -> (T, T, T) {
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);
        let two = T::TWO;
        let sinp = two * (w * y - z * x);
        if sinp.abs() >= T::ONE - small_angle::<T>() {
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
    pub fn transform_point(self, point: Vector3D<T>) -> Vector3D<T> {
        let [x, y, z] = *point.as_array();
        let pure = Quaternion {
            w: T::ZERO,
            x,
            y,
            z,
        };
        let rotated = self * pure * self.conjugate();
        Vector::new([rotated.x, rotated.y, rotated.z])
    }

    /// Rotates a point by the inverse rotation, the sandwich product `q^{-1} . (0, v) . q`.
    /// Assumes a unit quaternion, for which this exactly undoes [`Quaternion::transform_point`]
    /// without the caller building the conjugate first.
    ///
    /// ```
    /// use multicalc::spatial::Quaternion;
    /// use multicalc::linear_algebra::Vector;
    ///
    /// let rotation = Quaternion::from_euler_zyx(0.3, -0.7, 1.2);
    /// let point = Vector::new([1.0, 2.0, 3.0]);
    /// let roundtrip = rotation.inverse_transform_point(rotation.transform_point(point));
    ///
    /// assert!((roundtrip - point).norm() < 1e-12);
    /// ```
    // No `#[must_use]`: the returned `Vector` already carries one, and clippy rejects the
    // duplicate. `rotation_angle_to` returns a bare `T` and does need its own.
    #[inline]
    pub fn inverse_transform_point(self, point: Vector<3, T>) -> Vector<3, T> {
        self.conjugate().transform_point(point)
    }

    /// The angle in `[0, pi]` between two rotations: the rotation angle of the relative rotation
    /// `self^{-1} . other`.
    /// Assumes unit quaternions.
    ///
    /// The shortest-path rule applies, so the double cover is respected: `q` and `-q` are the same
    /// rotation and are zero apart. Measured with `atan2` rather than as `acos` of the dot product,
    /// which loses about half the significant digits for nearly equal rotations.
    ///
    /// ```
    /// use multicalc::spatial::Quaternion;
    /// use multicalc::linear_algebra::Vector;
    ///
    /// let axis = Vector::<3, f64>::new([0.0, 0.0, 1.0]);
    /// let first = Quaternion::from_axis_angle(axis, 0.5);
    /// let second = Quaternion::from_axis_angle(axis, 1.25);
    ///
    /// assert!((first.rotation_angle_to(second) - 0.75).abs() < 1e-12);
    /// ```
    #[inline]
    #[must_use]
    pub fn rotation_angle_to(self, other: Self) -> T {
        let relative = self.conjugate() * other;
        let vector_norm =
            (relative.x * relative.x + relative.y * relative.y + relative.z * relative.z).sqrt();

        // `|w|` rather than `w` folds the `theta > pi` half of the double cover back onto `2pi - theta`.
        T::TWO * vector_norm.atan2(relative.w.abs())
    }

    /// Spherical linear interpolation from `self` (`t = 0`) to `other` (`t = 1`). Assumes unit
    /// quaternions. Takes the shortest path (the `other` quaternion is negated when the dot
    /// product is negative) and falls back to normalized linear interpolation when the endpoints
    /// are nearly parallel. The result is renormalized.
    #[inline]
    #[must_use]
    pub fn slerp(self, other: Self, amount: T) -> Self {
        let mut dot_product = self.dot(other);
        let mut other_shortest = other;
        if dot_product < T::ZERO {
            dot_product = -dot_product;
            other_shortest = -other_shortest;
        }
        if dot_product > T::ONE - T::EPSILON {
            // Endpoints nearly identical: the great-arc formula divides by ~0, so lerp instead.
            return (self * (T::ONE - amount) + other_shortest * amount).normalized();
        }
        let theta = dot_product.acos();
        let sin_theta = theta.sin();
        let weight_start = ((T::ONE - amount) * theta).sin() / sin_theta;
        let weight_end = (amount * theta).sin() / sin_theta;
        (self * weight_start + other_shortest * weight_end).normalized()
    }
}

impl<T: Numeric> Add for Quaternion<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Quaternion {
            w: self.w + rhs.w,
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl<T: Numeric> Sub for Quaternion<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Quaternion {
            w: self.w - rhs.w,
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
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
    fn mul(self, scalar: T) -> Self {
        Quaternion {
            w: self.w * scalar,
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

/// The Hamilton product `(w1 + v1)(w2 + v2) = w1·w2 − v1·v2 + w1·v2 + w2·v1 + v1×v2`. This is the
/// exact algebra product and does not renormalize.
impl<T: Numeric> Mul for Quaternion<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Quaternion {
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        }
    }
}

impl<T: Numeric> Default for Quaternion<T> {
    /// Returns the identity as default.
    ///
    /// ```
    /// use multicalc::Quaternion;
    ///
    /// let default_quaternion = Quaternion::default();
    /// let quaternion = Quaternion::<f64>::from_array([1.0, 2.0, 3.0, 4.0]);
    ///
    /// assert_eq!(quaternion * default_quaternion, quaternion);
    /// ```
    fn default() -> Self {
        Self::identity()
    }
}
