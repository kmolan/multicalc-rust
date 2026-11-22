//! A rigid body's mass distribution.

use crate::error::SpatialError;
use crate::linear_algebra::{Matrix, Matrix3D, Matrix6D, Vector, Vector3D};
use crate::scalar::Numeric;
use crate::spatial::lie::skew3;
use crate::spatial::{Twist, Wrench};

/// How a rigid body's mass is spread out: how much there is, where it balances, and how hard it is
/// to spin.
///
/// The rotational inertia is always stated about the balance point, along the body's own axes. To
/// ask about any other point, use [`inertia_about`](SpatialInertia::inertia_about) — a body is
/// harder to spin about a point it does not balance on.
///
/// ```
/// use multicalc::linear_algebra::{Matrix, Vector};
/// use multicalc::spatial::SpatialInertia;
/// let body = SpatialInertia::new(
///     2.0_f64,
///     Vector::new([0.0, 0.0, 0.0]),
///     Matrix::from_diagonal([1.0, 1.0, 1.0]),
/// )
/// .unwrap();
/// assert_eq!(body.mass(), 2.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialInertia<T: Numeric = f64> {
    mass: T,
    center_of_mass: Vector3D<T>,
    rotational_inertia: Matrix3D<T>,
}

impl<T: Numeric> SpatialInertia<T> {
    /// Unchecked constructor for parts already satisfying [`SpatialInertia::new`]'s constraints.
    ///
    /// Crate-internal: a body derived from a validated one — transformed or merged — cannot
    /// violate them.
    #[inline]
    #[must_use]
    pub(crate) fn from_parts(
        mass: T,
        center_of_mass: Vector3D<T>,
        rotational_inertia: Matrix3D<T>,
    ) -> Self {
        SpatialInertia {
            mass,
            center_of_mass,
            rotational_inertia,
        }
    }

    /// A body from its mass, the point it balances about, and how it resists being spun about that
    /// point.
    ///
    /// The inertia has to read the same across the diagonal and carry a positive diagonal, and the
    /// mass has to be positive.
    ///
    /// ```
    /// use multicalc::linear_algebra::{Matrix, Vector};
    /// use multicalc::spatial::SpatialInertia;
    ///
    /// // A 2 kg body that balances 10 cm above its own origin, and is harder to spin about the
    /// // up axis than the other two.
    /// let body = SpatialInertia::new(
    ///     2.0_f64,
    ///     Vector::new([0.0, 0.0, 0.1]),
    ///     Matrix::from_diagonal([0.05, 0.05, 0.08]),
    /// )
    /// .unwrap();
    ///
    /// assert_eq!(body.mass(), 2.0);
    /// assert_eq!(body.center_of_mass(), Vector::new([0.0, 0.0, 0.1]));
    /// assert_eq!(body.rotational_inertia().diagonal(), [0.05, 0.05, 0.08]);
    /// ```
    pub fn new(
        mass: T,
        center_of_mass: Vector3D<T>,
        rotational_inertia: Matrix3D<T>,
    ) -> Result<Self, SpatialError> {
        if !mass.is_finite() || !center_of_mass.is_finite() || !rotational_inertia.is_finite() {
            return Err(SpatialError::NonFinite);
        }
        if mass <= T::ZERO {
            return Err(SpatialError::NonPositiveMass);
        }
        if !rotational_inertia.is_symmetric() {
            return Err(SpatialError::NotSymmetric);
        }
        for index in 0..3 {
            if rotational_inertia[(index, index)] <= T::ZERO {
                return Err(SpatialError::NonPositiveInertia);
            }
        }
        Ok(SpatialInertia {
            mass,
            center_of_mass,
            rotational_inertia,
        })
    }

    /// A body whose resistance to spinning is given along its own axes, with nothing coupling one
    /// axis to another.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::SpatialInertia;
    /// let body = SpatialInertia::from_diagonal_inertia(
    ///     2.0_f64,
    ///     Vector::new([0.0, 0.0, 0.0]),
    ///     Vector::new([1.0, 2.0, 3.0]),
    /// )
    /// .unwrap();
    /// assert_eq!(body.rotational_inertia().diagonal(), [1.0, 2.0, 3.0]);
    /// ```
    pub fn from_diagonal_inertia(
        mass: T,
        center_of_mass: Vector3D<T>,
        diagonal: Vector3D<T>,
    ) -> Result<Self, SpatialError> {
        Self::new(
            mass,
            center_of_mass,
            Matrix::from_diagonal(diagonal.into_array()),
        )
    }

    /// How much mass the body carries.
    #[inline]
    #[must_use]
    pub fn mass(self) -> T {
        self.mass
    }

    /// The point the body balances about, in body axes.
    #[inline]
    pub fn center_of_mass(self) -> Vector3D<T> {
        self.center_of_mass
    }

    /// How the body resists being spun about its balance point, in body axes.
    #[inline]
    pub fn rotational_inertia(self) -> Matrix3D<T> {
        self.rotational_inertia
    }

    /// How the body resists being spun about some other point.
    ///
    /// Moving the reference point away from where the body balances makes it harder to spin, by an
    /// amount set by the mass and how far the point moved.
    ///
    /// ```
    /// use multicalc::linear_algebra::{Matrix, Vector};
    /// use multicalc::spatial::SpatialInertia;
    /// let body = SpatialInertia::new(
    ///     1.0_f64,
    ///     Vector::new([0.0, 0.0, 0.0]),
    ///     Matrix::from_diagonal([1.0, 1.0, 1.0]),
    /// )
    /// .unwrap();
    /// // A metre along x: spinning about x is unchanged, about y and z it doubles.
    /// let shifted = body.inertia_about(Vector::new([1.0, 0.0, 0.0]));
    /// assert_eq!(shifted.diagonal(), [1.0, 2.0, 2.0]);
    /// ```
    pub fn inertia_about(self, point: Vector3D<T>) -> Matrix3D<T> {
        let offset = point - self.center_of_mass;
        let distance_squared = offset.dot(offset);
        Matrix::from_fn(|row, col| {
            let spread = if row == col {
                distance_squared
            } else {
                T::ZERO
            };
            self.rotational_inertia[(row, col)] + self.mass * (spread - offset[row] * offset[col])
        })
    }

    /// Spatial momentum `I·v` about the body-frame origin: `[m(v + ω×c) ; I_c·ω + c×linear]`.
    ///
    /// Two cross products and one 3×3 product; the origin-referred inertia is never formed.
    ///
    /// ```
    /// use multicalc::linear_algebra::{Matrix, Vector};
    /// use multicalc::spatial::{SpatialInertia, Twist};
    /// let body = SpatialInertia::new(
    ///     2.0_f64,
    ///     Vector::new([0.0, 0.0, 0.0]),
    ///     Matrix::from_diagonal([1.0, 1.0, 1.0]),
    /// )
    /// .unwrap();
    /// // c = 0, ω = 0, so only m·v survives.
    /// let motion = Twist::new(Vector::new([1.0, 0.0, 0.0]), Vector::zeros());
    /// assert_eq!(body.momentum(motion).as_array(), [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    /// ```
    #[inline]
    #[must_use]
    pub fn momentum(self, velocity: Twist<T>) -> Wrench<T> {
        let balance_point_velocity =
            velocity.linear() + velocity.angular().cross(self.center_of_mass);
        let linear = balance_point_velocity.scale(self.mass);
        let angular =
            self.rotational_inertia * velocity.angular() + self.center_of_mass.cross(linear);
        Wrench::new(linear, angular)
    }

    /// The velocity-product term `v ×* (I·v)`, as RNEA needs it.
    ///
    /// ```
    /// use multicalc::linear_algebra::{Matrix, Vector};
    /// use multicalc::spatial::{SpatialInertia, Twist};
    /// let body = SpatialInertia::new(
    ///     2.0_f64,
    ///     Vector::new([0.0, 0.0, 0.0]),
    ///     Matrix::from_diagonal([1.0, 1.0, 1.0]),
    /// )
    /// .unwrap();
    /// // Isotropic inertia: ω × (I_c·ω) vanishes.
    /// let spin = Twist::new(Vector::zeros(), Vector::new([0.0, 0.0, 3.0]));
    /// assert!(body.bias_wrench(spin).to_vector().norm() < 1e-12);
    /// ```
    #[inline]
    #[must_use]
    pub fn bias_wrench(self, velocity: Twist<T>) -> Wrench<T> {
        velocity.cross_wrench(self.momentum(velocity))
    }

    /// Kinetic energy `½·vᵀ·I·v`, evaluated block-wise.
    ///
    /// ```
    /// use multicalc::linear_algebra::{Matrix, Vector};
    /// use multicalc::spatial::{SpatialInertia, Twist};
    /// let body = SpatialInertia::new(
    ///     2.0_f64,
    ///     Vector::new([0.0, 0.0, 0.0]),
    ///     Matrix::from_diagonal([1.0, 1.0, 1.0]),
    /// )
    /// .unwrap();
    /// let motion = Twist::new(Vector::new([3.0, 0.0, 0.0]), Vector::zeros());
    /// assert_eq!(body.kinetic_energy(motion), 9.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn kinetic_energy(self, velocity: Twist<T>) -> T {
        T::HALF * velocity.dot_wrench(self.momentum(velocity))
    }

    /// The 6×6 inertia about the body-frame origin, in `[v; ω]` blocks:
    /// `[[m·1₃, −m·[c]×], [m·[c]×, I_o]]`.
    ///
    /// Not hot-path — [`momentum`](SpatialInertia::momentum) and
    /// [`kinetic_energy`](SpatialInertia::kinetic_energy) never build it.
    ///
    /// ```
    /// use multicalc::linear_algebra::{Matrix, Vector};
    /// use multicalc::spatial::SpatialInertia;
    /// let body = SpatialInertia::new(
    ///     1.0_f64,
    ///     Vector::new([0.0, 0.0, 0.0]),
    ///     Matrix::from_diagonal([1.0, 1.0, 1.0]),
    /// )
    /// .unwrap();
    /// // c = 0, so the off-diagonal blocks vanish.
    /// let block = body.to_matrix();
    /// for row in 0..6 {
    ///     for col in 0..6 {
    ///         let want = if row == col { 1.0 } else { 0.0 };
    ///         assert!((block[(row, col)] - want).abs() < 1e-12);
    ///     }
    /// }
    /// ```
    #[inline]
    pub fn to_matrix(self) -> Matrix6D<T> {
        let balance_offset = skew3(self.center_of_mass);
        let about_origin = self.inertia_about(Vector::zeros());
        let mut entries = Matrix::zeros();
        for i in 0..3 {
            entries[(i, i)] = self.mass;
            for j in 0..3 {
                let offset = self.mass * balance_offset[(i, j)];
                entries[(i, j + 3)] = -offset;
                entries[(i + 3, j)] = offset;
                entries[(i + 3, j + 3)] = about_origin[(i, j)];
            }
        }
        entries
    }

    /// Two rigidly attached bodies as one, both stated in the same frame.
    ///
    /// `m = m₁ + m₂`, `c = (m₁c₁ + m₂c₂)/m`, and both inertias are shifted to `c` before adding.
    /// Same result as summing the two [`to_matrix`](SpatialInertia::to_matrix) forms.
    ///
    /// ```
    /// use multicalc::linear_algebra::{Matrix, Vector};
    /// use multicalc::spatial::SpatialInertia;
    /// let unit_inertia = Matrix::from_diagonal([1.0, 1.0, 1.0]);
    /// let left = SpatialInertia::new(1.0_f64, Vector::new([-1.0, 0.0, 0.0]), unit_inertia).unwrap();
    /// let right = SpatialInertia::new(1.0_f64, Vector::new([1.0, 0.0, 0.0]), unit_inertia).unwrap();
    /// // Equal masses, so c lands midway.
    /// let whole = left.combined(right);
    /// assert_eq!(whole.mass(), 2.0);
    /// assert!(whole.center_of_mass().norm() < 1e-12);
    /// ```
    #[inline]
    #[must_use]
    pub fn combined(self, other: Self) -> Self {
        let mass = self.mass + other.mass;
        let center_of_mass = (self.center_of_mass.scale(self.mass)
            + other.center_of_mass.scale(other.mass))
        .scale(T::ONE / mass);
        SpatialInertia {
            mass,
            center_of_mass,
            rotational_inertia: self.inertia_about(center_of_mass)
                + other.inertia_about(center_of_mass),
        }
    }

    /// `true` when every stored number is finite.
    #[inline]
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.mass.is_finite()
            && self.center_of_mass.is_finite()
            && self.rotational_inertia.is_finite()
    }
}
