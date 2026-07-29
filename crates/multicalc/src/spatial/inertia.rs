//! A rigid body's mass distribution.

use crate::error::SpatialError;
use crate::linear_algebra::{Matrix, Matrix3D, Vector3D};
use crate::scalar::Numeric;

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
        for (row, col) in [(0, 1), (0, 2), (1, 2)] {
            let upper = rotational_inertia[(row, col)];
            let lower = rotational_inertia[(col, row)];
            // Scaling by the larger of the two keeps this meaningful for both f32 and f64 and for
            // large tensors; the floor at one stops it collapsing to exact equality near zero.
            let scale = upper.abs().max(lower.abs()).max(T::ONE);
            if (upper - lower).abs() > T::EPSILON_X30 * scale {
                return Err(SpatialError::NotSymmetric);
            }
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

    /// `true` when every stored number is finite.
    #[inline]
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.mass.is_finite()
            && self.center_of_mass.is_finite()
            && self.rotational_inertia.is_finite()
    }
}
