//! A controller output after a per-axis clamp.

use crate::linear_algebra::Vector;
use crate::scalar::Numeric;

/// A controller output after a per-axis clamp, with a flag for each axis that hit a limit.
///
/// [`Lqr::control_limited`](crate::control::Lqr::control_limited) and
/// [`GeometricAttitudeController::update`](crate::control::GeometricAttitudeController::update)
/// return it.
///
/// ```
/// use multicalc::control::Lqr;
/// use multicalc::linear_algebra::{Matrix, Vector};
///
/// let state_transition = Matrix::<2, 2>::new([[1.0, 0.1], [0.0, 1.0]]);
/// let input_model = Matrix::<2, 1>::new([[0.005], [0.1]]);
/// let controller = Lqr::new(state_transition, input_model, Matrix::identity(), Matrix::identity())
///     .unwrap()
///     .with_input_limits(Vector::new([-0.5]), Vector::new([0.5]))
///     .unwrap();
///
/// let far = controller.control_limited(Vector::new([10.0, 0.0]));
/// assert_eq!(far.output()[0], -0.5);
/// assert!(far.any_limited());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LimitedOutput<const N: usize, T: Numeric = f64> {
    output: Vector<N, T>,
    limited: [bool; N],
}

impl<const N: usize, T: Numeric> LimitedOutput<N, T> {
    /// Clamps each axis of `unlimited` to `[minimum[axis], maximum[axis]]`.
    pub(crate) fn clamp(
        unlimited: Vector<N, T>,
        minimum: Vector<N, T>,
        maximum: Vector<N, T>,
    ) -> Self {
        let output = Vector::from_fn(|axis| unlimited[axis].max(minimum[axis]).min(maximum[axis]));
        let limited = core::array::from_fn(|axis| {
            unlimited[axis] < minimum[axis] || unlimited[axis] > maximum[axis]
        });
        Self { output, limited }
    }

    /// The clamped output.
    #[inline]
    pub fn output(self) -> Vector<N, T> {
        self.output
    }

    /// `true` for each axis that hit a limit.
    #[inline]
    #[must_use]
    pub fn limited(self) -> [bool; N] {
        self.limited
    }

    /// `true` if any axis hit a limit.
    #[inline]
    #[must_use]
    pub fn any_limited(self) -> bool {
        self.limited.iter().any(|&axis_limited| axis_limited)
    }
}
