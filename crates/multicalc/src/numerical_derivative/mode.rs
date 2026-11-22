use crate::scalar::Numeric;

/// The finite-difference stencil used to approximate a derivative.
///
/// Central is the most accurate for most cases; start there and tweak the mode and step size
/// if the result needs it.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FiniteDifferenceMode {
    /// Samples at the point and one step forward.
    Forward,
    /// Samples at the point and one step backward.
    Backward,
    /// Samples one step either side of the point; most accurate.
    Central,
}

impl FiniteDifferenceMode {
    /// Returns the default step size for this stencil and scalar type.
    ///
    /// Central differences balance rounding and truncation error at the cube root of machine
    /// epsilon. First-order one-sided differences use its square root.
    ///
    /// # Examples
    ///
    /// ```
    /// use multicalc::numerical_derivative::FiniteDifferenceMode;
    ///
    /// assert_eq!(
    ///     FiniteDifferenceMode::Central.default_step_size::<f32>(),
    ///     f32::EPSILON.cbrt()
    /// );
    /// assert_eq!(
    ///     FiniteDifferenceMode::Forward.default_step_size::<f64>(),
    ///     f64::EPSILON.sqrt()
    /// );
    /// ```
    #[inline]
    #[must_use]
    pub fn default_step_size<T: Numeric>(self) -> T {
        match self {
            FiniteDifferenceMode::Forward | FiniteDifferenceMode::Backward => T::EPSILON.sqrt(),
            FiniteDifferenceMode::Central => T::EPSILON.cbrt(),
        }
    }
}

/// Default central finite-difference step size for `f64`.
///
/// Generic code should use [`FiniteDifferenceMode::default_step_size`] so the value follows both
/// the scalar type and stencil.
pub const DEFAULT_STEP_SIZE: f64 = 6.055_454_452_393_343e-6;
/// Default factor the step is scaled by at each recursion level (third derivatives and higher).
pub const DEFAULT_STEP_SIZE_MULTIPLIER: f64 = 10.0;
