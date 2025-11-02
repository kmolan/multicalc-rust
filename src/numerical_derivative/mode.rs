/// @brief Finite difference modes for numerical differentiation.
///
/// In most cases, `Central` is recommended for the highest accuracy.
/// If unsure, start with `Central` and then tweak based on results.
///
/// @note The accuracy of results also depends on the chosen step size.
#[derive(Debug, Copy, Clone)]
pub enum FiniteDifferenceMode {
    /// @brief Forward difference method.
    ///
    /// Uses f'(x) ≈ (f(x + h) - f(x)) / h.
    Forward,

    /// @brief Backward difference method.
    ///
    /// Uses f'(x) ≈ (f(x) - f(x - h)) / h.
    Backward,

    /// @brief Central difference method.
    ///
    /// Uses f'(x) ≈ (f(x + h) - f(x - h)) / (2h).  
    /// Provides the highest accuracy among finite difference schemes.
    Central,
}

/// @brief Default step size used by the finite difference module.
pub const DEFAULT_STEP_SIZE: f64 = 1.0e-5;

/// @brief Default multiplier applied to the step size after each iteration.
/// Primarily relevant for higher-order derivatives.
pub const DEFAULT_STEP_SIZE_MULTIPLIER: f64 = 10.0;
