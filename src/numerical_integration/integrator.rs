/// @brief Base trait for single-variable numerical integration.
///
/// Defines the interface for performing numerical integration of a single-variable function.
/// Supports multiple integrations (nested integrals).
pub trait IntegratorSingleVariable: Default + Clone + Copy {
    /// @brief Generic n-th integration of a single-variable function.
    ///
    /// @tparam NUM_INTEGRATIONS Number of nested integrations.
    /// @param number_of_integrations The number of integration steps to perform.
    /// @param func The single-variable function `f(x)` to integrate.
    /// @param integration_limit The integration bounds for each step as `[[a, b]; NUM_INTEGRATIONS]`.
    ///
    /// @return Result containing the integrated value, or an error string if the process fails.
    fn get<const NUM_INTEGRATIONS: usize>(
        &self,
        number_of_integrations: usize,
        func: &dyn Fn(f64) -> f64,
        integration_limit: &[[f64; 2]; NUM_INTEGRATIONS],
    ) -> Result<f64, &'static str>;

    /// @brief Convenience wrapper for a single integral of a single-variable function.
    ///
    /// @param func The function `f(x)` to integrate.
    /// @param integration_limit The integration bounds `[a, b]`.
    ///
    /// @return Result containing the computed integral.
    fn get_single(
        &self,
        func: &dyn Fn(f64) -> f64,
        integration_limit: &[f64; 2],
    ) -> Result<f64, &'static str> {
        let new_limits: [[f64; 2]; 1] = [*integration_limit];
        self.get(1, func, &new_limits)
    }

    /// @brief Convenience wrapper for a double integral of a single-variable function.
    ///
    /// @param func The function `f(x)` to integrate.
    /// @param integration_limit The integration limits for both integrations as `[[a, b]; 2]`.
    ///
    /// @return Result containing the computed double integral.
    fn get_double(
        &self,
        func: &dyn Fn(f64) -> f64,
        integration_limit: &[[f64; 2]; 2],
    ) -> Result<f64, &'static str> {
        self.get(2, func, integration_limit)
    }
}

/// @brief Base trait for multi-variable numerical integration.
///
/// Defines the interface for performing partial or multiple integrations of multi-variable functions.
/// Each variable can be integrated selectively.
pub trait IntegratorMultiVariable: Default + Clone + Copy {
    /// @brief Generic n-th partial integration of a multi-variable function.
    ///
    /// @tparam NUM_VARS Number of variables in the function.
    /// @tparam NUM_INTEGRATIONS Number of integrations to perform.
    ///
    /// @param number_of_integrations The number of integration steps.
    /// @param idx_to_integrate The indices of variables with respect to which the function should be integrated.
    /// @param func The multi-variable function `f(x₁, x₂, ..., xₙ)`.
    /// @param integration_limits The integration bounds for each variable as `[[a, b]; NUM_INTEGRATIONS]`.
    /// @param point The evaluation point for the remaining variables.
    ///
    /// @return Result containing the integrated value, or an error string.
    fn get<const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(
        &self,
        number_of_integrations: usize,
        idx_to_integrate: [usize; NUM_INTEGRATIONS],
        func: &dyn Fn(&[f64; NUM_VARS]) -> f64,
        integration_limits: &[[f64; 2]; NUM_INTEGRATIONS],
        point: &[f64; NUM_VARS],
    ) -> Result<f64, &'static str>;

    /// @brief Convenience wrapper for a single partial integral of a multi-variable function.
    ///
    /// @param func The function `f(x₁, x₂, ..., xₙ)` to integrate.
    /// @param idx_to_integrate The index of the variable to integrate with respect to.
    /// @param integration_limits The integration bounds `[a, b]`.
    /// @param point The fixed point for all other variables.
    ///
    /// @return Result containing the computed partial integral.
    fn get_single_partial<const NUM_VARS: usize>(
        &self,
        func: &dyn Fn(&[f64; NUM_VARS]) -> f64,
        idx_to_integrate: usize,
        integration_limits: &[f64; 2],
        point: &[f64; NUM_VARS],
    ) -> Result<f64, &'static str> {
        let new_limits: [[f64; 2]; 1] = [*integration_limits];
        let new_idx: [usize; 1] = [idx_to_integrate];
        self.get(1, new_idx, func, &new_limits, point)
    }

    /// @brief Convenience wrapper for a double partial integral of a multi-variable function.
    ///
    /// @param func The function `f(x₁, x₂, ..., xₙ)` to integrate.
    /// @param idx_to_integrate The indices of the variables to integrate with respect to.
    /// @param integration_limits The integration bounds for both integrations as `[[a, b]; 2]`.
    /// @param point The fixed point for all other variables.
    ///
    /// @return Result containing the computed double partial integral.
    fn get_double_partial<const NUM_VARS: usize>(
        &self,
        func: &dyn Fn(&[f64; NUM_VARS]) -> f64,
        idx_to_integrate: [usize; 2],
        integration_limits: &[[f64; 2]; 2],
        point: &[f64; NUM_VARS],
    ) -> Result<f64, &'static str> {
        self.get(2, idx_to_integrate, func, integration_limits, point)
    }
}

/// @brief Computes the transformed function value `f(x(t)) * dx/dt`
/// according to the correct domain-mapping rule.
///
/// This transformation enables integration over infinite or semi-infinite domains
/// by mapping them to a finite domain `[0, 1]`.
///
/// **Mapping functions:**
/// 1. (-∞, ∞):   x = tan(π(t - ½)),     dx/dt = π / cos²(π(t - ½))
/// 2. (a, ∞):    x = a + t / (1 - t),   dx/dt = 1 / (1 - t)²
/// 3. (-∞, b):   x = b - t / (1 - t),   dx/dt = 1 / (1 - t)²
/// 4. Finite (a, b): x = t,             dx/dt = 1
///
/// @param func The original function `f(x)`.
/// @param original_integration_limit The integration range `[a, b]`, possibly infinite.
/// @param point The current evaluation point `t` in the transformed finite domain `[0, 1]`.
///
/// @return The transformed function value `f(x(t)) * dx/dt`.
pub fn get_domain_change_function_value(
    func: &dyn Fn(f64) -> f64,
    original_integration_limit: &[f64; 2],
    point: f64,
) -> f64 {
    const PI: f64 = core::f64::consts::PI;
    let lower_limit = original_integration_limit[0];
    let upper_limit = original_integration_limit[1];

    use const_poly::function_approximations::*;

    if lower_limit.is_infinite() && upper_limit.is_infinite() {
        // (-∞, ∞)
        let x = tan_approx(PI * (point - 0.5));
        let jac = PI / static_powi(cos_approx(PI * (point - 0.5)), 2);
        func(x) * jac
    } else if lower_limit.is_finite() && upper_limit.is_infinite() {
        // (a, ∞)
        let x = lower_limit + point / (1.0 - point);
        let jac = 1.0 / static_powi(1.0 - point, 2);
        func(x) * jac
    } else if lower_limit.is_infinite() && upper_limit.is_finite() {
        // (-∞, b)
        let x = upper_limit - point / (1.0 - point);
        let jac = 1.0 / static_powi(1.0 - point, 2);
        func(x) * jac
    } else {
        // Finite domain (a, b)
        func(point)
    }
}

/// @brief Returns the transformed integration limits `(t₀, t₁)`
/// that map an infinite or semi-infinite domain to a finite one.
///
/// **Mapping rules:**
/// - Finite (a, b): unchanged → (a, b)
/// - Semi-infinite (a, ∞): t ∈ [0, 1)
/// - Semi-infinite (-∞, b): t ∈ [0, 1)
/// - Infinite (-∞, ∞): t ∈ [0, 1)
///
/// @param original_integration_limit The original integration limits `[a, b]`.
///
/// @return A tuple `(t₀, t₁)` representing the transformed finite domain.
pub fn get_domain_change_limits(original_integration_limit: &[f64; 2]) -> (f64, f64) {
    let a = original_integration_limit[0];
    let b = original_integration_limit[1];
    const EPSILON: f64 = f64::EPSILON;

    if a.is_infinite() || b.is_infinite() {
        // For all infinite forms, map into [0, 1].
        // Do not map exactly to 0 or 1 to avoid undefined behavior at infinities.
        (EPSILON, 1.0 - EPSILON)
    } else {
        // Finite range: leave unchanged.
        (a, b)
    }
}
