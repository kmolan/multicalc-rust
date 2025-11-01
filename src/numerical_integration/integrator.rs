///Base trait for single variable numerical integration
pub trait IntegratorSingleVariable: Default + Clone + Copy {
    ///generic n-th integration of a single variable function
    fn get<const NUM_INTEGRATIONS: usize>(
        &self,
        number_of_integrations: usize,
        func: &dyn Fn(f64) -> f64,
        integration_limit: &[[f64; 2]; NUM_INTEGRATIONS],
    ) -> Result<f64, &'static str>;

    ///convenience wrapper for a single integral of a single variable function
    fn get_single(
        &self,
        func: &dyn Fn(f64) -> f64,
        integration_limit: &[f64; 2],
    ) -> Result<f64, &'static str> {
        let new_limits: [[f64; 2]; 1] = [*integration_limit];

        self.get(1, func, &new_limits)
    }

    ///convenience wrapper for a double integral of a single variable function
    fn get_double(
        &self,
        func: &dyn Fn(f64) -> f64,
        integration_limit: &[[f64; 2]; 2],
    ) -> Result<f64, &'static str> {
        self.get(2, func, integration_limit)
    }
}

///Base trait for multi-variable numerical integration
pub trait IntegratorMultiVariable: Default + Clone + Copy {
    ///generic n-th partial integration of a multi variable function
    fn get<const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(
        &self,
        number_of_integrations: usize,
        idx_to_integrate: [usize; NUM_INTEGRATIONS],
        func: &dyn Fn(&[f64; NUM_VARS]) -> f64,
        integration_limits: &[[f64; 2]; NUM_INTEGRATIONS],
        point: &[f64; NUM_VARS],
    ) -> Result<f64, &'static str>;

    ///convenience wrapper for a single partial integral of a multi variable function
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

    ///convenience wrapper for a double partial integral of a multi variable function
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

/// Evaluates the transformed function value `f(x(t)) * dx/dt`
/// according to the correct domain-mapping rule.
///
/// Mapping functions:
/// 1. (-∞, ∞):  x = tan(π(t - ½)),     dx/dt = π / cos²(π(t - ½))
/// 2. (a, ∞):   x = a + t/(1 - t),     dx/dt = 1 / (1 - t)²
/// 3. (-∞, b):  x = b - t/(1 - t),     dx/dt = 1 / (1 - t)²
/// 4. Finite (a, b):  x = t,           dx/dt = 1
///
/// # Arguments
/// - `func`: the original function f(x)
/// - `original_integration_limit`: `[a, b]` integration range, possibly infinite
/// - `point`: current evaluation point `t` in the transformed finite domain [0, 1]
///
/// # Returns
/// The function value in transformed domain: f(x(t)) * dx/dt
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
        // (lower_limit, ∞)

        let x = lower_limit + point / (1.0 - point);
        let jac = 1.0 / static_powi(1.0 - point, 2);
        func(x) * jac
    } else if lower_limit.is_infinite() && upper_limit.is_finite() {
        // (-∞, upper_limit)

        let x = upper_limit - point / (1.0 - point);
        let jac = 1.0 / static_powi(1.0 - point, 2);
        func(x) * jac
    } else {
        func(point)
    }
}

/// Returns the transformed integration limits `(t0, t1)`
/// that map an infinite or semi-infinite domain to a finite one.
///
/// Mapping rules:
/// - Finite (a, b): unchanged → (a, b)
/// - Semi-infinite (a, ∞): t ∈ [0, 1)
/// - Semi-infinite (-∞, b): t ∈ [0, 1)
/// - Infinite (-∞, ∞): t ∈ [0, 1)
pub fn get_domain_change_limits(original_integration_limit: &[f64; 2]) -> (f64, f64) {
    let a = original_integration_limit[0];
    let b = original_integration_limit[1];

    const EPSILON: f64 = core::f64::EPSILON;

    if a.is_infinite() || b.is_infinite() {
        // For all infinite forms, map into [0, 1]
        (EPSILON, 1.0 - EPSILON) //don't actually map to [0,1] but very close to it, because evaluating at infinity gives undefined behavior
    } else {
        // Finite range: leave unchanged
        (a, b)
    }
}
