use crate::utils::error_codes::CalcError;
use core::f64::consts::PI;

/// Smallest inset from an infinite endpoint, keeping the domain transform away from
/// its singular limit (a larger value trades tail truncation for endpoint stability).
const EPSILON: f64 = f64::EPSILON;

/// Classification of an integration interval, distinguishing finite domains from the
/// three infinite/semi-infinite shapes that need a domain transform.
pub(crate) enum Domain {
    Finite(f64, f64),
    LowerToInf(f64),
    UpperToInf(f64),
    BothInf,
}

/// Validates a single integration limit and classifies its domain.
///
/// Rejects `NaN` limits, equal/reversed finite limits, and infinite limits whose
/// finite end points the wrong way (e.g. `(a, -inf)` or `(+inf, b)`).
pub(crate) fn classify(limit: &[f64; 2]) -> Result<Domain, CalcError> {
    let (a, b) = (limit[0], limit[1]);
    if a.is_nan() || b.is_nan() {
        return Err(CalcError::IntegrationLimitsIllDefined);
    }
    match (a.is_finite(), b.is_finite()) {
        (true, true) if a < b => Ok(Domain::Finite(a, b)),
        (true, false) if b > 0.0 => Ok(Domain::LowerToInf(a)), // (a, +inf); rejects (a, -inf)
        (false, true) if a < 0.0 => Ok(Domain::UpperToInf(b)), // (-inf, b); rejects (+inf, b)
        (false, false) if a < 0.0 && b > 0.0 => Ok(Domain::BothInf), // (-inf, +inf)
        _ => Err(CalcError::IntegrationLimitsIllDefined),
    }
}

/// Returns the `t`-interval the rule walks for a domain. The finite end of a
/// semi-infinite domain sits at `t = 0` and is perfectly regular, so it is included;
/// only an infinite end needs the `EPSILON` inset.
pub(crate) fn t_bounds(d: &Domain) -> (f64, f64) {
    match d {
        Domain::Finite(a, b) => (*a, *b),
        Domain::LowerToInf(_) => (0.0, 1.0 - EPSILON), // finite end t=0, +inf at t=1
        Domain::UpperToInf(_) => (0.0, 1.0 - EPSILON), // finite end t=0, -inf at t=1
        Domain::BothInf => (EPSILON, 1.0 - EPSILON),
    }
}

/// Maps a sample `t` to its position `x` and the Jacobian `dx/dt` for a domain.
/// Finite domains are the identity, so the finite path pays nothing extra.
pub(crate) fn map_sample(d: &Domain, t: f64) -> (f64, f64) {
    match *d {
        Domain::Finite(_, _) => (t, 1.0),
        Domain::LowerToInf(a) => {
            let q = 1.0 - t;
            (a + t / q, 1.0 / (q * q))
        }
        Domain::UpperToInf(b) => {
            let q = 1.0 - t;
            (b - t / q, 1.0 / (q * q))
        }
        Domain::BothInf => {
            let u = PI * (t - 0.5);
            let c = libm::cos(u);
            (libm::tan(u), PI / (c * c))
        }
    }
}

/// Base trait for single variable numerical integration.
pub trait IntegratorSingleVariable {
    /// Generic n-th integration of a single variable function. The number of
    /// integrations equals the length of `integration_limit`.
    ///
    /// # Errors
    /// [`CalcError::IterationsZero`] if the configured iteration count is zero, or
    /// [`CalcError::IntegrationLimitsIllDefined`] if any limit is ill-defined.
    fn get<F: Fn(f64) -> f64, const NUM_INTEGRATIONS: usize>(
        &self,
        func: &F,
        integration_limit: &[[f64; 2]; NUM_INTEGRATIONS],
    ) -> Result<f64, CalcError>;

    /// Convenience wrapper for a single integral of a single variable function.
    fn get_single<F: Fn(f64) -> f64>(
        &self,
        func: &F,
        integration_limit: &[f64; 2],
    ) -> Result<f64, CalcError> {
        self.get(func, &[*integration_limit])
    }

    /// Convenience wrapper for a double integral of a single variable function.
    fn get_double<F: Fn(f64) -> f64>(
        &self,
        func: &F,
        integration_limit: &[[f64; 2]; 2],
    ) -> Result<f64, CalcError> {
        self.get(func, integration_limit)
    }
}

/// Base trait for multi-variable numerical integration.
pub trait IntegratorMultiVariable {
    /// Generic n-th partial integration of a multi variable function. The number of
    /// integrations equals the length of `integration_limits`.
    ///
    /// # Arguments
    /// * `idx_to_integrate` - the variable index integrated at each level.
    /// * `func` - the function to integrate.
    /// * `integration_limits` - the limit for each level of integration.
    /// * `point` - the value of every variable. A variable being integrated holds its final
    ///   upper limit; a variable held constant holds that constant.
    ///
    /// # Errors
    /// [`CalcError::IterationsZero`] if the configured iteration count is zero, or
    /// [`CalcError::IntegrationLimitsIllDefined`] if any limit is ill-defined.
    fn get<F: Fn(&[f64; NUM_VARS]) -> f64, const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(
        &self,
        idx_to_integrate: [usize; NUM_INTEGRATIONS],
        func: &F,
        integration_limits: &[[f64; 2]; NUM_INTEGRATIONS],
        point: &[f64; NUM_VARS],
    ) -> Result<f64, CalcError>;

    /// Convenience wrapper for a single partial integral of a multi variable function.
    fn get_single_partial<F: Fn(&[f64; NUM_VARS]) -> f64, const NUM_VARS: usize>(
        &self,
        func: &F,
        idx_to_integrate: usize,
        integration_limits: &[f64; 2],
        point: &[f64; NUM_VARS],
    ) -> Result<f64, CalcError> {
        self.get([idx_to_integrate], func, &[*integration_limits], point)
    }

    /// Convenience wrapper for a double partial integral of a multi variable function.
    fn get_double_partial<F: Fn(&[f64; NUM_VARS]) -> f64, const NUM_VARS: usize>(
        &self,
        func: &F,
        idx_to_integrate: [usize; 2],
        integration_limits: &[[f64; 2]; 2],
        point: &[f64; NUM_VARS],
    ) -> Result<f64, CalcError> {
        self.get(idx_to_integrate, func, integration_limits, point)
    }
}
