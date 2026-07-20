use crate::error::IntegrateError;
use crate::scalar::Numeric;

/// Classification of an integration interval, distinguishing finite domains from the
/// three infinite/semi-infinite shapes that need a domain transform.
pub(crate) enum Domain<T: Numeric> {
    Finite(T, T),
    LowerToInf(T),
    UpperToInf(T),
    BothInf,
}

/// Validates a single integration limit and classifies its domain.
///
/// Rejects `NaN` limits, equal/reversed finite limits, and infinite limits whose
/// finite end points the wrong way (e.g. `(a, -inf)` or `(+inf, b)`).
pub(crate) fn classify<T: Numeric>(limit: &[T; 2]) -> Result<Domain<T>, IntegrateError> {
    let (a, b) = (limit[0], limit[1]);
    if a.is_nan() || b.is_nan() {
        return Err(IntegrateError::LimitsIllDefined);
    }
    match (a.is_finite(), b.is_finite()) {
        (true, true) if a < b => Ok(Domain::Finite(a, b)),
        (true, false) if b > T::ZERO => Ok(Domain::LowerToInf(a)), // (a, +inf); rejects (a, -inf)
        (false, true) if a < T::ZERO => Ok(Domain::UpperToInf(b)), // (-inf, b); rejects (+inf, b)
        (false, false) if a < T::ZERO && b > T::ZERO => Ok(Domain::BothInf), // (-inf, +inf)
        _ => Err(IntegrateError::LimitsIllDefined),
    }
}

/// Returns `sample` unchanged, or [`IntegrateError::NonFinite`] if it is NaN or infinite.
/// The quadrature and iterative rules never check their samples otherwise, so a blow-up in
/// the integrand would silently propagate into a garbage result. Mirrors the `Rk45` policy
/// in `ode/rk45.rs`.
pub(crate) fn is_finite<T: Numeric>(sample: T) -> Result<T, IntegrateError> {
    if sample.is_finite() {
        Ok(sample)
    } else {
        Err(IntegrateError::NonFinite)
    }
}

/// Returns the `t`-interval the rule walks for a domain. The finite end of a
/// semi-infinite domain sits at `t = 0` and is perfectly regular, so it is included;
/// only an infinite end needs the `T::EPSILON` inset that keeps the transform away from
/// its singular limit.
pub(crate) fn t_bounds<T: Numeric>(d: &Domain<T>) -> (T, T) {
    match d {
        Domain::Finite(a, b) => (*a, *b),
        Domain::LowerToInf(_) => (T::ZERO, T::ONE - T::EPSILON), // finite end t=0, +inf at t=1
        Domain::UpperToInf(_) => (T::ZERO, T::ONE - T::EPSILON), // finite end t=0, -inf at t=1
        Domain::BothInf => (T::EPSILON, T::ONE - T::EPSILON),
    }
}

/// Maps a sample `t` to its position `x` and the Jacobian `dx/dt` for a domain.
/// Finite domains are the identity, so the finite path pays nothing extra.
pub(crate) fn map_sample<T: Numeric>(d: &Domain<T>, t: T) -> (T, T) {
    match *d {
        Domain::Finite(_, _) => (t, T::ONE),
        Domain::LowerToInf(a) => {
            let q = T::ONE - t;
            (a + t / q, T::ONE / (q * q))
        }
        Domain::UpperToInf(b) => {
            let q = T::ONE - t;
            (b - t / q, T::ONE / (q * q))
        }
        Domain::BothInf => {
            let u = T::PI * (t - T::HALF);
            let c = u.cos();
            (u.tan(), T::PI / (c * c))
        }
    }
}

/// Base trait for single variable numerical integration.
pub trait IntegratorSingleVariable {
    /// The scalar the integral is computed in.
    type Scalar: Numeric;

    /// Generic n-th integration of a single variable function. The number of
    /// integrations equals the length of `integration_limit`.
    ///
    /// # Errors
    /// [`IntegrateError::IterationsZero`] if the configured iteration count is zero, or
    /// [`IntegrateError::LimitsIllDefined`] if any limit is ill-defined.
    fn get<F: Fn(Self::Scalar) -> Self::Scalar, const NUM_INTEGRATIONS: usize>(
        &self,
        func: &F,
        integration_limit: &[[Self::Scalar; 2]; NUM_INTEGRATIONS],
    ) -> Result<Self::Scalar, IntegrateError>;

    /// Convenience wrapper for a single integral of a single variable function.
    fn get_single<F: Fn(Self::Scalar) -> Self::Scalar>(
        &self,
        func: &F,
        integration_limit: &[Self::Scalar; 2],
    ) -> Result<Self::Scalar, IntegrateError> {
        self.get(func, &[*integration_limit])
    }

    /// Convenience wrapper for a double integral of a single variable function.
    fn get_double<F: Fn(Self::Scalar) -> Self::Scalar>(
        &self,
        func: &F,
        integration_limit: &[[Self::Scalar; 2]; 2],
    ) -> Result<Self::Scalar, IntegrateError> {
        self.get(func, integration_limit)
    }
}

/// Base trait for multi-variable numerical integration.
pub trait IntegratorMultiVariable {
    /// The scalar the integral is computed in.
    type Scalar: Numeric;

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
    /// [`IntegrateError::IterationsZero`] if the configured iteration count is zero, or
    /// [`IntegrateError::LimitsIllDefined`] if any limit is ill-defined.
    fn get<
        F: Fn(&[Self::Scalar; NUM_VARS]) -> Self::Scalar,
        const NUM_VARS: usize,
        const NUM_INTEGRATIONS: usize,
    >(
        &self,
        idx_to_integrate: [usize; NUM_INTEGRATIONS],
        func: &F,
        integration_limits: &[[Self::Scalar; 2]; NUM_INTEGRATIONS],
        point: &[Self::Scalar; NUM_VARS],
    ) -> Result<Self::Scalar, IntegrateError>;

    /// Convenience wrapper for a single partial integral of a multi variable function.
    fn get_single_partial<
        F: Fn(&[Self::Scalar; NUM_VARS]) -> Self::Scalar,
        const NUM_VARS: usize,
    >(
        &self,
        func: &F,
        idx_to_integrate: usize,
        integration_limits: &[Self::Scalar; 2],
        point: &[Self::Scalar; NUM_VARS],
    ) -> Result<Self::Scalar, IntegrateError> {
        self.get([idx_to_integrate], func, &[*integration_limits], point)
    }

    /// Convenience wrapper for a double partial integral of a multi variable function.
    fn get_double_partial<
        F: Fn(&[Self::Scalar; NUM_VARS]) -> Self::Scalar,
        const NUM_VARS: usize,
    >(
        &self,
        func: &F,
        idx_to_integrate: [usize; 2],
        integration_limits: &[[Self::Scalar; 2]; 2],
        point: &[Self::Scalar; NUM_VARS],
    ) -> Result<Self::Scalar, IntegrateError> {
        self.get(idx_to_integrate, func, integration_limits, point)
    }
}
