use core::marker::PhantomData;

use crate::gaussian_tables::nodes;
use crate::numerical_integration::integrator::{IntegratorMultiVariable, IntegratorSingleVariable};
use crate::numerical_integration::mode::GaussianQuadratureMethod;
use crate::scalar::Numeric;
use crate::error::IntegrateError;

/// Default quadrature order (number of nodes).
pub const DEFAULT_QUADRATURE_ORDERS: usize = 4;

/// Configuration shared by the single- and multi-variable Gaussian integrators.
#[derive(Debug, Clone, Copy)]
pub struct GaussianConfig {
    /// Number of quadrature nodes (the order). See [`DEFAULT_QUADRATURE_ORDERS`].
    pub order: usize,
    /// The quadrature family: GaussLegendre, GaussHermite or GaussLaguerre.
    pub integration_method: GaussianQuadratureMethod,
}

impl Default for GaussianConfig {
    /// Gauss-Legendre at [`DEFAULT_QUADRATURE_ORDERS`]; optimal for most generic polynomial equations.
    fn default() -> Self {
        GaussianConfig {
            order: DEFAULT_QUADRATURE_ORDERS,
            integration_method: GaussianQuadratureMethod::GaussLegendre,
        }
    }
}

impl GaussianConfig {
    /// Builds a config with an explicit order and quadrature family.
    pub fn from_parameters(order: usize, integration_method: GaussianQuadratureMethod) -> Self {
        GaussianConfig {
            order,
            integration_method,
        }
    }

    /// Validates each integration limit against the method's fixed domain.
    ///
    /// Gauss-Legendre integrates over a finite `[a, b]`; Gauss-Hermite over `(-inf, +inf)`;
    /// Gauss-Laguerre over `[0, +inf)`. The canonical domain is required (not ignored) so a
    /// mismatched limit cannot silently return a wrong result. `NaN` comparisons are false and
    /// are therefore rejected.
    fn check_limits<T: Numeric, const NUM_INTEGRATIONS: usize>(
        &self,
        integration_limit: &[[T; 2]; NUM_INTEGRATIONS],
    ) -> Result<(), IntegrateError> {
        for limit in integration_limit {
            let ok = match self.integration_method {
                GaussianQuadratureMethod::GaussLegendre => {
                    limit[0].is_finite() && limit[1].is_finite() && limit[0] < limit[1]
                }
                GaussianQuadratureMethod::GaussHermite => {
                    limit[0] == T::NEG_INFINITY && limit[1] == T::INFINITY
                }
                GaussianQuadratureMethod::GaussLaguerre => {
                    limit[0] == T::ZERO && limit[1] == T::INFINITY
                }
            };

            if !ok {
                return Err(IntegrateError::LimitsIllDefined);
            }
        }

        Ok(())
    }
}

/// Implements the gaussian quadrature methods for numerical integration for single variable functions
#[derive(Debug, Clone, Copy)]
pub struct GaussianSingle<T = f64> {
    pub config: GaussianConfig,
    _marker: PhantomData<T>,
}

impl<T> Default for GaussianSingle<T> {
    fn default() -> Self {
        GaussianSingle {
            config: GaussianConfig::default(),
            _marker: PhantomData,
        }
    }
}

impl<T> GaussianSingle<T> {
    /// custom constructor, optimal for fine-tuning for specific cases
    pub fn from_parameters(order: usize, integration_method: GaussianQuadratureMethod) -> Self {
        GaussianSingle {
            config: GaussianConfig::from_parameters(order, integration_method),
            _marker: PhantomData,
        }
    }
}

impl<T: Numeric> GaussianSingle<T> {
    /// Gauss-Legendre over a finite `[a, b]`: nodes (defined on `[-1, 1]`) are affine-mapped
    /// by `(b-a)/2 * x + (b+a)/2` and the result scaled by `(b-a)/2`. Inner folds of a
    /// single-variable integral are constant in the outer variable, so the inner result is
    /// computed once and reused. Each tabulated `(weight, abscissa)` is converted to `T` in place.
    fn integrate_legendre<F: Fn(T) -> T, const NUM_INTEGRATIONS: usize>(
        &self,
        level: usize,
        table: &'static [(f64, f64)],
        func: &F,
        integration_limit: &[[T; 2]; NUM_INTEGRATIONS],
    ) -> T {
        let a = integration_limit[level - 1][0];
        let b = integration_limit[level - 1][1];
        let half = (b - a) * T::HALF;
        let mid = (b + a) * T::HALF;

        if level == 1 {
            let mut ans = T::ZERO;
            for &(weight, abscissa) in table {
                ans += T::from_f64(weight) * func(half * T::from_f64(abscissa) + mid);
            }
            return half * ans;
        }

        let inner = self.integrate_legendre(level - 1, table, func, integration_limit);
        let mut ans = T::ZERO;
        for &(weight, _) in table {
            ans += T::from_f64(weight) * inner;
        }
        half * ans
    }

    /// Gauss-Hermite / Gauss-Laguerre over their fixed domain: nodes are used as-is with no
    /// affine map and no exponential factor, since the tabulated weights already carry the
    /// `e^{-x^2}` / `e^{-x}` weighting function.
    fn integrate_canonical<F: Fn(T) -> T>(
        &self,
        level: usize,
        table: &'static [(f64, f64)],
        func: &F,
    ) -> T {
        if level == 1 {
            let mut ans = T::ZERO;
            for &(weight, abscissa) in table {
                ans += T::from_f64(weight) * func(T::from_f64(abscissa));
            }
            return ans;
        }

        let inner = self.integrate_canonical(level - 1, table, func);
        let mut ans = T::ZERO;
        for &(weight, _) in table {
            ans += T::from_f64(weight) * inner;
        }
        ans
    }
}

impl<T: Numeric> IntegratorSingleVariable for GaussianSingle<T> {
    type Scalar = T;

    /// Integrates `func` by Gaussian quadrature, once for each limit in `integration_limit`
    /// (so the array length sets the number of integrations).
    ///
    /// The integrand is passed bare; the tabulated weights carry the implicit weighting
    /// function, and each method has a fixed domain:
    /// - `GaussLegendre`: a finite `[a, b]`, computing `∫_a^b f(x) dx`.
    /// - `GaussHermite`: `[f64::NEG_INFINITY, f64::INFINITY]`, computing `∫ f(x) e^{-x²} dx`.
    /// - `GaussLaguerre`: `[0.0, f64::INFINITY]`, computing `∫_0^∞ f(x) e^{-x} dx`.
    ///
    /// # Arguments
    /// * `func` - the bare integrand `f(x)`.
    /// * `integration_limit` - the limit for each level of integration; each must match the
    ///   method's fixed domain.
    ///
    /// # Errors
    /// [`IntegrateError::QuadratureOrderOutOfRange`] if the configured order is unsupported, or
    /// [`IntegrateError::LimitsIllDefined`] if any limit does not match the method's domain.
    ///
    /// # Examples
    /// ```
    /// use multicalc::numerical_integration::integrator::IntegratorSingleVariable;
    /// use multicalc::numerical_integration::gaussian_integration::GaussianSingle;
    ///
    /// // Gauss-Legendre is exact for polynomials: integral of 4x^3 - 3x^2 over [0, 2] is 8
    /// let my_func = |x: f64| 4.0 * x * x * x - 3.0 * x * x;
    /// let integrator = GaussianSingle::default();
    /// let val = integrator.get(&my_func, &[[0.0, 2.0]; 1]).unwrap();
    /// assert!(f64::abs(val - 8.0) < 1e-7);
    /// ```
    fn get<F: Fn(T) -> T, const NUM_INTEGRATIONS: usize>(
        &self,
        func: &F,
        integration_limit: &[[T; 2]; NUM_INTEGRATIONS],
    ) -> Result<T, IntegrateError> {
        let table = nodes(self.config.integration_method, self.config.order)?;
        self.config.check_limits(integration_limit)?;

        Ok(match self.config.integration_method {
            GaussianQuadratureMethod::GaussLegendre => {
                self.integrate_legendre(NUM_INTEGRATIONS, table, func, integration_limit)
            }
            _ => self.integrate_canonical(NUM_INTEGRATIONS, table, func),
        })
    }
}

/// Implements the gaussian quadrature methods for numerical integration for multi variable functions
#[derive(Debug, Clone, Copy)]
pub struct GaussianMulti<T = f64> {
    pub config: GaussianConfig,
    _marker: PhantomData<T>,
}

impl<T> Default for GaussianMulti<T> {
    fn default() -> Self {
        GaussianMulti {
            config: GaussianConfig::default(),
            _marker: PhantomData,
        }
    }
}

impl<T> GaussianMulti<T> {
    /// custom constructor, optimal for fine-tuning for specific cases
    pub fn from_parameters(order: usize, integration_method: GaussianQuadratureMethod) -> Self {
        GaussianMulti {
            config: GaussianConfig::from_parameters(order, integration_method),
            _marker: PhantomData,
        }
    }
}

impl<T: Numeric> GaussianMulti<T> {
    /// Gauss-Legendre partial integration over a finite `[a, b]`. The affine-mapped node is
    /// written into the integrated variable's slot before recursing; the inner fold depends
    /// on the outer node, so it is recomputed for each one.
    fn integrate_legendre<
        F: Fn(&[T; NUM_VARS]) -> T,
        const NUM_VARS: usize,
        const NUM_INTEGRATIONS: usize,
    >(
        &self,
        level: usize,
        idx_to_integrate: [usize; NUM_INTEGRATIONS],
        table: &'static [(f64, f64)],
        func: &F,
        integration_limits: &[[T; 2]; NUM_INTEGRATIONS],
        point: &[T; NUM_VARS],
    ) -> T {
        let a = integration_limits[level - 1][0];
        let b = integration_limits[level - 1][1];
        let half = (b - a) * T::HALF;
        let mid = (b + a) * T::HALF;
        let var = idx_to_integrate[level - 1];

        let mut current = *point;
        let mut ans = T::ZERO;

        if level == 1 {
            for &(weight, abscissa) in table {
                current[var] = half * T::from_f64(abscissa) + mid;
                ans += T::from_f64(weight) * func(&current);
            }
            return half * ans;
        }

        for &(weight, abscissa) in table {
            current[var] = half * T::from_f64(abscissa) + mid;
            ans += T::from_f64(weight)
                * self.integrate_legendre(
                    level - 1,
                    idx_to_integrate,
                    table,
                    func,
                    integration_limits,
                    &current,
                );
        }
        half * ans
    }

    /// Gauss-Hermite / Gauss-Laguerre partial integration over the fixed domain. The node is
    /// written into the integrated variable's slot as-is (no map, no exponential factor) and
    /// the recursion stays in the same method.
    fn integrate_canonical<
        F: Fn(&[T; NUM_VARS]) -> T,
        const NUM_VARS: usize,
        const NUM_INTEGRATIONS: usize,
    >(
        &self,
        level: usize,
        idx_to_integrate: [usize; NUM_INTEGRATIONS],
        table: &'static [(f64, f64)],
        func: &F,
        point: &[T; NUM_VARS],
    ) -> T {
        let var = idx_to_integrate[level - 1];

        let mut current = *point;
        let mut ans = T::ZERO;

        if level == 1 {
            for &(weight, abscissa) in table {
                current[var] = T::from_f64(abscissa);
                ans += T::from_f64(weight) * func(&current);
            }
            return ans;
        }

        for &(weight, abscissa) in table {
            current[var] = T::from_f64(abscissa);
            ans += T::from_f64(weight)
                * self.integrate_canonical(level - 1, idx_to_integrate, table, func, &current);
        }
        ans
    }
}

impl<T: Numeric> IntegratorMultiVariable for GaussianMulti<T> {
    type Scalar = T;

    /// Partially integrates `func` by Gaussian quadrature over the variables in
    /// `idx_to_integrate`, once for each limit in `integration_limits` (so the array length
    /// sets the number of integrations).
    ///
    /// The integrand is passed bare; the tabulated weights carry the implicit weighting
    /// function (see [`GaussianSingle`] for the per-method domains and integral forms).
    ///
    /// # Arguments
    /// * `idx_to_integrate` - the variable index integrated at each level.
    /// * `func` - the bare integrand.
    /// * `integration_limits` - the limit for each level; each must match the method's domain.
    /// * `point` - the value of every variable. A variable being integrated holds its final
    ///   upper limit; a variable held constant holds that constant.
    ///
    /// # Errors
    /// [`IntegrateError::QuadratureOrderOutOfRange`] if the configured order is unsupported, or
    /// [`IntegrateError::LimitsIllDefined`] if any limit does not match the method's domain.
    ///
    /// # Examples
    /// ```
    /// use multicalc::numerical_integration::integrator::IntegratorMultiVariable;
    /// use multicalc::numerical_integration::gaussian_integration::GaussianMulti;
    ///
    /// // f(x, y, z) = 2x + yz, integrated over x in [0, 1] with (y, z) = (2, 3); result is 7
    /// let my_func = |args: &[f64; 3]| 2.0 * args[0] + args[1] * args[2];
    /// let integrator = GaussianMulti::default();
    /// let point = [1.0, 2.0, 3.0];
    ///
    /// let val = integrator.get([0; 1], &my_func, &[[0.0, 1.0]; 1], &point).unwrap();
    /// assert!(f64::abs(val - 7.0) < 1e-7);
    /// ```
    fn get<F: Fn(&[T; NUM_VARS]) -> T, const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(
        &self,
        idx_to_integrate: [usize; NUM_INTEGRATIONS],
        func: &F,
        integration_limits: &[[T; 2]; NUM_INTEGRATIONS],
        point: &[T; NUM_VARS],
    ) -> Result<T, IntegrateError> {
        let table = nodes(self.config.integration_method, self.config.order)?;
        self.config.check_limits(integration_limits)?;

        Ok(match self.config.integration_method {
            GaussianQuadratureMethod::GaussLegendre => self.integrate_legendre(
                NUM_INTEGRATIONS,
                idx_to_integrate,
                table,
                func,
                integration_limits,
                point,
            ),
            _ => self.integrate_canonical(NUM_INTEGRATIONS, idx_to_integrate, table, func, point),
        })
    }
}
