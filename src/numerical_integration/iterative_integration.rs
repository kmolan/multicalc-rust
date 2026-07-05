use core::marker::PhantomData;

use crate::numerical_integration::integrator::*;
use crate::numerical_integration::mode::IterativeMethod;
use crate::scalar::Numeric;
use crate::utils::error_codes::CalcError;
use crate::utils::summation::PairwiseSum;

/// Default interval count. A multiple of 12 so Boole (needs a multiple of 4) and
/// Simpson 3/8 (needs a multiple of 3) both align with the composite-rule weights.
pub const DEFAULT_TOTAL_ITERATIONS: u64 = 120;

/// Configuration shared by the single- and multi-variable iterative integrators.
#[derive(Debug, Clone, Copy)]
pub struct IterativeConfig {
    /// Number of intervals the composite rule walks. See [`DEFAULT_TOTAL_ITERATIONS`].
    pub total_iterations: u64,
    /// The composite rule to use: Booles, Simpsons or Trapezoidal.
    pub integration_method: IterativeMethod,
}

impl Default for IterativeConfig {
    /// Boole's rule with [`DEFAULT_TOTAL_ITERATIONS`] intervals; optimal for most generic equations.
    fn default() -> Self {
        IterativeConfig {
            total_iterations: DEFAULT_TOTAL_ITERATIONS,
            integration_method: IterativeMethod::Booles,
        }
    }
}

impl IterativeConfig {
    /// Builds a config with an explicit iteration count and rule.
    pub fn from_parameters(total_iterations: u64, integration_method: IterativeMethod) -> Self {
        IterativeConfig {
            total_iterations,
            integration_method,
        }
    }

    /// Checks that the iteration count is non-zero and every limit is well-defined.
    /// The iteration count is checked before the limits so a zero count reports
    /// [`CalcError::IterationsZero`] regardless of the limits.
    fn check_for_errors<T: Numeric, const NUM_INTEGRATIONS: usize>(
        &self,
        integration_limit: &[[T; 2]; NUM_INTEGRATIONS],
    ) -> Result<(), CalcError> {
        if self.total_iterations == 0 {
            return Err(CalcError::IterationsZero);
        }

        for limit in integration_limit {
            classify(limit)?;
        }

        Ok(())
    }
}

/// Dispatches to the chosen rule, integrating `g` over `[lo, hi]` with `iterations`
/// intervals. The caller decides the domain branch before building `g`, so a finite
/// integral passes `func` straight through with no per-sample transform.
fn integrate_rule<T: Numeric, G: FnMut(T) -> T>(
    method: IterativeMethod,
    iterations: u64,
    lo: T,
    hi: T,
    g: G,
) -> T {
    match method {
        IterativeMethod::Booles => booles(iterations, lo, hi, g),
        IterativeMethod::Simpsons => simpsons(iterations, lo, hi, g),
        IterativeMethod::Trapezoidal => trapezoidal(iterations, lo, hi, g),
    }
}

/// Boole's composite rule over `[lo, hi]`.
fn booles<T: Numeric, G: FnMut(T) -> T>(iterations: u64, lo: T, hi: T, mut g: G) -> T {
    let delta = (hi - lo) / T::from_u64(iterations);
    let mut point = lo;

    let mut ans = PairwiseSum::new();
    ans.add(T::from_f64(7.0) * g(point));
    let mut multiplier = T::from_f64(32.0);

    for iter in 0..iterations - 1 {
        point += delta;
        ans.add(multiplier * g(point));

        if (iter + 2) % 2 != 0 {
            multiplier = T::from_f64(32.0);
        } else if (iter + 2) % 4 == 0 {
            multiplier = T::from_f64(14.0);
        } else {
            multiplier = T::from_f64(12.0);
        }
    }

    ans.add(T::from_f64(7.0) * g(hi));

    T::TWO * delta * ans.total() / T::from_f64(45.0)
}

/// Simpson's 3/8 composite rule over `[lo, hi]`.
fn simpsons<T: Numeric, G: FnMut(T) -> T>(iterations: u64, lo: T, hi: T, mut g: G) -> T {
    let delta = (hi - lo) / T::from_u64(iterations);
    let mut point = lo;

    let mut ans = PairwiseSum::new();
    ans.add(g(point));
    let mut multiplier = T::from_f64(3.0);

    for iter in 0..iterations - 1 {
        point += delta;
        ans.add(multiplier * g(point));

        if (iter + 2) % 3 == 0 {
            multiplier = T::TWO;
        } else {
            multiplier = T::from_f64(3.0);
        }
    }

    ans.add(g(hi));

    T::from_f64(3.0) * delta * ans.total() / T::from_f64(8.0)
}

/// Trapezoidal composite rule over `[lo, hi]`.
fn trapezoidal<T: Numeric, G: FnMut(T) -> T>(iterations: u64, lo: T, hi: T, mut g: G) -> T {
    let delta = (hi - lo) / T::from_u64(iterations);
    let mut point = lo;

    let mut ans = PairwiseSum::new();
    ans.add(g(point));

    for _ in 0..iterations - 1 {
        point += delta;
        ans.add(T::TWO * g(point));
    }

    ans.add(g(hi));

    T::HALF * delta * ans.total()
}

/// Implements the iterative methods for numerical integration for single variable functions
#[derive(Debug, Clone, Copy)]
pub struct IterativeSingle<T = f64> {
    pub config: IterativeConfig,
    _marker: PhantomData<T>,
}

impl<T> Default for IterativeSingle<T> {
    fn default() -> Self {
        IterativeSingle {
            config: IterativeConfig::default(),
            _marker: PhantomData,
        }
    }
}

impl<T> IterativeSingle<T> {
    /// custom constructor. Optimal for fine-tuning for more complex equations
    pub fn from_parameters(total_iterations: u64, integration_method: IterativeMethod) -> Self {
        IterativeSingle {
            config: IterativeConfig::from_parameters(total_iterations, integration_method),
            _marker: PhantomData,
        }
    }
}

impl<T: Numeric> IterativeSingle<T> {
    /// Integrates the `level`-th limit (1-based). Inner folds of a single-variable
    /// integral are constant in the outer variable, so the inner result is computed
    /// once and reused; an infinite outer limit weights it by `dx/dt`. A finite limit
    /// skips the domain transform entirely.
    fn integrate<F: Fn(T) -> T, const NUM_INTEGRATIONS: usize>(
        &self,
        level: usize,
        func: &F,
        integration_limit: &[[T; 2]; NUM_INTEGRATIONS],
    ) -> T {
        let method = self.config.integration_method;
        let iterations = self.config.total_iterations;

        let domain = match classify(&integration_limit[level - 1]) {
            Ok(d) => d,
            Err(_) => return T::NAN, // limits validated in check_for_errors; unreachable
        };

        if level == 1 {
            return match domain {
                Domain::Finite(a, b) => integrate_rule(method, iterations, a, b, func),
                _ => {
                    let (lo, hi) = t_bounds(&domain);
                    integrate_rule(method, iterations, lo, hi, |t| {
                        let (x, jacobian) = map_sample(&domain, t);
                        func(x) * jacobian
                    })
                }
            };
        }

        let inner = self.integrate(level - 1, func, integration_limit);
        match domain {
            Domain::Finite(a, b) => integrate_rule(method, iterations, a, b, |_| inner),
            _ => {
                let (lo, hi) = t_bounds(&domain);
                integrate_rule(method, iterations, lo, hi, |t| {
                    let (_, jacobian) = map_sample(&domain, t);
                    inner * jacobian
                })
            }
        }
    }
}

impl<T: Numeric> IntegratorSingleVariable for IterativeSingle<T> {
    type Scalar = T;

    /// Integrates `func`, once for each limit in `integration_limit` (so the array length
    /// sets the number of integrations).
    ///
    /// A limit may be finite, or use `f64::INFINITY` / `f64::NEG_INFINITY` for an infinite or
    /// semi-infinite range. Infinite ranges are mapped onto a finite interval and are accurate
    /// only for integrands that decay toward the infinite end.
    ///
    /// # Arguments
    /// * `func` - the function to integrate.
    /// * `integration_limit` - the `[lower, upper]` limit for each level of integration.
    ///
    /// # Errors
    /// [`CalcError::IterationsZero`] if the configured iteration count is zero, or
    /// [`CalcError::IntegrationLimitsIllDefined`] if any limit is ill-defined.
    ///
    /// # Examples
    /// ```
    /// use multicalc::numerical_integration::integrator::IntegratorSingleVariable;
    /// use multicalc::numerical_integration::iterative_integration::IterativeSingle;
    ///
    /// let my_func = |x: f64| 2.0 * x;
    /// let integrator = IterativeSingle::default();
    ///
    /// // single integration of 2x over [0, 2] is 4
    /// let val = integrator.get(&my_func, &[[0.0, 2.0]; 1]).unwrap();
    /// assert!(f64::abs(val - 4.0) < 1e-6);
    ///
    /// // double integration over [0, 2] then [-1, 1] is 8
    /// let val = integrator.get(&my_func, &[[0.0, 2.0], [-1.0, 1.0]]).unwrap();
    /// assert!(f64::abs(val - 8.0) < 1e-6);
    ///
    /// // an infinite limit, for a decaying integrand: integral of e^(-x^2) over the real line is sqrt(pi)
    /// let val = integrator.get(&|x| (-x * x).exp(), &[[f64::NEG_INFINITY, f64::INFINITY]]).unwrap();
    /// assert!(f64::abs(val - std::f64::consts::PI.sqrt()) < 1e-6);
    /// ```
    fn get<F: Fn(T) -> T, const NUM_INTEGRATIONS: usize>(
        &self,
        func: &F,
        integration_limit: &[[T; 2]; NUM_INTEGRATIONS],
    ) -> Result<T, CalcError> {
        self.config.check_for_errors(integration_limit)?;
        Ok(self.integrate(NUM_INTEGRATIONS, func, integration_limit))
    }
}

/// Implements the iterative methods for numerical integration for multi variable functions
#[derive(Debug, Clone, Copy)]
pub struct IterativeMulti<T = f64> {
    pub config: IterativeConfig,
    _marker: PhantomData<T>,
}

impl<T> Default for IterativeMulti<T> {
    fn default() -> Self {
        IterativeMulti {
            config: IterativeConfig::default(),
            _marker: PhantomData,
        }
    }
}

impl<T> IterativeMulti<T> {
    /// custom constructor, optimal for fine-tuning the integrator for more complex equations
    pub fn from_parameters(total_iterations: u64, integration_method: IterativeMethod) -> Self {
        IterativeMulti {
            config: IterativeConfig::from_parameters(total_iterations, integration_method),
            _marker: PhantomData,
        }
    }
}

impl<T: Numeric> IterativeMulti<T> {
    /// Integrates the `level`-th limit (1-based) of a partial integral. The sampled
    /// abscissa is written into the integrated variable's slot before recursing, and
    /// an infinite limit weights the whole inner integral by `dx/dt`. A finite limit
    /// skips the domain transform entirely.
    fn integrate<
        F: Fn(&[T; NUM_VARS]) -> T,
        const NUM_VARS: usize,
        const NUM_INTEGRATIONS: usize,
    >(
        &self,
        level: usize,
        idx_to_integrate: [usize; NUM_INTEGRATIONS],
        func: &F,
        integration_limits: &[[T; 2]; NUM_INTEGRATIONS],
        point: &[T; NUM_VARS],
    ) -> T {
        let method = self.config.integration_method;
        let iterations = self.config.total_iterations;

        let domain = match classify(&integration_limits[level - 1]) {
            Ok(d) => d,
            Err(_) => return T::NAN, // limits validated in check_for_errors; unreachable
        };
        let var = idx_to_integrate[level - 1];

        if level == 1 {
            let mut current = *point;
            return match domain {
                Domain::Finite(a, b) => integrate_rule(method, iterations, a, b, |x| {
                    current[var] = x;
                    func(&current)
                }),
                _ => {
                    let (lo, hi) = t_bounds(&domain);
                    integrate_rule(method, iterations, lo, hi, |t| {
                        let (x, jacobian) = map_sample(&domain, t);
                        current[var] = x;
                        func(&current) * jacobian
                    })
                }
            };
        }

        let mut current = *point;
        match domain {
            Domain::Finite(a, b) => integrate_rule(method, iterations, a, b, |x| {
                current[var] = x;
                self.integrate(
                    level - 1,
                    idx_to_integrate,
                    func,
                    integration_limits,
                    &current,
                )
            }),
            _ => {
                let (lo, hi) = t_bounds(&domain);
                integrate_rule(method, iterations, lo, hi, |t| {
                    let (x, jacobian) = map_sample(&domain, t);
                    current[var] = x;
                    let inner = self.integrate(
                        level - 1,
                        idx_to_integrate,
                        func,
                        integration_limits,
                        &current,
                    );
                    inner * jacobian
                })
            }
        }
    }
}

impl<T: Numeric> IntegratorMultiVariable for IterativeMulti<T> {
    type Scalar = T;

    /// Partially integrates `func` over the variables in `idx_to_integrate`, once for each
    /// limit in `integration_limits` (so the array length sets the number of integrations).
    ///
    /// # Arguments
    /// * `idx_to_integrate` - the variable index integrated at each level.
    /// * `func` - the function to integrate.
    /// * `integration_limits` - the `[lower, upper]` limit for each level of integration.
    /// * `point` - the value of every variable. A variable being integrated holds its final
    ///   upper limit; a variable held constant holds that constant.
    ///
    /// # Errors
    /// [`CalcError::IterationsZero`] if the configured iteration count is zero, or
    /// [`CalcError::IntegrationLimitsIllDefined`] if any limit is ill-defined.
    ///
    /// # Examples
    /// ```
    /// use multicalc::numerical_integration::integrator::IntegratorMultiVariable;
    /// use multicalc::numerical_integration::iterative_integration::IterativeMulti;
    ///
    /// // f(x, y, z) = 2x + yz, integrated over x in [0, 1] with (y, z) = (2, 3); result is 7
    /// let func = |args: &[f64; 3]| 2.0 * args[0] + args[1] * args[2];
    /// let point = [1.0, 2.0, 3.0];
    /// let integrator = IterativeMulti::default();
    ///
    /// let val = integrator.get([0; 1], &func, &[[0.0, 1.0]; 1], &point).unwrap();
    /// assert!(f64::abs(val - 7.0) < 1e-6);
    /// ```
    fn get<F: Fn(&[T; NUM_VARS]) -> T, const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(
        &self,
        idx_to_integrate: [usize; NUM_INTEGRATIONS],
        func: &F,
        integration_limits: &[[T; 2]; NUM_INTEGRATIONS],
        point: &[T; NUM_VARS],
    ) -> Result<T, CalcError> {
        self.config.check_for_errors(integration_limits)?;
        Ok(self.integrate(
            NUM_INTEGRATIONS,
            idx_to_integrate,
            func,
            integration_limits,
            point,
        ))
    }
}
