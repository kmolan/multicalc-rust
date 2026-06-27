use crate::numerical_integration::integrator::*;
use crate::numerical_integration::mode::IterativeMethod;
use crate::utils::error_codes::CalcError;

/// Default interval count. A multiple of 12 so Boole (needs a multiple of 4) and
/// Simpson 3/8 (needs a multiple of 3) both align with the composite-rule weights.
pub const DEFAULT_TOTAL_ITERATIONS: u64 = 120;

/// Dispatches to the chosen rule, integrating `g` over `[lo, hi]` with `iterations`
/// intervals. The caller decides the domain branch before building `g`, so a finite
/// integral passes `func` straight through with no per-sample transform.
fn integrate_rule<G: FnMut(f64) -> f64>(
    method: IterativeMethod,
    iterations: u64,
    lo: f64,
    hi: f64,
    g: G,
) -> f64 {
    match method {
        IterativeMethod::Booles => booles(iterations, lo, hi, g),
        IterativeMethod::Simpsons => simpsons(iterations, lo, hi, g),
        IterativeMethod::Trapezoidal => trapezoidal(iterations, lo, hi, g),
    }
}

/// Boole's composite rule over `[lo, hi]`.
fn booles<G: FnMut(f64) -> f64>(iterations: u64, lo: f64, hi: f64, mut g: G) -> f64 {
    let delta = (hi - lo) / iterations as f64;
    let mut point = lo;

    let mut ans = 7.0 * g(point);
    let mut multiplier = 32.0;

    for iter in 0..iterations - 1 {
        point += delta;
        ans += multiplier * g(point);

        if (iter + 2) % 2 != 0 {
            multiplier = 32.0;
        } else if (iter + 2) % 4 == 0 {
            multiplier = 14.0;
        } else {
            multiplier = 12.0;
        }
    }

    ans += 7.0 * g(hi);

    2.0 * delta * ans / 45.0
}

/// Simpson's 3/8 composite rule over `[lo, hi]`.
fn simpsons<G: FnMut(f64) -> f64>(iterations: u64, lo: f64, hi: f64, mut g: G) -> f64 {
    let delta = (hi - lo) / iterations as f64;
    let mut point = lo;

    let mut ans = g(point);
    let mut multiplier = 3.0;

    for iter in 0..iterations - 1 {
        point += delta;
        ans += multiplier * g(point);

        if (iter + 2) % 3 == 0 {
            multiplier = 2.0;
        } else {
            multiplier = 3.0;
        }
    }

    ans += g(hi);

    3.0 * delta * ans / 8.0
}

/// Trapezoidal composite rule over `[lo, hi]`.
fn trapezoidal<G: FnMut(f64) -> f64>(iterations: u64, lo: f64, hi: f64, mut g: G) -> f64 {
    let delta = (hi - lo) / iterations as f64;
    let mut point = lo;

    let mut ans = g(point);

    for _ in 0..iterations - 1 {
        point += delta;
        ans += 2.0 * g(point);
    }

    ans += g(hi);

    0.5 * delta * ans
}

/// Implements the iterative methods for numerical integration for single variable functions
#[derive(Clone, Copy)]
pub struct IterativeSingle {
    total_iterations: u64,
    integration_method: IterativeMethod,
}

impl Default for IterativeSingle {
    /// default constructor, optimal for most generic equations
    fn default() -> Self {
        IterativeSingle {
            total_iterations: DEFAULT_TOTAL_ITERATIONS,
            integration_method: IterativeMethod::Booles,
        }
    }
}

impl IterativeSingle {
    /// returns the total number of iterations
    pub fn get_total_iterations(&self) -> u64 {
        self.total_iterations
    }

    /// sets the total number of iterations
    pub fn set_total_iterations(&mut self, total_iterations: u64) {
        self.total_iterations = total_iterations;
    }

    /// returns the chosen integration method
    /// choices are: Booles, Simpsons and Trapezoidal
    pub fn get_integration_method(&self) -> IterativeMethod {
        self.integration_method
    }

    /// sets the integration method
    /// choices are: Booles, Simpsons and Trapezoidal
    pub fn set_integration_method(&mut self, integration_method: IterativeMethod) {
        self.integration_method = integration_method;
    }

    /// custom constructor. Optimal for fine-tuning for more complex equations
    pub fn from_parameters(total_iterations: u64, integration_method: IterativeMethod) -> Self {
        IterativeSingle {
            total_iterations,
            integration_method,
        }
    }

    /// Checks that the iteration count is non-zero and every limit is well-defined.
    fn check_for_errors<const NUM_INTEGRATIONS: usize>(
        &self,
        integration_limit: &[[f64; 2]; NUM_INTEGRATIONS],
    ) -> Result<(), CalcError> {
        if self.total_iterations == 0 {
            return Err(CalcError::IterationsZero);
        }

        for limit in integration_limit {
            classify(limit)?;
        }

        Ok(())
    }

    /// Integrates the `level`-th limit (1-based). Inner folds of a single-variable
    /// integral are constant in the outer variable, so the inner result is computed
    /// once and reused; an infinite outer limit weights it by `dx/dt`. A finite limit
    /// skips the domain transform entirely.
    fn integrate<F: Fn(f64) -> f64, const NUM_INTEGRATIONS: usize>(
        &self,
        level: usize,
        func: &F,
        integration_limit: &[[f64; 2]; NUM_INTEGRATIONS],
    ) -> f64 {
        let domain = match classify(&integration_limit[level - 1]) {
            Ok(d) => d,
            Err(_) => return f64::NAN, // limits validated in check_for_errors; unreachable
        };

        if level == 1 {
            return match domain {
                Domain::Finite(a, b) => {
                    integrate_rule(self.integration_method, self.total_iterations, a, b, func)
                }
                _ => {
                    let (lo, hi) = t_bounds(&domain);
                    integrate_rule(self.integration_method, self.total_iterations, lo, hi, |t| {
                        let (x, jacobian) = map_sample(&domain, t);
                        func(x) * jacobian
                    })
                }
            };
        }

        let inner = self.integrate(level - 1, func, integration_limit);
        match domain {
            Domain::Finite(a, b) => {
                integrate_rule(self.integration_method, self.total_iterations, a, b, |_| inner)
            }
            _ => {
                let (lo, hi) = t_bounds(&domain);
                integrate_rule(self.integration_method, self.total_iterations, lo, hi, |t| {
                    let (_, jacobian) = map_sample(&domain, t);
                    inner * jacobian
                })
            }
        }
    }
}

impl IntegratorSingleVariable for IterativeSingle {
    /// returns the numeric integration value for a single variable function
    /// func: The function to integrate
    /// integration_limit: the integration bound(s) for each round of integration
    ///
    /// The number of integrations equals the length of `integration_limit`.
    ///
    /// NOTE: Returns a Result<f64, CalcError>, where possible Err are:
    /// CalcError::IterationsZero -> if the configured iteration count is zero
    /// CalcError::IntegrationLimitsIllDefined -> if any limit is ill-defined
    ///
    /// assume we want to integrate 2*x . the function would be:
    /// ```
    ///    let my_func = | arg: f64 | -> f64
    ///    {
    ///        return 2.0*arg;
    ///    };
    ///
    /// use multicalc::numerical_integration::integrator::IntegratorSingleVariable;
    /// use multicalc::numerical_integration::iterative_integration::IterativeSingle;
    ///
    /// let integrator = IterativeSingle::default();
    ///
    /// let integration_limit = [[0.0, 2.0]; 1]; //desired integration limit
    /// let val = integrator.get(&my_func, &integration_limit).unwrap(); //single integration
    /// assert!(f64::abs(val - 4.0) < 1e-6);
    ///
    /// let integration_limit = [[0.0, 2.0], [-1.0, 1.0]]; //desired integration limits
    /// let val = integrator.get(&my_func, &integration_limit).unwrap(); //double integration
    /// assert!(f64::abs(val - 8.0) < 1e-6);
    ///
    /// let integration_limit = [[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]]; //desired integration limits
    /// let val = integrator.get(&my_func, &integration_limit).unwrap(); //triple integration
    /// assert!(f64::abs(val - 16.0) < 1e-6);
    ///```
    fn get<F: Fn(f64) -> f64, const NUM_INTEGRATIONS: usize>(
        &self,
        func: &F,
        integration_limit: &[[f64; 2]; NUM_INTEGRATIONS],
    ) -> Result<f64, CalcError> {
        self.check_for_errors(integration_limit)?;
        Ok(self.integrate(NUM_INTEGRATIONS, func, integration_limit))
    }
}

/// Implements the iterative methods for numerical integration for multi variable functions
#[derive(Clone, Copy)]
pub struct IterativeMulti {
    total_iterations: u64,
    integration_method: IterativeMethod,
}

impl Default for IterativeMulti {
    /// default constructor, optimal for most generic equations
    fn default() -> Self {
        IterativeMulti {
            total_iterations: DEFAULT_TOTAL_ITERATIONS,
            integration_method: IterativeMethod::Booles,
        }
    }
}

impl IterativeMulti {
    /// returns the total number of iterations
    pub fn get_total_iterations(&self) -> u64 {
        self.total_iterations
    }

    /// sets the total number of iterations
    pub fn set_total_iterations(&mut self, total_iterations: u64) {
        self.total_iterations = total_iterations;
    }

    /// returns the chosen integration method
    /// choices are: Booles, Simpsons and Trapezoidal
    pub fn get_integration_method(&self) -> IterativeMethod {
        self.integration_method
    }

    /// sets the integration method
    /// choices are: Booles, Simpsons and Trapezoidal
    pub fn set_integration_method(&mut self, integration_method: IterativeMethod) {
        self.integration_method = integration_method;
    }

    /// custom constructor, optimal for fine-tuning the integrator for more complex equations
    pub fn from_parameters(total_iterations: u64, integration_method: IterativeMethod) -> Self {
        IterativeMulti {
            total_iterations,
            integration_method,
        }
    }

    /// Checks that the iteration count is non-zero and every limit is well-defined.
    fn check_for_errors<const NUM_INTEGRATIONS: usize>(
        &self,
        integration_limit: &[[f64; 2]; NUM_INTEGRATIONS],
    ) -> Result<(), CalcError> {
        if self.total_iterations == 0 {
            return Err(CalcError::IterationsZero);
        }

        for limit in integration_limit {
            classify(limit)?;
        }

        Ok(())
    }

    /// Integrates the `level`-th limit (1-based) of a partial integral. The sampled
    /// abscissa is written into the integrated variable's slot before recursing, and
    /// an infinite limit weights the whole inner integral by `dx/dt`. A finite limit
    /// skips the domain transform entirely.
    fn integrate<F: Fn(&[f64; NUM_VARS]) -> f64, const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(
        &self,
        level: usize,
        idx_to_integrate: [usize; NUM_INTEGRATIONS],
        func: &F,
        integration_limits: &[[f64; 2]; NUM_INTEGRATIONS],
        point: &[f64; NUM_VARS],
    ) -> f64 {
        let domain = match classify(&integration_limits[level - 1]) {
            Ok(d) => d,
            Err(_) => return f64::NAN, // limits validated in check_for_errors; unreachable
        };
        let var = idx_to_integrate[level - 1];

        if level == 1 {
            let mut current = *point;
            return match domain {
                Domain::Finite(a, b) => {
                    integrate_rule(self.integration_method, self.total_iterations, a, b, |x| {
                        current[var] = x;
                        func(&current)
                    })
                }
                _ => {
                    let (lo, hi) = t_bounds(&domain);
                    integrate_rule(self.integration_method, self.total_iterations, lo, hi, |t| {
                        let (x, jacobian) = map_sample(&domain, t);
                        current[var] = x;
                        func(&current) * jacobian
                    })
                }
            };
        }

        let mut current = *point;
        match domain {
            Domain::Finite(a, b) => {
                integrate_rule(self.integration_method, self.total_iterations, a, b, |x| {
                    current[var] = x;
                    self.integrate(level - 1, idx_to_integrate, func, integration_limits, &current)
                })
            }
            _ => {
                let (lo, hi) = t_bounds(&domain);
                integrate_rule(self.integration_method, self.total_iterations, lo, hi, |t| {
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

impl IntegratorMultiVariable for IterativeMulti {
    /// returns the numeric integration value for a multi-variable function
    /// idx_to_integrate: the variables' index/indices that needs to be integrated
    /// func: The function to integrate
    /// integration_limit: the integration bound(s) for each round of integration
    /// point: for variables not being integrated, it is their constant value, otherwise it is their final upper limit of integration
    ///
    /// The number of integrations equals the length of `integration_limits`.
    ///
    /// NOTE: Returns a Result<f64, CalcError>, where possible Err are:
    /// CalcError::IterationsZero -> if the configured iteration count is zero
    /// CalcError::IntegrationLimitsIllDefined -> if any limit is ill-defined
    ///
    /// assume we want to integrate 2.0*x + y*z . the function would be:
    /// ```
    /// let func = | args: &[f64; 3] | -> f64
    ///{
    ///    return 2.0*args[0] + args[1]*args[2];
    ///};
    /// let point = [1.0, 2.0, 3.0];
    ///
    /// use multicalc::numerical_integration::integrator::IntegratorMultiVariable;
    /// use multicalc::numerical_integration::iterative_integration::IterativeMulti;
    ///
    /// let integrator = IterativeMulti::default();
    ///
    /// let integration_limit = [[0.0, 1.0]; 1]; //desired integation limit
    /// let val = integrator.get([0; 1], &func, &integration_limit, &point).unwrap();
    /// assert!(f64::abs(val - 7.0) < 1e-6);
    /// ```
    fn get<F: Fn(&[f64; NUM_VARS]) -> f64, const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(
        &self,
        idx_to_integrate: [usize; NUM_INTEGRATIONS],
        func: &F,
        integration_limits: &[[f64; 2]; NUM_INTEGRATIONS],
        point: &[f64; NUM_VARS],
    ) -> Result<f64, CalcError> {
        self.check_for_errors(integration_limits)?;
        Ok(self.integrate(NUM_INTEGRATIONS, idx_to_integrate, func, integration_limits, point))
    }
}
