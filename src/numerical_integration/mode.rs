/// @brief Options for iterative integration methods. These are good generalist methods that will
/// work for almost any type of equation. 
/// 
/// In most cases, `Trapezoidal` is recommended for the highest accuracy.
/// If unsure, start with `Trapezoidal` and then tweak based on results.
///
/// @note The accuracy of results also depends on the chosen number of iterations.
#[derive(Debug, Clone, Copy)]
pub enum IterativeMethod {
    /// @brief Good generalist method, but trapezoidal outperforms in most cases.
    Booles,

    /// @brief Least accurate. Needs a huge number of iterations to match other methods listed here.
    Simpsons,

    /// @brief Highly accuracate, needs few iterations to converge, best generalist out of all options.
    Trapezoidal,
}


/// @brief Options for gaussian quadrature methods. These are highly specialized methods, such that they
/// are extremely accurate but only for a narrow use-case. Use these methods if you know the equation form ahead of time.
/// 
/// If unsure, start with `GaussLegendre` and then tweak based on results.
///
/// @note The accuracy of results also depends on the chosen number of quadratures/nodes.
#[derive(Debug, Clone, Copy)]
pub enum GaussianQuadratureMethod {
    /// @brief Extremely accurate, but only recommended for polynomial equations. A specialist method with a narrow use case.
    GaussLegendre,

    /// @brief Extremely accurate, but only recommended for integrands of the form ∫exp(-X*X)*f(X), where f(X) is a polynomial equations.
    GaussHermite,

    /// @brief Extremely accurate, but only recommended for integrands of the form ∫exp(-X)*f(X), where f(X) is a polynomial equations.
    GaussLaguerre,
}
