/// The composite rule used by the iterative integrators.
#[derive(Debug, Clone, Copy)]
pub enum IterativeMethod {
    /// Highest-order rule here; the most accurate generalist for smooth integrands.
    Booles,
    /// Intermediate order and accuracy.
    Simpsons,
    /// Lowest order; simplest and a solid generalist.
    Trapezoidal,
}

/// The Gaussian quadrature family used by the integrators. Each is most accurate for
/// polynomial-like integrands over its fixed domain.
#[derive(Debug, Clone, Copy)]
pub enum GaussianQuadratureMethod {
    /// Integrates `f(x)` over a finite `[a, b]`.
    GaussLegendre,
    /// Integrates `f(x) * e^{-x^2}` over the whole real line.
    GaussHermite,
    /// Integrates `f(x) * e^{-x}` over `[0, +inf)`.
    GaussLaguerre,
}
