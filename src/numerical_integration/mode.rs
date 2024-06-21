#[derive(Debug, Clone, Copy)]
pub enum IntegrationMethod
{
    Booles, //Highly accurate, but needs more iterations than the trapezoidal method. Good generalist method, but trapezoidal outperforms in most cases
    GaussLegendre, //Highly accurate, but only recommended for polynomial equations whose highest order is known. A specialist method with narrow use case.
    Simpsons, //Least accurate. Needs a huge number of iterations to match other methods listed here
    Trapezoidal //Highly accuracate, do not need to do many iterations, best generalist out of all options
}

#[derive(Debug, Clone, Copy)]
pub enum IterativeMethod
{
    Booles, //Highly accurate, but needs more iterations than the trapezoidal method. Good generalist method, but trapezoidal outperforms in most cases
    Simpsons, //Least accurate. Needs a huge number of iterations to match other methods listed here
    Trapezoidal //Highly accuracate, do not need to do many iterations, best generalist out of all options
}

#[derive(Debug, Clone, Copy)]
pub enum GaussianQuadratureMethod
{
    //Extremely accurate, but only recommended for polynomial equations. A specialist method with a narrow use case.
    GaussLegendre,

    //Extremely accurate, but only recommended for integrands of the form ∫exp(-X*X)*f(X), where f(X) is a polynomial equations.
    GaussHermite,

    //Extremely accurate, but only recommended for integrands of the form ∫exp(-X)*f(X), where f(X) is a polynomial equations.
    GaussLaguerre,
}