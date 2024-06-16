#[derive(Debug)]
#[derive(Clone)]
#[derive(Copy)]
pub enum IntegrationMethod
{
    Booles, //Highly accurate, but needs more iterations than the trapezoidal method. Good generalist method, but trapezoidal outperforms in most cases
    GaussLegendre, //Highly accurate, but only recommended for polynomial equations whose highest order is known. A specialist method with narrow use case.
    Simpsons, //Least accurate. Needs a huge number of iterations to match other methods listed here
    Trapezoidal //Highly accuracate, do not need to do many iterations, best generalist out of all options
}