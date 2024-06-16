#[derive(Debug)]
#[derive(Clone)]
#[derive(Copy)]
#[derive(PartialEq)]
pub enum ErrorCode
{
    //Can be returned by every module in this crate
    //For methods utilizing derivatives, it means that the derivative step size is zero
    //For methods utilizing integration, it means that number of iterations is zero
    NumberOfStepsCannotBeZero,

    //Can be returned by single_derivative, double_derivative, and triple_derivate modules.
    //When trying to get a partial derivative, the variable's index has to be provided by the user
    //Returned if the value of index is greater than the number of variables
    IndexToDerivativeOutOfRange,

    //Can be returned by the jacobian module
    //To compute a jacobian, user must pass in a vector of functions
    //Returned if that vector of functions is empty
    VectorOfFunctionsCannotBeEmpty,

    //Can be returned by single_integration, double_integration, line_integral and flux_integral modules
    //Returned if the integration lower limit is not strictly lesser than the integration upper limit
    IntegrationLimitsIllDefined,

    //Can be returned by single_integration and double_integration if using the Gauss Legendre integration method
    //Returned if requested order of integration is < 2 or > 20
    GaussLegendreOrderOutOfRange
}