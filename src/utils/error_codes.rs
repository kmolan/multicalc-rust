pub const NUMBER_OF_DERIVATIVE_STEPS_CANNOT_BE_ZERO: &str = "Cannot specify the step size for differentiation as zero!";
pub const DERIVATE_ORDER_CANNOT_BE_ZERO: &str = "The 'order' argument for computing the derivative cannot be zero!";
pub const INDEX_TO_DERIVATE_ILL_FORMED: &str = "The 'idx_to_derivate' argument length must match exactly with the 'order' argument!";
pub const INDEX_TO_DERIVATIVE_OUT_OF_RANGE: &str = "One of the values in 'idx_to_derivate' argument is greater than the length of 'point' argument!";

pub const INTEGRATION_CANNOT_HAVE_ZERO_ITERATIONS: &str = "Total number of iterations cannot be zero!";
pub const INCORRECT_NUMBER_OF_INTEGRATION_LIMITS: &str = "The 'number_of_integrations' argument value must match the length of 'integration_limit' exactly!";
pub const INTEGRATION_LIMITS_ILL_DEFINED: &str = "Each lower integration limit must be strictly less than its upper limit!";
pub const GAUSSIAN_QUADRATURE_ORDER_OUT_OF_RANGE: &str = "Gaussian Quadrature for this order is not supported!";

pub const VECTOR_OF_FUNCTIONS_CANNOT_BE_EMPTY: &str = "Cannot pass in an empty 'function_matrix' argument!";