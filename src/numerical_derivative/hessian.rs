use crate::numerical_derivative::derivator::Derivator;
use crate::numerical_derivative::fixed_step::FixedStep;
use crate::numerical_derivative::mode as mode;
use crate::utils::error_codes::ErrorCode;
use num_complex::ComplexFloat;

pub struct Hessian
{
    derivator: FixedStep
}

impl Default for Hessian
{
    fn default() -> Self 
    {
        return Hessian { derivator: FixedStep::default() };    
    }
}

impl Hessian
{
    pub fn set_step_size(&mut self, step_size: f64)
    {
        self.derivator.set_step_size(step_size);
    }

    pub fn get_step_size(&self) -> f64
    {
        return self.derivator.get_step_size();
    }

    pub fn set_derivative_method(&mut self, method: mode::FixedStepMode)
    {
        self.derivator.set_method(method);
    }

    pub fn get_derivative_method(&self) -> mode::FixedStepMode
    {
        return self.derivator.get_method();
    }

    pub fn from_parameters(step_size: f64, method: mode::FixedStepMode) -> Self
    {
        return Hessian { derivator: FixedStep::from_parameters(step_size, method) };
    }

    pub fn from_derivator(derivator: FixedStep) -> Self
    {
        return Hessian {derivator: derivator}
    }

    /// Returns the hessian matrix for a given function
    /// Can handle multivariable functions of any order or complexity
    /// 
    /// The 2-D matrix returned has the structure [[d2f/d2var1, d2f/dvar1*dvar2, ... , d2f/dvar1*dvarN], 
    ///                                            [                   ...                            ], 
    ///                                            [d2f/dvar1*dvarN, d2f/dvar2*dvarN, ... , dfM/d2varN]]
    /// where 'N' is the total number of variables
    /// 
    /// NOTE: Returns a Result<T, ErrorCode>
    /// Possible ErrorCode are:
    /// NumberOfStepsCannotBeZero -> if the derivative step size is zero
    /// 
    /// assume our function is y*sin(x) + 2*x*e^y. First define the function
    /// ```
    /// use multicalc::numerical_derivative::hessian;
    ///    let my_func = | args: &[f64; 2] | -> f64 
    ///    { 
    ///        return args[1]*args[0].sin() + 2.0*args[0]*args[1].exp();
    ///    };
    /// 
    /// let points = [1.0, 2.0]; //the point around which we want the hessian matrix
    /// 
    /// let result = hessian::get(&my_func, &points);
    /// ```
    /// 
    /// the above example can also be extended to complex numbers:
    ///```
    /// use multicalc::numerical_derivative::hessian;
    ///    let my_func = | args: &[num_complex::Complex64; 2] | -> num_complex::Complex64 
    ///    { 
    ///        return args[1]*args[0].sin() + 2.0*args[0]*args[1].exp();
    ///    };
    /// 
    /// //the point around which we want the hessian matrix
    /// let points = [num_complex::c64(1.0, 2.5), num_complex::c64(2.0, 5.0)];
    /// 
    /// let result = hessian::get(&my_func, &points);
    ///``` 
    /// 
    pub fn get<T: ComplexFloat, const NUM_VARS: usize>(&self, function: &dyn Fn(&[T; NUM_VARS]) -> T, vector_of_points: &[T; NUM_VARS]) -> Result<[[T; NUM_VARS]; NUM_VARS], ErrorCode>
    {
        let mut result = [[T::from(f64::NAN).unwrap(); NUM_VARS]; NUM_VARS];

        for row_index in 0..NUM_VARS
        {
            for col_index in 0..NUM_VARS
            {
                if result[row_index][col_index].is_nan()
                {
                    result[row_index][col_index] = self.derivator.get_double_partial(function, &[row_index, col_index], vector_of_points)?;

                    result[col_index][row_index] = result[row_index][col_index]; //exploit the fact that a hessian is a symmetric matrix
                }
            }
        }

        return Ok(result);
    }

}