use crate::numerical_derivative::derivator::DerivatorMultiVariable;

use num_complex::ComplexFloat;

///computes the hessian matrix for a given function
/// Can handle single and multivariable equations of any complexity or size
pub struct Hessian<D: DerivatorMultiVariable>
{
    derivator: D
}

impl<D: DerivatorMultiVariable> Default for Hessian<D>
{
    ///the default constructor, optimal for most generic cases
    fn default() -> Self 
    {
        return Hessian { derivator: D::default() };    
    }
}

impl<D: DerivatorMultiVariable> Hessian<D>
{
    ///custom constructor, optimal for fine tuning
    /// You can create a custom multivariable derivator from this crate
    /// or supply your own by implementing the base traits yourself 
    pub fn from_derivator(derivator: D) -> Self
    {
        return Hessian {derivator}
    }

    /// Returns the hessian matrix for a given function
    /// Can handle multivariable functions of any order or complexity
    /// 
    /// The 2-D matrix returned has the structure [[d2f/d2var1, d2f/dvar1*dvar2, ... , d2f/dvar1*dvarN], 
    ///                                            [                   ...                            ], 
    ///                                            [d2f/dvar1*dvarN, d2f/dvar2*dvarN, ... , dfM/d2varN]]
    /// where 'N' is the total number of variables
    /// 
    /// NOTE: Returns a Result<T, &'static str>
    /// Possible &'static str are:
    /// NUMBER_OF_DERIVATIVE_STEPS_CANNOT_BE_ZERO -> if the derivative step size is zero
    /// 
    /// assume our function is y*sin(x) + 2*x*e^y. First define the function
    /// ```
    /// use multicalc::numerical_derivative::finite_difference::MultiVariableSolver;
    /// use multicalc::numerical_derivative::hessian::Hessian;
    ///    let my_func = | args: &[f64; 2] | -> f64 
    ///    { 
    ///        return args[1]*args[0].sin() + 2.0*args[0]*args[1].exp();
    ///    };
    /// 
    /// let points = [1.0, 2.0]; //the point around which we want the hessian matrix
    /// let hessian = Hessian::<MultiVariableSolver>::default();
    /// 
    /// let result = hessian.get(&my_func, &points).unwrap();
    /// ```
    /// 
    pub fn get<T: ComplexFloat, const NUM_VARS: usize>(&self, function: &dyn Fn(&[T; NUM_VARS]) -> T, vector_of_points: &[T; NUM_VARS]) -> Result<[[T; NUM_VARS]; NUM_VARS], &'static str>
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