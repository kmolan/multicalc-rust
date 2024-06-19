use crate::numerical_derivative::derivator::Derivator;
use crate::numerical_derivative::fixed_step::FixedStep;
use crate::numerical_derivative::mode as mode;
use crate::utils::error_codes::ErrorCode;
use num_complex::ComplexFloat;

#[cfg(feature = "heap")]
use std::{boxed::Box, vec::Vec};

pub struct Jacobian
{
    derivator: FixedStep
}

impl Default for Jacobian
{
    fn default() -> Self 
    {
        return Jacobian { derivator: FixedStep::default() };    
    }
}

impl Jacobian
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
        return Jacobian { derivator: FixedStep::from_parameters(step_size, method) };
    }

    pub fn from_derivator(derivator: FixedStep) -> Self
    {
        return Jacobian {derivator: derivator}
    }

    /// Returns the jacobian matrix for a given vector of functions
    /// Can handle multivariable functions of any order or complexity
    /// 
    /// The 2-D matrix returned has the structure [[df1/dvar1, df1/dvar2, ... , df1/dvarN],
    ///                                            [         ...              ], 
    ///                                            [dfM/dvar1, dfM/dvar2, ... , dfM/dvarN]]
    /// 
    /// where 'N' is the total number of variables, and 'M' is the total number of functions
    /// 
    /// NOTE: Returns a Result<T, ErrorCode>
    /// Possible ErrorCode are:
    /// VectorOfFunctionsCannotBeEmpty -> if function_matrix argument is an empty array
    /// NumberOfStepsCannotBeZero -> if the derivative step size is zero
    /// 
    /// assume our function vector is (x*y*z ,  x^2 + y^2). First define both the functions
    /// ```
    /// use multicalc::numerical_derivative::jacobian;
    ///    let my_func1 = | args: &[f64; 3] | -> f64 
    ///    { 
    ///        return args[0]*args[1]*args[2]; //x*y*z
    ///    };
    /// 
    ///    let my_func2 = | args: &[f64; 3] | -> f64 
    ///    { 
    ///        return args[0].powf(2.0) + args[1].powf(2.0); //x^2 + y^2
    ///    };
    /// 
    /// //define the function vector
    /// let function_matrix: [&dyn Fn(&[f64; 3]) -> f64; 2] = [&my_func1, &my_func2];
    /// let points = [1.0, 2.0, 3.0]; //the point around which we want the jacobian matrix
    /// 
    /// let result = jacobian::get(&function_matrix, &points).unwrap();
    /// ```
    /// 
    /// the above example can also be extended to complex numbers:
    ///```
    /// use multicalc::numerical_derivative::jacobian;
    ///    let my_func1 = | args: &[num_complex::Complex64; 3] | -> num_complex::Complex64 
    ///    { 
    ///        return args[0]*args[1]*args[2]; //x*y*z
    ///    };
    /// 
    ///    let my_func2 = | args: &[num_complex::Complex64; 3] | -> num_complex::Complex64 
    ///    { 
    ///        return args[0].powf(2.0) + args[1].powf(2.0); //x^2 + y^2
    ///    };
    /// 
    /// //define the function vector
    /// let function_matrix: [&dyn Fn(&[num_complex::Complex64; 3]) -> num_complex::Complex64; 2] = [&my_func1, &my_func2];
    /// 
    /// //the point around which we want the jacobian matrix
    /// let points = [num_complex::c64(1.0, 3.0), num_complex::c64(2.0, 3.5), num_complex::c64(3.0, 0.0)];
    /// 
    /// let result = jacobian::get(&function_matrix, &points).unwrap();
    ///``` 
    /// 
    pub fn get<T: ComplexFloat, const NUM_FUNCS: usize, const NUM_VARS: usize>(&self, function_matrix: &[&dyn Fn(&[T; NUM_VARS]) -> T; NUM_FUNCS], vector_of_points: &[T; NUM_VARS]) -> Result<[[T; NUM_VARS]; NUM_FUNCS], ErrorCode>
    {
        if function_matrix.is_empty()
        {
            return Err(ErrorCode::VectorOfFunctionsCannotBeEmpty);
        }

        let mut result = [[T::zero(); NUM_VARS]; NUM_FUNCS];

        for row_index in 0..NUM_FUNCS
        {
            for col_index in 0..NUM_VARS
            {
                result[row_index][col_index] = self.derivator.get_single_partial(&function_matrix[row_index], col_index, vector_of_points)?;
            }
        }
        
        return Ok(result);
    }

    /// same as [Jacobian::get] but uses heap-allocated vectors to generate the jacobian matrix.
    /// Useful when handling large datasets to avoid a stack overflow. To use, turn on the feature "heap" (false by default)
    /// Can handle multivariable functions of any order or complexity
    /// 
    /// The 2-D matrix returned has the structure [[df1/dvar1, df1/dvar2, ... , df1/dvarN],
    ///                                            [         ...              ], 
    ///                                            [dfM/dvar1, dfM/dvar2, ... , dfM/dvarN]]
    /// 
    /// where 'N' is the total number of variables, and 'M' is the total number of functions
    /// 
    /// NOTE: Returns a Result<T, ErrorCode>
    /// Possible ErrorCode are:
    /// VectorOfFunctionsCannotBeEmpty -> if function_matrix argument is an empty array
    /// NumberOfStepsCannotBeZero -> if the derivative step size is zero
    #[cfg(feature = "heap")]
    pub fn get_on_heap<T: ComplexFloat, const NUM_VARS: usize>(&self, function_matrix: &Vec<Box<dyn Fn(&[T; NUM_VARS]) -> T>>, vector_of_points: &[T; NUM_VARS]) -> Result<Vec<Vec<T>>, ErrorCode>
    {    
        if function_matrix.is_empty()
        {
            return Err(ErrorCode::VectorOfFunctionsCannotBeEmpty);
        }

        let num_funcs = function_matrix.len();

        let mut result: Vec<Vec<T>> = Vec::new();

        for row_index in 0..num_funcs
        {
            let mut cur_row: Vec<T> = Vec::new();
            for col_index in 0..NUM_VARS
            {
                cur_row.push(self.derivator.get_single_partial(&function_matrix[row_index], col_index, vector_of_points)?);
            }

            result.push(cur_row);
        }
        
        return Ok(result);
    }
}