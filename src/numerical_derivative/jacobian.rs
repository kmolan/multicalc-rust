use crate::numerical_derivative::finite_difference::MultiVariableSolver;
use crate::utils::error_codes::*;
use const_poly::Polynomial;

#[cfg(feature = "heap")]
use std::{boxed::Box, vec::Vec};

pub struct Jacobian {
    derivator: MultiVariableSolver,
}

impl Default for Jacobian {
    fn default() -> Self {
        Self {
            derivator: MultiVariableSolver::default(),
        }
    }
}

impl Jacobian {
    /// Returns the jacobian matrix for a given vector of functions
    /// Can handle multivariable functions of any order or complexity
    ///
    /// The 2-D matrix returned has the structure [[df1/dvar1, df1/dvar2, ... , df1/dvarN],
    ///                                            [         ...              ],
    ///                                            [dfM/dvar1, dfM/dvar2, ... , dfM/dvarN]]
    ///
    /// where 'N' is the total number of variables, and 'M' is the total number of functions
    ///
    /// NOTE: Returns a Result<T, &'static str>
    /// Possible &'static str are:
    /// VECTOR_OF_FUNCTIONS_CANNOT_BE_EMPTY -> if function_matrix argument is an empty array
    /// NUMBER_OF_DERIVATIVE_STEPS_CANNOT_BE_ZERO -> if the derivative step size is zero
    ///
    /// assume our function vector is (x*y*z ,  x^2 + y^2). First define both the functions
    /// ```
    /// use multicalc::numerical_derivative::finite_difference::MultiVariableSolver;
    /// use multicalc::numerical_derivative::jacobian::Jacobian;
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
    /// let jacobian = Jacobian::<MultiVariableSolver>::default();
    /// let result = jacobian.get(&function_matrix, &points).unwrap();
    /// ```
    ///
    pub const fn get<const NUM_FUNCS: usize, const NUM_VARS: usize>(
        &self,
        function_matrix: &[&Polynomial<NUM_VARS>; NUM_FUNCS],
        vector_of_points: &[f64; NUM_VARS],
    ) -> Result<[[f64; NUM_VARS]; NUM_FUNCS], &'static str> {
        if NUM_FUNCS == 0 {
            return Err(VECTOR_OF_FUNCTIONS_CANNOT_BE_EMPTY);
        }

        let mut result = [[0.0; NUM_VARS]; NUM_FUNCS];

        let mut row_index = 0;
        while row_index < NUM_FUNCS {
            let mut col_index = 0;
            while col_index < NUM_VARS {
                let val = self.derivator.get_single_partial(
                    function_matrix[row_index],
                    col_index,
                    vector_of_points,
                );

                match val {
                    Ok(v) => result[row_index][col_index] = v,
                    Err(e) => return Err(e),
                }

                col_index += 1;
            }

            row_index += 1;
        }

        Ok(result)
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
    /// NOTE: Returns a Result<T, &'static str>
    /// Possible &'static str are:
    /// VectorOfFunctionsCannotBeEmpty -> if function_matrix argument is an empty array
    /// NumberOfStepsCannotBeZero -> if the derivative step size is zero
    #[cfg(feature = "heap")]
    pub fn get_on_heap<T: ComplexFloat, const NUM_VARS: usize>(
        &self,
        function_matrix: &Vec<Box<dyn Fn(&[T; NUM_VARS]) -> T>>,
        vector_of_points: &[T; NUM_VARS],
    ) -> Result<Vec<Vec<T>>, &'static str> {
        if function_matrix.is_empty() {
            return Err(VECTOR_OF_FUNCTIONS_CANNOT_BE_EMPTY);
        }

        let num_funcs = function_matrix.len();

        let mut result: Vec<Vec<T>> = Vec::new();

        for row_index in 0..num_funcs {
            let mut cur_row: Vec<T> = Vec::new();
            for col_index in 0..NUM_VARS {
                cur_row.push(self.derivator.get_single_partial(
                    &function_matrix[row_index],
                    col_index,
                    vector_of_points,
                )?);
            }

            result.push(cur_row);
        }

        return Ok(result);
    }
}
