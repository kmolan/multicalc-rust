use crate::numerical_derivative::finite_difference::MultiVariableSolver;
use const_poly::Polynomial;

///computes the hessian matrix for a given function
/// Can handle single and multivariable equations of any complexity or size
pub struct Hessian {
    derivator: MultiVariableSolver,
}

impl Default for Hessian {
    ///the default constructor, optimal for most generic cases
    fn default() -> Self {
        return Hessian {
            derivator: MultiVariableSolver::default(),
        };
    }
}

impl Hessian {

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
    pub const fn get<const NUM_VARS: usize>(
        &self,
        function: &Polynomial<NUM_VARS>,
        vector_of_points: &[f64; NUM_VARS],
    ) -> Result<[[f64; NUM_VARS]; NUM_VARS], &'static str> {
        let mut result = [[0.0; NUM_VARS]; NUM_VARS];

        let mut row_index = 0;

        while row_index < NUM_VARS {
            let mut col_index = 0;
            while col_index < NUM_VARS {

                // compute only upper triangle (symmetric Hessian)
                if col_index >= row_index {
                    let res = self.derivator.get_double_partial(
                        function,
                        &[row_index, col_index],
                        vector_of_points,
                    );

                    match res {
                        Ok(value) => {
                            result[row_index][col_index] = value;
                            result[col_index][row_index] = value;
                        }
                        Err(e) => return Err(e),
                    }
                }

                col_index += 1;
            }
            row_index += 1;
        }

        Ok(result)
    }
}
