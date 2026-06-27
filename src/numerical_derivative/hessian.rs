use crate::numerical_derivative::derivator::DerivatorMultiVariable;
use crate::utils::error_codes::CalcError;

///computes the hessian matrix for a given function
/// Can handle single and multivariable equations of any complexity or size
pub struct Hessian<D: DerivatorMultiVariable> {
    derivator: D,
}

impl<D: DerivatorMultiVariable + Default> Default for Hessian<D> {
    ///the default constructor, optimal for most generic cases
    fn default() -> Self {
        Hessian {
            derivator: D::default(),
        }
    }
}

impl<D: DerivatorMultiVariable> Hessian<D> {
    ///custom constructor, optimal for fine tuning
    /// You can create a custom multivariable derivator from this crate
    /// or supply your own by implementing the base traits yourself
    pub fn from_derivator(derivator: D) -> Self {
        Hessian { derivator }
    }

    /// Returns the hessian matrix for a given function
    /// Can handle multivariable functions of any order or complexity
    ///
    /// The 2-D matrix returned has the structure [[d2f/d2var1, d2f/dvar1*dvar2, ... , d2f/dvar1*dvarN],
    ///                                            [                   ...                            ],
    ///                                            [d2f/dvar1*dvarN, d2f/dvar2*dvarN, ... , dfM/d2varN]]
    /// where 'N' is the total number of variables
    ///
    /// NOTE: Returns a Result<_, CalcError>
    /// Possible CalcError are:
    /// CalcError::StepSizeZero -> if the derivative step size is zero
    ///
    /// assume our function is y*sin(x) + 2*x*e^y. First define the function
    /// ```
    /// use multicalc::numerical_derivative::finite_difference::FiniteDifferenceMulti;
    /// use multicalc::numerical_derivative::hessian::Hessian;
    ///    let my_func = | args: &[f64; 2] | -> f64
    ///    {
    ///        return args[1]*args[0].sin() + 2.0*args[0]*args[1].exp();
    ///    };
    ///
    /// let points = [1.0, 2.0]; //the point around which we want the hessian matrix
    /// let hessian = Hessian::<FiniteDifferenceMulti>::default();
    ///
    /// let result = hessian.get(&my_func, &points).unwrap();
    /// ```
    ///
    pub fn get<F: Fn(&[f64; NUM_VARS]) -> f64, const NUM_VARS: usize>(
        &self,
        function: &F,
        vector_of_points: &[f64; NUM_VARS],
    ) -> Result<[[f64; NUM_VARS]; NUM_VARS], CalcError> {
        let mut result = [[f64::NAN; NUM_VARS]; NUM_VARS];

        for row_index in 0..NUM_VARS {
            for col_index in 0..NUM_VARS {
                if result[row_index][col_index].is_nan() {
                    result[row_index][col_index] = self.derivator.get_double_partial(
                        function,
                        &[row_index, col_index],
                        vector_of_points,
                    )?;

                    result[col_index][row_index] = result[row_index][col_index];
                    //exploit the fact that a hessian is a symmetric matrix
                }
            }
        }

        Ok(result)
    }
}
