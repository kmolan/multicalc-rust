use crate::numerical_derivative::derivator::DerivatorMultiVariable;
use crate::utils::error_codes::*;

#[cfg(feature = "heap")]
use std::{boxed::Box, vec::Vec};

/// @brief Computes the Jacobian matrix for a given vector of functions. It can handle
/// multivariable functions of any order or complexity.
pub struct Jacobian<D: DerivatorMultiVariable> {
    derivator: D,
}

impl<D: DerivatorMultiVariable> Default for Jacobian<D> {
    /// @brief Default constructor.
    fn default() -> Self {
        Jacobian {
            derivator: D::default(),
        }
    }
}

impl<D: DerivatorMultiVariable> Jacobian<D> {
    /// @brief Custom constructor, optimal for fine-tuning.
    ///
    /// You can create a custom multivariable derivator from this crate
    /// or supply your own by implementing the base traits yourself.
    ///
    /// @param derivator A custom instance implementing `DerivatorMultiVariable`.
    ///
    /// @return A new `Jacobian` instance using the provided derivator.
    pub fn from_derivator(derivator: D) -> Self {
        Jacobian { derivator }
    }

    /// @brief Returns the Jacobian matrix for a given vector of functions. It can
    /// handle multivariable functions of any order or complexity.
    ///
    /// The 2-D matrix returned has the structure:
    ///
    /// [[∂f₁/∂x₁, ∂f₁/∂x₂, ... , ∂f₁/∂xₙ],
    ///  [...                       ...],
    ///  [∂fₘ/∂x₁, ∂fₘ/∂x₂, ... , ∂fₘ/∂xₙ]]
    ///
    /// where `N` is the total number of variables, and `M` is the total number of functions.
    ///
    /// @tparam NUM_FUNCS Number of functions in the function vector.
    /// @tparam NUM_VARS Number of variables per function.
    ///
    /// @param function_matrix The vector of function references (one per output dimension).
    /// @param vector_of_points The evaluation point `[x₁, x₂, ..., xₙ]`.
    ///
    /// @return Result containing the computed Jacobian matrix `[[f64; NUM_VARS]; NUM_FUNCS]`
    /// or an error string if computation fails.
    ///
    /// @note
    /// Possible error codes include:
    /// - `VECTOR_OF_FUNCTIONS_CANNOT_BE_EMPTY`: If `function_matrix` is empty.
    /// - `NUMBER_OF_DERIVATIVE_STEPS_CANNOT_BE_ZERO`: If the derivative step size is zero.
    ///
    /// @example
    /// ```
    /// use multicalc::numerical_derivative::finite_difference::MultiVariableSolver;
    /// use multicalc::numerical_derivative::jacobian::Jacobian;
    ///
    /// let my_func1 = |args: &[f64; 3]| -> f64 {
    ///     args[0] * args[1] * args[2] // x*y*z
    /// };
    ///
    /// let my_func2 = |args: &[f64; 3]| -> f64 {
    ///     args[0].powf(2.0) + args[1].powf(2.0) // x² + y²
    /// };
    ///
    /// // Define the function vector.
    /// let function_matrix: [&dyn Fn(&[f64; 3]) -> f64; 2] = [&my_func1, &my_func2];
    /// let points = [1.0, 2.0, 3.0]; // The point around which we want the Jacobian matrix.
    ///
    /// let jacobian = Jacobian::<MultiVariableSolver>::default();
    /// let result = jacobian.get(&function_matrix, &points).unwrap();
    /// ```
    pub fn get<const NUM_FUNCS: usize, const NUM_VARS: usize>(
        &self,
        function_matrix: &[&dyn Fn(&[f64; NUM_VARS]) -> f64; NUM_FUNCS],
        vector_of_points: &[f64; NUM_VARS],
    ) -> Result<[[f64; NUM_VARS]; NUM_FUNCS], &'static str> {
        if function_matrix.is_empty() {
            return Err(VECTOR_OF_FUNCTIONS_CANNOT_BE_EMPTY);
        }

        let mut result = [[0.0; NUM_VARS]; NUM_FUNCS];

        for row_index in 0..NUM_FUNCS {
            for col_index in 0..NUM_VARS {
                result[row_index][col_index] = self.derivator.get_single_partial(
                    &function_matrix[row_index],
                    col_index,
                    vector_of_points,
                )?;
            }
        }

        Ok(result)
    }

    /// @brief Heap-allocated version of [Jacobian::get].
    ///
    /// Uses heap-allocated vectors to compute the Jacobian matrix,
    /// reducing stack usage for large systems. Enable the `"heap"` feature
    /// (disabled by default) to use this version.
    ///
    /// Can handle multivariable functions of any order or complexity.
    ///
    /// The 2-D matrix returned has the structure:
    ///
    /// [[∂f₁/∂x₁, ∂f₁/∂x₂, ... , ∂f₁/∂xₙ],
    ///  [...                       ...],
    ///  [∂fₘ/∂x₁, ∂fₘ/∂x₂, ... , ∂fₘ/∂xₙ]]
    ///
    /// where `N` is the total number of variables, and `M` is the total number of functions.
    ///
    /// @tparam NUM_VARS Number of variables per function.
    ///
    /// @param function_matrix Heap-allocated vector of boxed function closures.
    /// @param vector_of_points The point `[x₁, x₂, ..., xₙ]` at which to evaluate.
    ///
    /// @return Result containing the computed Jacobian matrix as `Vec<Vec<f64>>`,
    /// or an error string if computation fails.
    ///
    /// @note
    /// Possible error codes include:
    /// - `VECTOR_OF_FUNCTIONS_CANNOT_BE_EMPTY`: If `function_matrix` is empty.
    /// - `NUMBER_OF_DERIVATIVE_STEPS_CANNOT_BE_ZERO`: If the derivative step size is zero.
    ///
    /// @example
    /// ```
    /// // Requires the "heap" feature enabled.
    /// use multicalc::numerical_derivative::finite_difference::MultiVariableSolver;
    /// use multicalc::numerical_derivative::jacobian::Jacobian;
    /// use std::{boxed::Box, vec::Vec};
    ///
    /// let f1 = Box::new(|args: &[f64; 3]| -> f64 { args[0] * args[1] * args[2] });
    /// let f2 = Box::new(|args: &[f64; 3]| -> f64 { args[0].powf(2.0) + args[1].powf(2.0) });
    ///
    /// let funcs: Vec<Box<dyn Fn(&[f64; 3]) -> f64>> = vec![f1, f2];
    /// let points = [1.0, 2.0, 3.0];
    ///
    /// let jacobian = Jacobian::<MultiVariableSolver>::default();
    /// let result = jacobian.get_on_heap(&funcs, &points).unwrap();
    /// ```
    #[cfg(feature = "heap")]
    pub fn get_on_heap<const NUM_VARS: usize>(
        &self,
        function_matrix: &Vec<Box<dyn Fn(&[f64; NUM_VARS]) -> f64>>,
        vector_of_points: &[f64; NUM_VARS],
    ) -> Result<Vec<Vec<f64>>, &'static str> {
        if function_matrix.is_empty() {
            return Err(VECTOR_OF_FUNCTIONS_CANNOT_BE_EMPTY);
        }

        let num_funcs = function_matrix.len();
        let mut result: Vec<Vec<f64>> = Vec::new();

        for row_index in 0..num_funcs {
            let mut cur_row: Vec<f64> = Vec::new();
            for col_index in 0..NUM_VARS {
                cur_row.push(self.derivator.get_single_partial(
                    &function_matrix[row_index],
                    col_index,
                    vector_of_points,
                )?);
            }
            result.push(cur_row);
        }

        Ok(result)
    }
}
