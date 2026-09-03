use multicalc::numerical_derivative::FiniteDifferenceMulti;
use multicalc::optimization::{GaussNewton, LevenbergMarquardt};
use pyo3::prelude::*;

use crate::callables::PythonVectorFn2x2;
use crate::errors;

/// Gauss–Newton for a 2-residual, 2-parameter problem (finite-difference Jacobian).
#[pyclass(name = "GaussNewton2x2")]
pub struct PyGaussNewton2x2 {
    inner: GaussNewton<FiniteDifferenceMulti>,
}

#[pymethods]
impl PyGaussNewton2x2 {
    #[new]
    fn new() -> Self {
        Self {
            inner: GaussNewton::from_derivator(FiniteDifferenceMulti::default()),
        }
    }

    /// Minimize a two-argument `residual` from a 2-vector `initial_guess`.
    ///
    /// Returns `(solution, objective, evaluations)`.
    fn minimize(
        &self,
        residual: Py<PyAny>,
        initial_guess: Vec<f64>,
    ) -> PyResult<(Vec<f64>, f64, usize)> {
        if initial_guess.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "initial_guess must have 2 values",
            ));
        }
        let residual_fn = PythonVectorFn2x2::new(residual);
        let report = residual_fn.finish(
            self.inner
                .minimize(&residual_fn, &[initial_guess[0], initial_guess[1]])
                .map_err(errors::solve_error),
        )?;
        Ok((
            report.solution.to_vec(),
            report.objective_function,
            report.evaluations,
        ))
    }
}

/// Levenberg–Marquardt for a 2-residual, 2-parameter problem (finite-difference Jacobian).
#[pyclass(name = "LevenbergMarquardt2x2")]
pub struct PyLevenbergMarquardt2x2 {
    inner: LevenbergMarquardt<FiniteDifferenceMulti>,
}

#[pymethods]
impl PyLevenbergMarquardt2x2 {
    #[new]
    fn new() -> Self {
        Self {
            inner: LevenbergMarquardt::from_derivator(FiniteDifferenceMulti::default()),
        }
    }

    /// Minimize a two-argument `residual` from a 2-vector `initial_guess`.
    ///
    /// Returns `(solution, objective, evaluations)`.
    fn minimize(
        &self,
        residual: Py<PyAny>,
        initial_guess: Vec<f64>,
    ) -> PyResult<(Vec<f64>, f64, usize)> {
        if initial_guess.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "initial_guess must have 2 values",
            ));
        }
        let residual_fn = PythonVectorFn2x2::new(residual);
        let report = residual_fn.finish(
            self.inner
                .minimize(&residual_fn, &[initial_guess[0], initial_guess[1]])
                .map_err(errors::solve_error),
        )?;
        Ok((
            report.solution.to_vec(),
            report.objective_function,
            report.evaluations,
        ))
    }
}

pub(crate) fn register<'python>(module: &Bound<'python, PyModule>) -> PyResult<()> {
    module.add_class::<PyGaussNewton2x2>()?;
    module.add_class::<PyLevenbergMarquardt2x2>()?;
    Ok(())
}
