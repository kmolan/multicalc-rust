use multicalc::numerical_derivative::FiniteDifferenceSingle;
use multicalc::root_finding::{Bisection, Brent, Newton};
use pyo3::prelude::*;

use crate::callables::PythonScalarFn;
use crate::errors;

/// Bisection root of a scalar callable on `[lower, upper]`.
#[pyfunction(name = "bisection")]
fn bind_bisection(callback: Py<PyAny>, lower: f64, upper: f64) -> PyResult<f64> {
    let function = PythonScalarFn::new(callback);
    function.finish(
        Bisection::default()
            .solve(&function, lower, upper)
            .map(|report| report.root)
            .map_err(errors::solve_error),
    )
}

/// Brent root of a scalar callable on `[lower, upper]`.
#[pyfunction(name = "brent")]
fn bind_brent(callback: Py<PyAny>, lower: f64, upper: f64) -> PyResult<f64> {
    let function = PythonScalarFn::new(callback);
    function.finish(
        Brent::default()
            .solve(&function, lower, upper)
            .map(|report| report.root)
            .map_err(errors::solve_error),
    )
}

/// Newton root from `initial_guess`, derivative by finite difference.
#[pyfunction(name = "newton")]
fn bind_newton(callback: Py<PyAny>, initial_guess: f64) -> PyResult<f64> {
    let function = PythonScalarFn::new(callback);
    let solver: Newton<FiniteDifferenceSingle> =
        Newton::from_derivator(FiniteDifferenceSingle::default());
    function.finish(
        solver
            .solve(&function, initial_guess)
            .map(|report| report.root)
            .map_err(errors::solve_error),
    )
}

pub(crate) fn register<'python>(module: &Bound<'python, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(bind_bisection, module)?)?;
    module.add_function(wrap_pyfunction!(bind_brent, module)?)?;
    module.add_function(wrap_pyfunction!(bind_newton, module)?)?;
    Ok(())
}
