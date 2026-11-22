use multicalc::numerical_derivative::{
    DerivatorMultiVariable, DerivatorSingleVariable, FiniteDifferenceMulti, FiniteDifferenceSingle,
};
use multicalc::numerical_integration::integral;
use multicalc::scalar::ScalarFn;
use pyo3::prelude::*;

use crate::callables::{PythonScalarFn, PythonScalarFn2};
use crate::errors;

/// First derivative of a scalar callable at `point` (finite difference).
#[pyfunction(name = "derivative")]
fn bind_derivative(callback: Py<PyAny>, point: f64) -> PyResult<f64> {
    let function = PythonScalarFn::new(callback);
    function.finish(
        FiniteDifferenceSingle::default()
            .first_derivative(&function, point)
            .map_err(errors::diff_error),
    )
}

/// Second derivative of a scalar callable at `point` (finite difference).
#[pyfunction(name = "second_derivative")]
fn bind_second_derivative(callback: Py<PyAny>, point: f64) -> PyResult<f64> {
    let function = PythonScalarFn::new(callback);
    function.finish(
        FiniteDifferenceSingle::default()
            .second_derivative(&function, point)
            .map_err(errors::diff_error),
    )
}

/// Partial derivative of a 2-argument `callback` at a length-2 `point`.
#[pyfunction(name = "partial")]
fn bind_partial(callback: Py<PyAny>, variable_index: usize, point: Vec<f64>) -> PyResult<f64> {
    if point.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "partial currently needs a 2-vector",
        ));
    }
    let function = PythonScalarFn2::new(callback);
    let coordinates = [point[0], point[1]];
    function.finish(
        FiniteDifferenceMulti::default()
            .first_partial_derivative(&function, variable_index, &coordinates)
            .map_err(errors::diff_error),
    )
}

/// Definite integral of a scalar callable from `lower` to `upper`.
#[pyfunction(name = "integral")]
fn bind_integral(callback: Py<PyAny>, lower: f64, upper: f64) -> PyResult<f64> {
    let function = PythonScalarFn::new(callback);
    function.finish(
        integral(&|argument| function.eval(argument), [lower, upper])
            .map_err(errors::integrate_error),
    )
}

pub(crate) fn register<'python>(module: &Bound<'python, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(bind_derivative, module)?)?;
    module.add_function(wrap_pyfunction!(bind_second_derivative, module)?)?;
    module.add_function(wrap_pyfunction!(bind_partial, module)?)?;
    module.add_function(wrap_pyfunction!(bind_integral, module)?)?;
    Ok(())
}
