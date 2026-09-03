use multicalc::ode::{ExponentialMap, Rk4, Rk45};
use pyo3::prelude::*;

use crate::callables::PythonOdeRate2;
use crate::convert::{vector_from_list, vector_to_list};
use crate::errors;
use crate::spatial::PySO3;

/// One RK4 step. `callback(time, state)` must return a length-2 derivative.
#[pyfunction]
fn rk4_step(callback: Py<PyAny>, time: f64, state: Vec<f64>, timestep: f64) -> PyResult<Vec<f64>> {
    let state_vector = vector_from_list(state)?;
    let rate = PythonOdeRate2::new(callback);
    rate.finish(Ok(vector_to_list(Rk4::step(
        &rate.rate(),
        time,
        &state_vector,
        timestep,
    ))))
}

/// Integrate a 2-state ODE from `time_start` to `time_final` with RK45.
#[pyfunction]
fn rk45_solve(
    callback: Py<PyAny>,
    time_start: f64,
    state: Vec<f64>,
    time_final: f64,
) -> PyResult<Vec<f64>> {
    let state_vector = vector_from_list(state)?;
    let rate = PythonOdeRate2::new(callback);
    rate.finish(
        Rk45::default()
            .solve(&rate.rate(), time_start, &state_vector, time_final)
            .map(vector_to_list)
            .map_err(errors::integrate_error),
    )
}

/// Integrate an SO(3) attitude by exponential map over `timestep`.
#[pyfunction]
fn exponential_map_attitude_step(
    orientation: &PySO3,
    angular_rate: Vec<f64>,
    timestep: f64,
) -> PyResult<PySO3> {
    Ok(PySO3 {
        inner: ExponentialMap::attitude_step(
            orientation.inner,
            vector_from_list::<3>(angular_rate)?,
            timestep,
        ),
    })
}

pub(crate) fn register<'python>(module: &Bound<'python, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(rk4_step, module)?)?;
    module.add_function(wrap_pyfunction!(rk45_solve, module)?)?;
    module.add_function(wrap_pyfunction!(exponential_map_attitude_step, module)?)?;
    Ok(())
}
