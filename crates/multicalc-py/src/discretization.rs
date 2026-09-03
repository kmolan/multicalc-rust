use multicalc::discretization::{q_discrete_white_noise, van_loan, zoh};
use pyo3::prelude::*;

use crate::convert::{matrix_from_rows, matrix_to_rows};
use crate::errors;

/// Zero-order hold of a 2-state, 1-input continuous linear system.
#[pyfunction(name = "zoh")]
fn bind_zoh(
    state_matrix: Vec<Vec<f64>>,
    input_matrix: Vec<Vec<f64>>,
    timestep: f64,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let (discrete_state, discrete_input) = zoh::<2, 1, 3, f64>(
        matrix_from_rows::<2, 2>(state_matrix)?,
        matrix_from_rows::<2, 1>(input_matrix)?,
        timestep,
    )
    .map_err(errors::linalg_error)?;
    Ok((
        matrix_to_rows(discrete_state),
        matrix_to_rows(discrete_input),
    ))
}

/// Van Loan discretisation of a 2-state linear process (`state_matrix`, `process_noise`).
#[pyfunction(name = "van_loan")]
fn bind_van_loan(
    state_matrix: Vec<Vec<f64>>,
    process_noise: Vec<Vec<f64>>,
    timestep: f64,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let (discrete_state, discrete_noise) = van_loan::<2, 4, f64>(
        matrix_from_rows::<2, 2>(state_matrix)?,
        matrix_from_rows::<2, 2>(process_noise)?,
        timestep,
    )
    .map_err(errors::linalg_error)?;
    Ok((
        matrix_to_rows(discrete_state),
        matrix_to_rows(discrete_noise),
    ))
}

/// Discrete white-noise process covariance for a 2-state integrator.
#[pyfunction(name = "q_discrete_white_noise")]
fn bind_q_discrete_white_noise(timestep: f64, variance: f64) -> Vec<Vec<f64>> {
    matrix_to_rows(q_discrete_white_noise::<2, f64>(timestep, variance))
}

pub(crate) fn register<'python>(module: &Bound<'python, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(bind_zoh, module)?)?;
    module.add_function(wrap_pyfunction!(bind_van_loan, module)?)?;
    module.add_function(wrap_pyfunction!(bind_q_discrete_white_noise, module)?)?;
    Ok(())
}
