mod control;
mod errors;
mod linalg;
mod polynomial;
mod spatial;

use pyo3::prelude::*;

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn multicalc_py<'py>(module: &Bound<'py, PyModule>) -> PyResult<()> {
    errors::register(module)?;
    linalg::register(module)?;
    control::register(module)?;
    polynomial::register(module)?;
    spatial::register(module)?;
    module.add_function(wrap_pyfunction!(version, module)?)?;
    Ok(())
}
