//! Host-only Python bindings for [`multicalc`](multicalc).
//!
//! Import name `multicalc_py`. Workspace crate (`publish = false`), `f64` only. Const-generic
//! sizes are fixed at bind time (`Vector3`, `KalmanFilter2x1`, `ScanGeometry5`), not chosen at
//! runtime. A slice of the crate-root API, not a 1:1 dump.
//!
//! Python callables used as residuals or ODE right-hand sides are evaluated as `f64`. Autodiff
//! numbers (`Dual`, `HyperDual`, `Jet7`) are separate Python types and are not passed through
//! those callbacks.

mod autodiff;
mod calculus;
mod callables;
mod control;
mod convert;
mod discretization;
mod dynamics;
mod errors;
mod estimation;
mod kinematics;
mod linalg;
mod mapping;
mod motion;
mod ode;
mod optimization;
mod plant;
mod polynomial;
mod random;
mod roots;
mod signal;
mod spatial;

use pyo3::prelude::*;

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn multicalc_py<'python>(module: &Bound<'python, PyModule>) -> PyResult<()> {
    errors::register(module)?;
    linalg::register(module)?;
    control::register(module)?;
    polynomial::register(module)?;
    spatial::register(module)?;
    calculus::register(module)?;
    discretization::register(module)?;
    ode::register(module)?;
    roots::register(module)?;
    random::register(module)?;
    signal::register(module)?;
    estimation::register(module)?;
    kinematics::register(module)?;
    motion::register(module)?;
    dynamics::register(module)?;
    plant::register(module)?;
    mapping::register(module)?;
    autodiff::register(module)?;
    optimization::register(module)?;
    module.add_function(wrap_pyfunction!(version, module)?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
