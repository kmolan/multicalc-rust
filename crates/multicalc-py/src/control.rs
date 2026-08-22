use multicalc::control::Pid;
use pyo3::prelude::*;

#[pyclass(name = "Pid")]
pub struct PyPid {
    inner: Pid,
}

#[pymethods]
impl PyPid {
    #[new]
    fn new(
        proportional_gain: f64,
        integral_gain: f64,
        derivative_gain: f64,
        timestep: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Pid::new(proportional_gain, integral_gain, derivative_gain, timestep)
                .map_err(|error| pyo3::exceptions::PyValueError::new_err(format!("{error:?}")))?,
        })
    }

    fn update(&mut self, setpoint: f64, measurement: f64) -> f64 {
        self.inner.update(setpoint, measurement)
    }
}

pub(crate) fn register<'py>(module: &Bound<'py, PyModule>) -> PyResult<()> {
    module.add_class::<PyPid>()?;
    Ok(())
}
