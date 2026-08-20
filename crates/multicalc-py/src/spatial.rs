use multicalc::linear_algebra::Vector;
use multicalc::spatial::SO3;
use pyo3::prelude::*;

#[pyclass(name = "SO3")]
pub struct PySO3 {
    inner: SO3,
}

#[pymethods]
impl PySO3 {
    #[staticmethod]
    fn exp(phi: [f64; 3]) -> Self {
        Self {
            inner: SO3::exp(Vector::new(phi)),
        }
    }

    fn act(&self, point: [f64; 3]) -> [f64; 3] {
        self.inner.act(Vector::new(point)).into_array()
    }
}

pub(crate) fn register<'py>(module: &Bound<'py, PyModule>) -> PyResult<()> {
    module.add_class::<PySO3>()?;
    Ok(())
}
