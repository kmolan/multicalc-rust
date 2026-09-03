use multicalc::random::{Pcg32, RandomSource};
use pyo3::prelude::*;

/// PCG32 random source (unit interval and standard normal).
#[pyclass(name = "Pcg32")]
pub struct PyPcg32 {
    inner: Pcg32<f64>,
}

#[pymethods]
impl PyPcg32 {
    #[new]
    fn new(seed: u64) -> Self {
        Self {
            inner: Pcg32::new(seed),
        }
    }

    fn next_unit(&mut self) -> f64 {
        self.inner.next_unit()
    }

    fn standard_normal(&mut self) -> f64 {
        self.inner.standard_normal()
    }
}

pub(crate) fn register<'python>(module: &Bound<'python, PyModule>) -> PyResult<()> {
    module.add_class::<PyPcg32>()?;
    Ok(())
}
