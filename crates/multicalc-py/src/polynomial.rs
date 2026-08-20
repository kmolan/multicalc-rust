use multicalc::Polynomial;
use pyo3::prelude::*;
use pyo3::types::PyList;

#[pyclass(name = "Polynomial3")]
pub struct PyPolynomial3 {
    inner: Polynomial<3>,
}

#[pymethods]
impl PyPolynomial3 {
    #[new]
    fn new(coefficients: Vec<f64>) -> PyResult<Self> {
        let array = polynomial_coefficients::<3>(coefficients)?;
        Ok(Self {
            inner: Polynomial::new(array),
        })
    }

    fn real_roots<'py>(&self, python: Python<'py>) -> PyResult<Py<PyList>> {
        let roots = self
            .inner
            .real_roots()
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(format!("{error:?}")))?;
        Ok(PyList::new(python, roots.as_slice())?.unbind())
    }
}

#[pyclass(name = "Polynomial8")]
pub struct PyPolynomial8 {
    inner: Polynomial<8>,
}

#[pymethods]
impl PyPolynomial8 {
    #[new]
    fn new(coefficients: Vec<f64>) -> PyResult<Self> {
        let array = polynomial_coefficients::<8>(coefficients)?;
        Ok(Self {
            inner: Polynomial::new(array),
        })
    }

    fn evaluate_with_derivatives(&self, x: f64) -> [f64; 3] {
        self.inner.evaluate_with_derivatives::<3>(x)
    }
}

fn polynomial_coefficients<const COUNT: usize>(coefficients: Vec<f64>) -> PyResult<[f64; COUNT]> {
    if coefficients.len() != COUNT {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "expected {COUNT} coefficients, got {}",
            coefficients.len()
        )));
    }
    coefficients
        .try_into()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("invalid coefficient count"))
}

pub(crate) fn register<'py>(module: &Bound<'py, PyModule>) -> PyResult<()> {
    module.add_class::<PyPolynomial3>()?;
    module.add_class::<PyPolynomial8>()?;
    Ok(())
}
