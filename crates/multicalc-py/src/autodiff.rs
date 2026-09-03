use multicalc::scalar::{Dual, HyperDual, Jet};
use pyo3::prelude::*;

/// First-order dual number (`value + deriv ε`).
#[pyclass(name = "Dual")]
pub struct PyDual {
    inner: Dual,
}

#[pymethods]
impl PyDual {
    /// Real part and first derivative.
    #[new]
    fn new(value: f64, deriv: f64) -> Self {
        Self {
            inner: Dual::new(value, deriv),
        }
    }

    /// Independent variable (`deriv = 1`).
    #[staticmethod]
    fn variable(value: f64) -> Self {
        Self {
            inner: Dual::variable(value),
        }
    }

    /// Constant (`deriv = 0`).
    #[staticmethod]
    fn constant(value: f64) -> Self {
        Self {
            inner: Dual::constant(value),
        }
    }

    #[getter]
    fn value(&self) -> f64 {
        self.inner.value
    }

    /// First derivative (ε coefficient).
    #[getter]
    fn deriv(&self) -> f64 {
        self.inner.deriv
    }

    fn __add__(&self, other: &Self) -> Self {
        Self {
            inner: self.inner + other.inner,
        }
    }

    fn __sub__(&self, other: &Self) -> Self {
        Self {
            inner: self.inner - other.inner,
        }
    }

    fn __mul__(&self, other: &Self) -> Self {
        Self {
            inner: self.inner * other.inner,
        }
    }

    fn __truediv__(&self, other: &Self) -> Self {
        Self {
            inner: self.inner / other.inner,
        }
    }
}

/// Hyper-dual number for mixed second derivatives.
#[pyclass(name = "HyperDual")]
pub struct PyHyperDual {
    inner: HyperDual,
}

#[pymethods]
impl PyHyperDual {
    /// Real part and `ε1`, `ε2`, `ε1ε2` coefficients.
    #[new]
    fn new(real: f64, eps1: f64, eps2: f64, eps1eps2: f64) -> Self {
        Self {
            inner: HyperDual::new(real, eps1, eps2, eps1eps2),
        }
    }

    /// Independent variable (`eps1 = 1`).
    #[staticmethod]
    fn variable(real: f64) -> Self {
        Self {
            inner: HyperDual::variable(real),
        }
    }

    #[getter]
    fn real(&self) -> f64 {
        self.inner.real
    }

    #[getter]
    fn eps1(&self) -> f64 {
        self.inner.eps1
    }

    #[getter]
    fn eps2(&self) -> f64 {
        self.inner.eps2
    }

    #[getter]
    fn eps1eps2(&self) -> f64 {
        self.inner.eps1eps2
    }

    fn __mul__(&self, other: &Self) -> Self {
        Self {
            inner: self.inner * other.inner,
        }
    }
}

/// Truncated Taylor jet of length 7.
#[pyclass(name = "Jet7")]
pub struct PyJet7 {
    inner: Jet<f64, 7>,
}

#[pymethods]
impl PyJet7 {
    /// Independent variable.
    #[staticmethod]
    fn variable(value: f64) -> Self {
        Self {
            inner: Jet::<f64, 7>::variable(value),
        }
    }

    fn value(&self) -> f64 {
        self.inner.value()
    }

    /// Derivative of the given order (`0` is the value).
    fn derivative(&self, order: usize) -> f64 {
        self.inner.derivative(order)
    }

    fn __mul__(&self, other: &Self) -> Self {
        Self {
            inner: self.inner * other.inner,
        }
    }
}

pub(crate) fn register<'python>(module: &Bound<'python, PyModule>) -> PyResult<()> {
    module.add_class::<PyDual>()?;
    module.add_class::<PyHyperDual>()?;
    module.add_class::<PyJet7>()?;
    Ok(())
}
