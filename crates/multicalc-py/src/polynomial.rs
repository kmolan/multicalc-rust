use multicalc::{MultivariatePolynomial, MultivariateTerm, PiecewisePolynomial, Polynomial};
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::errors;

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

macro_rules! bind_polynomial {
    ($py_name:ident, $count:expr) => {
        /// Univariate polynomial. Coefficient count is in the type name (`Polynomial2`, …).
        #[pyclass]
        pub struct $py_name {
            inner: Polynomial<$count>,
        }

        #[pymethods]
        impl $py_name {
            /// Coefficients, lowest degree first, matching the type length.
            #[new]
            fn new(coefficients: Vec<f64>) -> PyResult<Self> {
                Ok(Self {
                    inner: Polynomial::new(polynomial_coefficients::<$count>(coefficients)?),
                })
            }

            fn evaluate(&self, argument: f64) -> f64 {
                self.inner.evaluate(argument)
            }

            /// Value and first two derivatives at `argument`.
            fn evaluate_with_derivatives(&self, argument: f64) -> [f64; 3] {
                self.inner.evaluate_with_derivatives::<3>(argument)
            }

            fn __len__(&self) -> usize {
                $count
            }

            fn __repr__(&self) -> String {
                format!(
                    "{}({:?})",
                    stringify!($py_name),
                    self.inner.coefficients().as_slice()
                )
            }
        }
    };
}

macro_rules! bind_polynomial_with_roots {
    ($py_name:ident, $count:expr) => {
        /// Univariate polynomial with real-root finding. Coefficient count is in the type name.
        #[pyclass]
        pub struct $py_name {
            inner: Polynomial<$count>,
        }

        #[pymethods]
        impl $py_name {
            /// Coefficients, lowest degree first, matching the type length.
            #[new]
            fn new(coefficients: Vec<f64>) -> PyResult<Self> {
                Ok(Self {
                    inner: Polynomial::new(polynomial_coefficients::<$count>(coefficients)?),
                })
            }

            fn evaluate(&self, argument: f64) -> f64 {
                self.inner.evaluate(argument)
            }

            /// Value and first two derivatives at `argument`.
            fn evaluate_with_derivatives(&self, argument: f64) -> [f64; 3] {
                self.inner.evaluate_with_derivatives::<3>(argument)
            }

            /// Real roots as a Python list.
            fn real_roots<'python>(&self, python: Python<'python>) -> PyResult<Py<PyList>> {
                let roots = self.inner.real_roots().map_err(errors::polynomial_error)?;
                Ok(PyList::new(python, roots.as_slice())?.unbind())
            }

            fn __len__(&self) -> usize {
                $count
            }

            fn __repr__(&self) -> String {
                format!(
                    "{}({:?})",
                    stringify!($py_name),
                    self.inner.coefficients().as_slice()
                )
            }
        }
    };
}

bind_polynomial_with_roots!(Polynomial2, 2);
bind_polynomial_with_roots!(Polynomial3, 3);
bind_polynomial_with_roots!(Polynomial4, 4);
bind_polynomial_with_roots!(Polynomial5, 5);
bind_polynomial!(Polynomial6, 6);
bind_polynomial!(Polynomial7, 7);
bind_polynomial!(Polynomial8, 8);

/// Two linear pieces on two spans.
#[pyclass(name = "PiecewisePolynomial2")]
pub struct PyPiecewisePolynomial2 {
    inner: PiecewisePolynomial<2, 2, 1>,
}

#[pymethods]
impl PyPiecewisePolynomial2 {
    /// Two coefficient rows of length 2 and two positive spans.
    #[new]
    fn new(coefficient_rows: Vec<Vec<f64>>, spans: Vec<f64>) -> PyResult<Self> {
        if coefficient_rows.len() != 2 || spans.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "PiecewisePolynomial2 needs 2 pieces and 2 spans",
            ));
        }
        let first = Polynomial::new(polynomial_coefficients::<2>(coefficient_rows[0].clone())?);
        let second = Polynomial::new(polynomial_coefficients::<2>(coefficient_rows[1].clone())?);
        let pieces = [[first], [second]];
        let span_array = [spans[0], spans[1]];
        Ok(Self {
            inner: PiecewisePolynomial::try_from_pieces(&pieces, &span_array)
                .map_err(errors::polynomial_error)?,
        })
    }

    /// Value at `parameter` along the concatenated pieces.
    fn evaluate(&self, parameter: f64) -> PyResult<f64> {
        let point = self
            .inner
            .evaluate(parameter)
            .map_err(errors::polynomial_error)?;
        Ok(point.into_array()[0])
    }

    fn __len__(&self) -> usize {
        2
    }

    fn __repr__(&self) -> String {
        "PiecewisePolynomial2(pieces=2)".to_string()
    }
}

/// Two-variable polynomial (up to 4 terms).
#[pyclass(name = "MultivariatePolynomial2")]
pub struct PyMultivariatePolynomial2 {
    inner: MultivariatePolynomial<2, 4>,
}

#[pymethods]
impl PyMultivariatePolynomial2 {
    /// Terms as `(coefficient, [exponent_x, exponent_y])`.
    #[new]
    fn new(terms: Vec<(f64, Vec<u32>)>) -> PyResult<Self> {
        let mut converted = Vec::with_capacity(terms.len());
        for (coefficient, exponents) in terms {
            if exponents.len() != 2 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "each term needs 2 exponents",
                ));
            }
            converted.push(MultivariateTerm::new(
                coefficient,
                [exponents[0], exponents[1]],
            ));
        }
        Ok(Self {
            inner: MultivariatePolynomial::try_from_terms(&converted)
                .map_err(errors::polynomial_error)?,
        })
    }

    /// Value at a 2-vector of variables.
    fn evaluate(&self, variables: Vec<f64>) -> PyResult<f64> {
        if variables.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "MultivariatePolynomial2 needs 2 variables",
            ));
        }
        Ok(self.inner.evaluate(&[variables[0], variables[1]]))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("MultivariatePolynomial2(terms={})", self.inner.len())
    }
}

pub(crate) fn register<'python>(module: &Bound<'python, PyModule>) -> PyResult<()> {
    module.add_class::<Polynomial2>()?;
    module.add_class::<Polynomial3>()?;
    module.add_class::<Polynomial4>()?;
    module.add_class::<Polynomial5>()?;
    module.add_class::<Polynomial6>()?;
    module.add_class::<Polynomial7>()?;
    module.add_class::<Polynomial8>()?;
    module.add_class::<PyPiecewisePolynomial2>()?;
    module.add_class::<PyMultivariatePolynomial2>()?;
    Ok(())
}
