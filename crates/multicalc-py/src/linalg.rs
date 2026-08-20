use multicalc::linear_algebra::{Matrix, Matrix2D, Matrix3D, Vector};
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::errors;

#[pyclass(name = "Vector4")]
pub struct PyVector4 {
    inner: Vector<4>,
}

#[pymethods]
impl PyVector4 {
    #[new]
    fn new(values: Vec<f64>) -> PyResult<Self> {
        if values.len() != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Vector4 expects 4 values, got {}",
                values.len()
            )));
        }
        let array: [f64; 4] = values
            .try_into()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Vector4 expects 4 values"))?;
        Ok(Self {
            inner: Vector::new(array),
        })
    }

    fn dot(&self, other: &PyVector4) -> f64 {
        self.inner.dot(other.inner)
    }
}

#[pyclass(name = "Matrix2")]
pub struct PyMatrix2 {
    inner: Matrix2D,
}

#[pymethods]
impl PyMatrix2 {
    #[new]
    fn new(rows: Vec<Vec<f64>>) -> PyResult<Self> {
        Ok(Self {
            inner: matrix_from_rows::<2, 2>(rows)?,
        })
    }

    fn cholesky(&self) -> PyResult<()> {
        let _ = self.inner.cholesky().map_err(errors::linalg_error)?;
        Ok(())
    }
}

#[pyclass(name = "Matrix3")]
pub struct PyMatrix3 {
    inner: Matrix3D,
}

#[pymethods]
impl PyMatrix3 {
    #[new]
    fn new(rows: Vec<Vec<f64>>) -> PyResult<Self> {
        Ok(Self {
            inner: matrix_from_rows::<3, 3>(rows)?,
        })
    }

    #[staticmethod]
    fn zeros() -> Self {
        Self {
            inner: Matrix3D::zeros(),
        }
    }

    fn lu_decompose(&self) -> PyResult<()> {
        let _ = self.inner.lu_decompose().map_err(errors::linalg_error)?;
        Ok(())
    }

    fn svd<'py>(&self, python: Python<'py>) -> PyResult<Py<PyList>> {
        let decomposition = self.inner.svd().map_err(errors::linalg_error)?;
        let values: Vec<f64> = decomposition.singular_values().as_slice().to_vec();
        Ok(PyList::new(python, values)?.unbind())
    }
}

fn matrix_from_rows<const ROWS: usize, const COLS: usize>(
    rows: Vec<Vec<f64>>,
) -> PyResult<Matrix<ROWS, COLS>> {
    if rows.len() != ROWS {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "expected {ROWS} rows, got {}",
            rows.len()
        )));
    }
    let mut data = [[0.0; COLS]; ROWS];
    for (row_index, row) in rows.into_iter().enumerate() {
        if row.len() != COLS {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "row {row_index} expected {COLS} values, got {}",
                row.len()
            )));
        }
        for (col_index, value) in row.into_iter().enumerate() {
            data[row_index][col_index] = value;
        }
    }
    Ok(Matrix::new(data))
}

pub(crate) fn register<'py>(module: &Bound<'py, PyModule>) -> PyResult<()> {
    module.add_class::<PyVector4>()?;
    module.add_class::<PyMatrix2>()?;
    module.add_class::<PyMatrix3>()?;
    Ok(())
}
