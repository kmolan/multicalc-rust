use multicalc::linear_algebra::{Matrix, Vector};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn vector_from_list<const COUNT: usize>(values: Vec<f64>) -> PyResult<Vector<COUNT>> {
    if values.len() != COUNT {
        return Err(PyValueError::new_err(format!(
            "expected {COUNT} values, got {}",
            values.len()
        )));
    }
    let array: [f64; COUNT] = values
        .try_into()
        .map_err(|_| PyValueError::new_err("invalid vector length"))?;
    Ok(Vector::new(array))
}

pub(crate) fn vector_to_list<const COUNT: usize>(vector: Vector<COUNT>) -> Vec<f64> {
    vector.into_array().to_vec()
}

pub(crate) fn matrix_from_rows<const ROWS: usize, const COLS: usize>(
    rows: Vec<Vec<f64>>,
) -> PyResult<Matrix<ROWS, COLS>> {
    if rows.len() != ROWS {
        return Err(PyValueError::new_err(format!(
            "expected {ROWS} rows, got {}",
            rows.len()
        )));
    }
    let mut data = [[0.0; COLS]; ROWS];
    for (row_index, row) in rows.into_iter().enumerate() {
        if row.len() != COLS {
            return Err(PyValueError::new_err(format!(
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

pub(crate) fn matrix_to_rows<const ROWS: usize, const COLS: usize>(
    matrix: Matrix<ROWS, COLS>,
) -> Vec<Vec<f64>> {
    matrix
        .as_slice_rows()
        .iter()
        .map(|row| row.to_vec())
        .collect()
}
