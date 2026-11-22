use multicalc::linear_algebra::{Matrix, Vector};
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::convert::{matrix_from_rows, matrix_to_rows, vector_from_list, vector_to_list};
use crate::errors;

macro_rules! bind_vector {
    ($py_name:ident, $count:expr) => {
        /// Fixed-length `f64` column vector. Length is in the type name (`Vector2`, `Vector3`, …).
        #[pyclass]
        pub struct $py_name {
            pub(crate) inner: Vector<$count>,
        }

        #[pymethods]
        impl $py_name {
            /// Build from as many values as this vector's length.
            #[new]
            fn new(values: Vec<f64>) -> PyResult<Self> {
                Ok(Self {
                    inner: vector_from_list::<$count>(values)?,
                })
            }

            #[staticmethod]
            fn zeros() -> Self {
                Self {
                    inner: Vector::zeros(),
                }
            }

            fn dot(&self, other: &$py_name) -> f64 {
                self.inner.dot(other.inner)
            }

            fn to_list(&self) -> Vec<f64> {
                vector_to_list(self.inner)
            }

            fn __len__(&self) -> usize {
                $count
            }

            fn __repr__(&self) -> String {
                format!("{}({:?})", stringify!($py_name), self.to_list())
            }
        }
    };
}

macro_rules! bind_square_matrix {
    ($py_name:ident, $vector_name:ident, $size:expr) => {
        /// Square `f64` matrix. Side length is in the type name (`Matrix2`, `Matrix3`, …).
        #[pyclass]
        pub struct $py_name {
            inner: Matrix<$size, $size>,
        }

        #[pymethods]
        impl $py_name {
            /// Build from a list of rows.
            #[new]
            fn new(rows: Vec<Vec<f64>>) -> PyResult<Self> {
                Ok(Self {
                    inner: matrix_from_rows(rows)?,
                })
            }

            #[staticmethod]
            fn zeros() -> Self {
                Self {
                    inner: Matrix::zeros(),
                }
            }

            #[staticmethod]
            fn identity() -> Self {
                Self {
                    inner: Matrix::identity(),
                }
            }

            fn to_list(&self) -> Vec<Vec<f64>> {
                matrix_to_rows(self.inner)
            }

            fn __len__(&self) -> usize {
                $size
            }

            fn __repr__(&self) -> String {
                format!("{}({:?})", stringify!($py_name), self.to_list())
            }

            fn transpose(&self) -> Self {
                Self {
                    inner: self.inner.transpose(),
                }
            }

            /// Solve against `right_hand_side`. Raises `LinalgError` if this matrix is singular.
            fn solve(&self, right_hand_side: &$vector_name) -> PyResult<$vector_name> {
                Ok($vector_name {
                    inner: self
                        .inner
                        .solve(right_hand_side.inner)
                        .map_err(errors::linalg_error)?,
                })
            }

            /// Lower-triangular Cholesky factor. Raises `LinalgError` if the matrix is not SPD.
            fn cholesky(&self) -> PyResult<Self> {
                let factor = self.inner.cholesky().map_err(errors::linalg_error)?;
                Ok(Self {
                    inner: factor.lower(),
                })
            }

            /// LU factorisation as `(lower, upper, permutation)`.
            fn lu_decompose<'python>(
                &self,
                python: Python<'python>,
            ) -> PyResult<(Self, Self, Py<PyList>)> {
                let factor = self.inner.lu_decompose().map_err(errors::linalg_error)?;
                let lower = Self {
                    inner: factor.lower(),
                };
                let upper = Self {
                    inner: factor.upper(),
                };
                let perm: Vec<usize> = factor.permutation().to_vec();
                Ok((lower, upper, PyList::new(python, perm)?.unbind()))
            }

            /// SVD as `(left, singular_values, right)`.
            fn svd<'python>(&self, python: Python<'python>) -> PyResult<(Self, Py<PyList>, Self)> {
                let decomposition = self.inner.svd().map_err(errors::linalg_error)?;
                let left = Self {
                    inner: decomposition.left(),
                };
                let values = vector_to_list(decomposition.singular_values());
                let right = Self {
                    inner: decomposition.right(),
                };
                Ok((left, PyList::new(python, values)?.unbind(), right))
            }
        }
    };
}

bind_vector!(Vector2, 2);
bind_vector!(Vector3, 3);
bind_vector!(Vector4, 4);
bind_vector!(Vector6, 6);

bind_square_matrix!(Matrix2, Vector2, 2);
bind_square_matrix!(Matrix3, Vector3, 3);
bind_square_matrix!(Matrix4, Vector4, 4);
bind_square_matrix!(Matrix6, Vector6, 6);

pub(crate) fn register<'python>(module: &Bound<'python, PyModule>) -> PyResult<()> {
    module.add_class::<Vector2>()?;
    module.add_class::<Vector3>()?;
    module.add_class::<Vector4>()?;
    module.add_class::<Vector6>()?;
    module.add_class::<Matrix2>()?;
    module.add_class::<Matrix3>()?;
    module.add_class::<Matrix4>()?;
    module.add_class::<Matrix6>()?;
    Ok(())
}
