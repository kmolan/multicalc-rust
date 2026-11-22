use multicalc::linear_algebra::Vector;
use multicalc::spatial::{
    FreeJointState, Quaternion, SE2, SE3, SO2, SO3, SpatialInertia, Twist, Wrench,
};
use pyo3::prelude::*;

use crate::convert::{vector_from_list, vector_to_list};
use crate::errors;

/// Unit quaternion `(real_part, vector_i, vector_j, vector_k)` for 3D rotation.
#[pyclass(name = "Quaternion")]
pub struct PyQuaternion {
    inner: Quaternion,
}

#[pymethods]
impl PyQuaternion {
    #[new]
    fn new(real_part: f64, vector_i: f64, vector_j: f64, vector_k: f64) -> Self {
        Self {
            inner: Quaternion::new(real_part, vector_i, vector_j, vector_k),
        }
    }

    #[staticmethod]
    fn identity() -> Self {
        Self {
            inner: Quaternion::identity(),
        }
    }

    fn as_array(&self) -> [f64; 4] {
        self.inner.as_array()
    }

    fn transform_point(&self, point: Vec<f64>) -> PyResult<Vec<f64>> {
        let point = vector_from_list::<3>(point)?;
        Ok(vector_to_list(self.inner.transform_point(point)))
    }
}

/// 2D rotation (Lie group SO(2)).
#[pyclass(name = "SO2")]
pub struct PySO2 {
    pub(crate) inner: SO2,
}

#[pymethods]
impl PySO2 {
    #[staticmethod]
    fn identity() -> Self {
        Self {
            inner: SO2::identity(),
        }
    }

    /// Rotation by `theta` radians.
    #[staticmethod]
    fn exp(theta: f64) -> Self {
        Self {
            inner: SO2::exp(theta),
        }
    }

    fn act(&self, point: Vec<f64>) -> PyResult<Vec<f64>> {
        let point = vector_from_list::<2>(point)?;
        Ok(vector_to_list(self.inner.act(point)))
    }
}

/// 3D rotation (Lie group SO(3)).
#[pyclass(name = "SO3")]
pub struct PySO3 {
    pub(crate) inner: SO3,
}

#[pymethods]
impl PySO3 {
    #[staticmethod]
    fn identity() -> Self {
        Self {
            inner: SO3::identity(),
        }
    }

    /// Rotation from a 3-vector (axis-angle).
    #[staticmethod]
    fn exp(rotation_vector: [f64; 3]) -> Self {
        Self {
            inner: SO3::exp(Vector::new(rotation_vector)),
        }
    }

    fn act(&self, point: [f64; 3]) -> [f64; 3] {
        self.inner.act(Vector::new(point)).into_array()
    }
}

/// 2D rigid transform (Lie group SE(2)).
#[pyclass(name = "SE2")]
pub struct PySE2 {
    pub(crate) inner: SE2,
}

#[pymethods]
impl PySE2 {
    #[staticmethod]
    fn identity() -> Self {
        Self {
            inner: SE2::identity(),
        }
    }

    /// Pose from rotation and 2-vector translation.
    #[staticmethod]
    fn from_parts(rotation: &PySO2, translation: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: SE2::from_parts(rotation.inner, vector_from_list::<2>(translation)?),
        })
    }

    fn act(&self, point: Vec<f64>) -> PyResult<Vec<f64>> {
        let point = vector_from_list::<2>(point)?;
        Ok(vector_to_list(self.inner.act(point)))
    }
}

/// 3D rigid transform (Lie group SE(3)).
#[pyclass(name = "SE3")]
pub struct PySE3 {
    pub(crate) inner: SE3,
}

#[pymethods]
impl PySE3 {
    #[staticmethod]
    fn identity() -> Self {
        Self {
            inner: SE3::identity(),
        }
    }

    /// Pose from rotation and 3-vector translation.
    #[staticmethod]
    fn from_parts(rotation: &PySO3, translation: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: SE3::from_parts(rotation.inner, vector_from_list::<3>(translation)?),
        })
    }

    fn act(&self, point: Vec<f64>) -> PyResult<Vec<f64>> {
        let point = vector_from_list::<3>(point)?;
        Ok(vector_to_list(self.inner.act(point)))
    }
}

/// Spatial velocity: linear then angular, six components.
#[pyclass(name = "Twist")]
pub struct PyTwist {
    pub(crate) inner: Twist,
}

#[pymethods]
impl PyTwist {
    #[new]
    fn new(linear: Vec<f64>, angular: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: Twist::new(
                vector_from_list::<3>(linear)?,
                vector_from_list::<3>(angular)?,
            ),
        })
    }

    #[staticmethod]
    fn zeros() -> Self {
        Self {
            inner: Twist::zeros(),
        }
    }

    fn as_array(&self) -> [f64; 6] {
        self.inner.as_array()
    }
}

/// Spatial force: force then torque, six components.
#[pyclass(name = "Wrench")]
pub struct PyWrench {
    pub(crate) inner: Wrench,
}

#[pymethods]
impl PyWrench {
    #[new]
    fn new(force: Vec<f64>, torque: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: Wrench::new(
                vector_from_list::<3>(force)?,
                vector_from_list::<3>(torque)?,
            ),
        })
    }

    #[staticmethod]
    fn zeros() -> Self {
        Self {
            inner: Wrench::zeros(),
        }
    }

    fn as_array(&self) -> [f64; 6] {
        self.inner.as_array()
    }
}

/// Rigid-body spatial inertia.
#[pyclass(name = "SpatialInertia")]
pub struct PySpatialInertia {
    pub(crate) inner: SpatialInertia,
}

#[pymethods]
impl PySpatialInertia {
    /// Mass, center of mass, and principal inertia diagonals.
    #[staticmethod]
    fn from_diagonal_inertia(
        mass: f64,
        center_of_mass: Vec<f64>,
        diagonal: Vec<f64>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: SpatialInertia::from_diagonal_inertia(
                mass,
                vector_from_list::<3>(center_of_mass)?,
                vector_from_list::<3>(diagonal)?,
            )
            .map_err(errors::spatial_error)?,
        })
    }

    fn mass(&self) -> f64 {
        self.inner.mass()
    }
}

/// Free-flying rigid body pose and twist.
#[pyclass(name = "FreeJointState")]
pub struct PyFreeJointState {
    inner: FreeJointState,
}

#[pymethods]
impl PyFreeJointState {
    #[staticmethod]
    fn identity() -> Self {
        Self {
            inner: FreeJointState::identity(),
        }
    }

    fn pose(&self) -> PySE3 {
        PySE3 {
            inner: self.inner.pose(),
        }
    }

    fn velocity(&self) -> PyTwist {
        PyTwist {
            inner: self.inner.velocity(),
        }
    }
}

pub(crate) fn register<'python>(module: &Bound<'python, PyModule>) -> PyResult<()> {
    module.add_class::<PyQuaternion>()?;
    module.add_class::<PySO2>()?;
    module.add_class::<PySO3>()?;
    module.add_class::<PySE2>()?;
    module.add_class::<PySE3>()?;
    module.add_class::<PyTwist>()?;
    module.add_class::<PyWrench>()?;
    module.add_class::<PySpatialInertia>()?;
    module.add_class::<PyFreeJointState>()?;
    Ok(())
}
