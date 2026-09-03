use pyo3::prelude::*;

use crate::convert::{vector_from_list, vector_to_list};
use crate::errors;
use crate::spatial::{PySO3, PySpatialInertia, PyWrench};

/// Single rigid body with gravity.
#[pyclass(name = "RigidBody")]
pub struct PyRigidBody {
    inner: multicalc::dynamics::RigidBody,
}

#[pymethods]
impl PyRigidBody {
    /// Inertia and gravity 3-vector.
    #[new]
    fn new(inertia: &PySpatialInertia, gravity: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: multicalc::dynamics::RigidBody::new(
                inertia.inner,
                vector_from_list::<3>(gravity)?,
            )
            .map_err(errors::dynamics_error)?,
        })
    }

    /// Linear and angular accelerations given orientation, rate, and applied wrench.
    fn accelerations(
        &self,
        orientation: &PySO3,
        angular_rate: Vec<f64>,
        applied_wrench: &PyWrench,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let accelerations = self.inner.accelerations(
            orientation.inner,
            vector_from_list::<3>(angular_rate)?,
            applied_wrench.inner,
        );
        Ok((
            vector_to_list(accelerations.linear()),
            vector_to_list(accelerations.angular()),
        ))
    }
}

pub(crate) fn register<'python>(module: &Bound<'python, PyModule>) -> PyResult<()> {
    module.add_class::<PyRigidBody>()?;
    Ok(())
}
