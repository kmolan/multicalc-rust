use multicalc::kinematics::{
    BodyTwist, DifferentialDrive, Joint, JointParent, KinematicTree, WheelVelocities,
};
use multicalc::linear_algebra::Vector;
use multicalc::spatial::{SE3, SO3};
use pyo3::prelude::*;

use crate::convert::{vector_from_list, vector_to_list};
use crate::errors;

/// Differential-drive kinematics.
#[pyclass(name = "DifferentialDrive")]
pub struct PyDifferentialDrive {
    inner: DifferentialDrive,
}

#[pymethods]
impl PyDifferentialDrive {
    /// Wheel radius and wheelbase.
    #[new]
    fn new(wheel_radius: f64, wheelbase: f64) -> PyResult<Self> {
        Ok(Self {
            inner: DifferentialDrive::new(wheel_radius, wheelbase)
                .map_err(errors::kinematics_error)?,
        })
    }

    /// Body twist `(linear, angular)` from left and right wheel speeds.
    fn forward(&self, left: f64, right: f64) -> (f64, f64) {
        let twist = self.inner.forward(WheelVelocities::new(left, right));
        (twist.linear(), twist.angular())
    }

    /// Left and right wheel speeds from a body twist.
    fn inverse(&self, linear: f64, angular: f64) -> (f64, f64) {
        let wheels = self.inner.inverse(BodyTwist::new(linear, angular));
        (wheels.left(), wheels.right())
    }
}

/// Two-joint planar kinematic tree.
#[pyclass(name = "KinematicTree2")]
pub struct PyKinematicTree2 {
    inner: KinematicTree<2, 2>,
}

#[pymethods]
impl PyKinematicTree2 {
    /// Equal-length two-link planar arm.
    #[staticmethod]
    fn planar_two_link(link_length: f64) -> PyResult<Self> {
        let axis_z = Vector::new([0.0, 0.0, 1.0]);
        let link = SE3::from_parts(SO3::identity(), Vector::new([link_length, 0.0, 0.0]));
        Ok(Self {
            inner: KinematicTree::<2, 2>::try_from_joints(
                &[
                    Joint::revolute(axis_z, SE3::identity()),
                    Joint::revolute(axis_z, link),
                ],
                &[JointParent::World, JointParent::Joint(0)],
            )
            .map_err(errors::kinematics_error)?,
        })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("KinematicTree2(joints={})", self.inner.len())
    }

    /// Translation of the last link for a 2-vector configuration.
    fn forward_kinematics(&self, configuration: Vec<f64>) -> PyResult<Vec<f64>> {
        let configuration = vector_from_list::<2>(configuration)?;
        let state = self
            .inner
            .forward_kinematics(&configuration)
            .map_err(errors::kinematics_error)?;
        let pose = state.pose(1).ok_or_else(|| {
            errors::kinematics_error(multicalc::error::KinematicsError::CapacityExceeded)
        })?;
        Ok(vector_to_list(pose.translation()))
    }
}

pub(crate) fn register<'python>(module: &Bound<'python, PyModule>) -> PyResult<()> {
    module.add_class::<PyDifferentialDrive>()?;
    module.add_class::<PyKinematicTree2>()?;
    Ok(())
}
