use multicalc::plant::{MultirotorMixer, RotorLag};
use pyo3::prelude::*;

use crate::convert::{vector_from_list, vector_to_list};
use crate::errors;
use crate::spatial::PyWrench;

/// Four-rotor mixer (quadrotor X).
#[pyclass(name = "MultirotorMixer4")]
pub struct PyMultirotorMixer4 {
    inner: MultirotorMixer<4>,
}

#[pymethods]
impl PyMultirotorMixer4 {
    /// Quadrotor-X mixer with thrust limits.
    #[staticmethod]
    fn quadrotor_x(
        arm_length: f64,
        torque_per_thrust: f64,
        minimum_thrust: f64,
        maximum_thrust: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: MultirotorMixer::<4>::quadrotor_x(
                arm_length,
                torque_per_thrust,
                minimum_thrust,
                maximum_thrust,
            )
            .map_err(errors::plant_error)?,
        })
    }

    /// Rotor thrusts and whether any axis saturated. Returns `(thrusts, saturated)`.
    fn rotor_thrusts(
        &self,
        collective_thrust: f64,
        torque: Vec<f64>,
    ) -> PyResult<(Vec<f64>, bool)> {
        let commands = self
            .inner
            .rotor_thrusts(collective_thrust, vector_from_list::<3>(torque)?);
        Ok((vector_to_list(commands.thrusts()), commands.saturated()))
    }

    /// Wrench from four rotor thrusts.
    fn wrench(&self, thrusts: Vec<f64>) -> PyResult<PyWrench> {
        Ok(PyWrench {
            inner: self.inner.wrench(vector_from_list::<4>(thrusts)?),
        })
    }
}

/// First-order lag on four rotor commands.
#[pyclass(name = "RotorLag4")]
pub struct PyRotorLag4 {
    inner: RotorLag<4>,
}

#[pymethods]
impl PyRotorLag4 {
    /// Time constant and sample period.
    #[new]
    fn new(lag_time: f64, tick: f64) -> PyResult<Self> {
        Ok(Self {
            inner: RotorLag::<4>::new(lag_time, tick).map_err(errors::plant_error)?,
        })
    }

    /// Advance one tick toward `commanded` thrusts.
    fn stepped(&mut self, commanded: Vec<f64>) -> PyResult<Vec<f64>> {
        Ok(vector_to_list(
            self.inner.stepped(vector_from_list::<4>(commanded)?),
        ))
    }

    fn thrusts(&self) -> Vec<f64> {
        vector_to_list(self.inner.thrusts())
    }
}

pub(crate) fn register<'python>(module: &Bound<'python, PyModule>) -> PyResult<()> {
    module.add_class::<PyMultirotorMixer4>()?;
    module.add_class::<PyRotorLag4>()?;
    Ok(())
}
