use multicalc::control::{
    FollowTheGap, FollowTheGapOutput, GeometricAttitudeController, Lqr, Pid, ThrustCommand,
    pure_pursuit_curvature, thrust_command_from_acceleration,
};
use pyo3::prelude::*;

use crate::convert::{matrix_from_rows, vector_from_list, vector_to_list};
use crate::errors;
use crate::spatial::{PySE2, PySO3};

/// Discrete PID controller.
#[pyclass(name = "Pid")]
pub struct PyPid {
    inner: Pid,
}

#[pymethods]
impl PyPid {
    /// Gains and sample period. Raises `ControlError` if the timestep is invalid.
    #[new]
    fn new(
        proportional_gain: f64,
        integral_gain: f64,
        derivative_gain: f64,
        timestep: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Pid::new(proportional_gain, integral_gain, derivative_gain, timestep)
                .map_err(errors::control_error)?,
        })
    }

    fn update(&mut self, setpoint: f64, measurement: f64) -> f64 {
        self.inner.update(setpoint, measurement)
    }
}

/// Discrete LQR for 2 states and 1 input.
#[pyclass(name = "Lqr2x1")]
pub struct PyLqr2x1 {
    inner: Lqr<2, 1>,
}

#[pymethods]
impl PyLqr2x1 {
    /// Discrete `state_transition`, `input_model`, `state_cost`, `input_cost` as nested lists.
    #[new]
    fn new(
        state_transition: Vec<Vec<f64>>,
        input_model: Vec<Vec<f64>>,
        state_cost: Vec<Vec<f64>>,
        input_cost: Vec<Vec<f64>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Lqr::new(
                matrix_from_rows::<2, 2>(state_transition)?,
                matrix_from_rows::<2, 1>(input_model)?,
                matrix_from_rows::<2, 2>(state_cost)?,
                matrix_from_rows::<1, 1>(input_cost)?,
            )
            .map_err(errors::control_error)?,
        })
    }

    fn control(&self, state: Vec<f64>) -> PyResult<Vec<f64>> {
        Ok(vector_to_list(
            self.inner.control(vector_from_list::<2>(state)?),
        ))
    }

    /// Raise `ControlError` if the closed-loop discrete system is not stable.
    fn certify_stability(&self) -> PyResult<()> {
        self.inner
            .certify_stability()
            .map(|_| ())
            .map_err(errors::control_error)
    }
}

/// Geometric SO(3) attitude controller.
#[pyclass(name = "GeometricAttitudeController")]
pub struct PyGeometricAttitudeController {
    inner: GeometricAttitudeController,
}

#[pymethods]
impl PyGeometricAttitudeController {
    /// Attitude/rate gains and 3×3 inertia.
    #[new]
    fn new(attitude_gain: f64, rate_gain: f64, inertia: Vec<Vec<f64>>) -> PyResult<Self> {
        Ok(Self {
            inner: GeometricAttitudeController::new(
                attitude_gain,
                rate_gain,
                matrix_from_rows::<3, 3>(inertia)?,
            )
            .map_err(errors::control_error)?,
        })
    }

    fn torque(
        &self,
        attitude: &PySO3,
        body_rate: Vec<f64>,
        desired_attitude: &PySO3,
        desired_body_rate: Vec<f64>,
        desired_body_rate_derivative: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        Ok(vector_to_list(self.inner.torque(
            attitude.inner,
            vector_from_list::<3>(body_rate)?,
            desired_attitude.inner,
            vector_from_list::<3>(desired_body_rate)?,
            vector_from_list::<3>(desired_body_rate_derivative)?,
        )))
    }
}

/// Result of a five-beam follow-the-gap query.
#[pyclass(name = "FollowTheGapOutput")]
pub struct PyFollowTheGapOutput {
    inner: FollowTheGapOutput,
}

#[pymethods]
impl PyFollowTheGapOutput {
    fn heading(&self) -> f64 {
        self.inner.heading()
    }

    fn is_blocked(&self) -> bool {
        self.inner.is_blocked()
    }

    fn minimum_clearance(&self) -> f64 {
        self.inner.minimum_clearance()
    }

    fn body_twist(&self) -> (f64, f64) {
        let twist = self.inner.body_twist();
        (twist.linear(), twist.angular())
    }
}

/// Follow-the-gap on a 5-beam scan.
#[pyclass(name = "FollowTheGap5")]
pub struct PyFollowTheGap5 {
    inner: FollowTheGap<5>,
}

#[pymethods]
impl PyFollowTheGap5 {
    /// Sensor and chassis parameters.
    #[staticmethod]
    fn try_new(
        field_of_view: f64,
        maximum_range: f64,
        chassis_width: f64,
        free_range_threshold: f64,
        cruise_speed: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: FollowTheGap::try_new(
                field_of_view,
                maximum_range,
                chassis_width,
                free_range_threshold,
                cruise_speed,
            )
            .map_err(errors::control_error)?,
        })
    }

    /// Heading from five ranges and a goal angle.
    fn compute(&self, beam_range: Vec<f64>, goal_angle: f64) -> PyResult<PyFollowTheGapOutput> {
        let ranges = vector_from_list::<5>(beam_range)?;
        Ok(PyFollowTheGapOutput {
            inner: self
                .inner
                .compute(ranges.as_array(), goal_angle)
                .map_err(errors::control_error)?,
        })
    }
}

/// Collective thrust plus desired attitude.
#[pyclass(name = "ThrustCommand")]
pub struct PyThrustCommand {
    inner: ThrustCommand,
}

#[pymethods]
impl PyThrustCommand {
    fn attitude(&self) -> PySO3 {
        PySO3 {
            inner: self.inner.attitude(),
        }
    }

    fn thrust_acceleration(&self) -> f64 {
        self.inner.thrust_acceleration()
    }
}

/// Pure-pursuit curvature for an SE(2) pose and lookahead point.
#[pyfunction(name = "pure_pursuit_curvature")]
fn bind_pure_pursuit_curvature(
    pose: &PySE2,
    lookahead_point: Vec<f64>,
    lookahead_distance: f64,
) -> PyResult<f64> {
    let curvature = pure_pursuit_curvature(
        pose.inner,
        vector_from_list::<2>(lookahead_point)?,
        lookahead_distance,
    )
    .map_err(errors::control_error)?;
    Ok(curvature.value())
}

/// Thrust plus attitude from a 3-vector acceleration command.
#[pyfunction(name = "thrust_command_from_acceleration")]
fn bind_thrust_command_from_acceleration(
    acceleration_command: Vec<f64>,
    desired_heading: f64,
    gravity: f64,
) -> PyResult<PyThrustCommand> {
    Ok(PyThrustCommand {
        inner: thrust_command_from_acceleration(
            vector_from_list::<3>(acceleration_command)?,
            desired_heading,
            gravity,
        )
        .map_err(errors::control_error)?,
    })
}

pub(crate) fn register<'python>(module: &Bound<'python, PyModule>) -> PyResult<()> {
    module.add_class::<PyPid>()?;
    module.add_class::<PyLqr2x1>()?;
    module.add_class::<PyGeometricAttitudeController>()?;
    module.add_class::<PyFollowTheGapOutput>()?;
    module.add_class::<PyFollowTheGap5>()?;
    module.add_class::<PyThrustCommand>()?;
    module.add_function(wrap_pyfunction!(bind_pure_pursuit_curvature, module)?)?;
    module.add_function(wrap_pyfunction!(
        bind_thrust_command_from_acceleration,
        module
    )?)?;
    Ok(())
}
