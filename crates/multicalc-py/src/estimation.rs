use multicalc::estimation::{
    GaussianLikelihood, KalmanFilter, KalmanModel, MadgwickFilter, MahonyFilter, ParticleFilter,
};
use multicalc::linear_algebra::Matrix;
use pyo3::prelude::*;

use crate::callables::PythonVectorFn2x2;
use crate::convert::{matrix_from_rows, matrix_to_rows, vector_from_list, vector_to_list};
use crate::errors;
use crate::spatial::PySO3;

/// Linear Kalman filter with 2 states and 1 measurement.
#[pyclass(name = "KalmanFilter2x1")]
pub struct PyKalmanFilter2x1 {
    inner: KalmanFilter<2, 1>,
}

#[pymethods]
impl PyKalmanFilter2x1 {
    /// Initial state/covariance and the discrete Kalman model matrices.
    #[new]
    fn new(
        initial_state: Vec<f64>,
        initial_covariance: Vec<Vec<f64>>,
        state_transition: Vec<Vec<f64>>,
        measurement_model: Vec<Vec<f64>>,
        process_noise: Vec<Vec<f64>>,
        measurement_noise: Vec<Vec<f64>>,
    ) -> PyResult<Self> {
        let model = KalmanModel {
            state_transition: matrix_from_rows::<2, 2>(state_transition)?,
            measurement_model: matrix_from_rows::<1, 2>(measurement_model)?,
            process_noise: matrix_from_rows::<2, 2>(process_noise)?,
            measurement_noise: matrix_from_rows::<1, 1>(measurement_noise)?,
        };
        Ok(Self {
            inner: KalmanFilter::new(
                vector_from_list(initial_state)?,
                matrix_from_rows::<2, 2>(initial_covariance)?,
                model,
            ),
        })
    }

    fn predict(&mut self) {
        self.inner.predict();
    }

    /// Measurement update. `measurement` is length 1.
    fn update(&mut self, measurement: Vec<f64>) -> PyResult<()> {
        self.inner
            .update(vector_from_list::<1>(measurement)?)
            .map_err(errors::estimation_error)
    }

    fn state(&self) -> Vec<f64> {
        vector_to_list(self.inner.state())
    }

    fn covariance(&self) -> Vec<Vec<f64>> {
        matrix_to_rows(self.inner.covariance())
    }
}

/// Madgwick IMU/AHRS filter.
#[pyclass(name = "MadgwickFilter")]
pub struct PyMadgwickFilter {
    inner: MadgwickFilter,
}

#[pymethods]
impl PyMadgwickFilter {
    #[new]
    fn new(orientation: &PySO3) -> Self {
        Self {
            inner: MadgwickFilter::new(orientation.inner),
        }
    }

    /// Gyro, accelerometer, optional magnetometer (each length 3), and timestep.
    fn step(
        &mut self,
        gyroscope: Vec<f64>,
        accelerometer: Vec<f64>,
        magnetometer: Option<Vec<f64>>,
        timestep: f64,
    ) -> PyResult<()> {
        let field = match magnetometer {
            Some(values) => Some(vector_from_list::<3>(values)?),
            None => None,
        };
        self.inner
            .step(
                vector_from_list::<3>(gyroscope)?,
                vector_from_list::<3>(accelerometer)?,
                field,
                timestep,
            )
            .map_err(errors::estimation_error)
    }

    fn orientation(&self) -> PySO3 {
        PySO3 {
            inner: self.inner.orientation(),
        }
    }
}

/// Mahony IMU/AHRS filter.
#[pyclass(name = "MahonyFilter")]
pub struct PyMahonyFilter {
    inner: MahonyFilter,
}

#[pymethods]
impl PyMahonyFilter {
    #[new]
    fn new(orientation: &PySO3) -> Self {
        Self {
            inner: MahonyFilter::new(orientation.inner),
        }
    }

    /// Gyro, accelerometer, optional magnetometer (each length 3), and timestep.
    fn step(
        &mut self,
        gyroscope: Vec<f64>,
        accelerometer: Vec<f64>,
        magnetometer: Option<Vec<f64>>,
        timestep: f64,
    ) -> PyResult<()> {
        let field = match magnetometer {
            Some(values) => Some(vector_from_list::<3>(values)?),
            None => None,
        };
        self.inner
            .step(
                vector_from_list::<3>(gyroscope)?,
                vector_from_list::<3>(accelerometer)?,
                field,
                timestep,
            )
            .map_err(errors::estimation_error)
    }

    fn orientation(&self) -> PySO3 {
        PySO3 {
            inner: self.inner.orientation(),
        }
    }
}

/// Particle filter with 2-D state and 2-D measurement.
#[pyclass(name = "ParticleFilter2x2")]
pub struct PyParticleFilter2x2 {
    inner: ParticleFilter<2, 2>,
    measurement_noise: Matrix<2, 2>,
}

#[pymethods]
impl PyParticleFilter2x2 {
    /// Particle count, prior mean/covariance, process noise, measurement noise, and RNG seed.
    #[new]
    fn new(
        particle_count: usize,
        initial_mean: Vec<f64>,
        initial_covariance: Vec<Vec<f64>>,
        process_noise: Vec<Vec<f64>>,
        measurement_noise: Vec<Vec<f64>>,
        seed: u64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: ParticleFilter::new(
                particle_count,
                vector_from_list::<2>(initial_mean)?,
                matrix_from_rows::<2, 2>(initial_covariance)?,
                matrix_from_rows::<2, 2>(process_noise)?,
                seed,
            )
            .map_err(errors::estimation_error)?,
            measurement_noise: matrix_from_rows::<2, 2>(measurement_noise)?,
        })
    }

    /// Propagate particles with `process_model` (two state components in, two out).
    fn predict(&mut self, process_model: Py<PyAny>) -> PyResult<()> {
        let model = PythonVectorFn2x2::new(process_model);
        model.finish(self.inner.predict(&model).map_err(errors::estimation_error))
    }

    /// Weight particles with `measurement_model` (two state components to a 2-vector prediction) and a 2-vector measurement.
    fn update(&mut self, measurement_model: Py<PyAny>, measurement: Vec<f64>) -> PyResult<()> {
        let likelihood = GaussianLikelihood::<2>::new(self.measurement_noise)
            .map_err(errors::estimation_error)?;
        let model = PythonVectorFn2x2::new(measurement_model);
        model.finish(
            self.inner
                .update(&model, &likelihood, vector_from_list::<2>(measurement)?)
                .map_err(errors::estimation_error),
        )
    }

    fn mean(&self) -> Vec<f64> {
        vector_to_list(self.inner.mean())
    }
}

pub(crate) fn register<'python>(module: &Bound<'python, PyModule>) -> PyResult<()> {
    module.add_class::<PyKalmanFilter2x1>()?;
    module.add_class::<PyMadgwickFilter>()?;
    module.add_class::<PyMahonyFilter>()?;
    module.add_class::<PyParticleFilter2x2>()?;
    Ok(())
}
