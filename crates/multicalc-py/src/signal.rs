use multicalc::signal_processing::{
    Biquad, BiquadCoefficients, Deadband, MovingAverage, OnePoleLowPass, RunningMedian,
    SlewRateLimiter,
};
use pyo3::prelude::*;

use crate::errors;

/// First-order IIR low-pass.
#[pyclass(name = "OnePoleLowPass")]
pub struct PyOnePoleLowPass {
    inner: OnePoleLowPass,
}

#[pymethods]
impl PyOnePoleLowPass {
    /// Smoothing factor in `(0, 1]`.
    #[new]
    fn new(smoothing: f64) -> PyResult<Self> {
        Ok(Self {
            inner: OnePoleLowPass::new(smoothing).map_err(errors::signal_error)?,
        })
    }

    fn filter(&mut self, sample: f64) -> f64 {
        self.inner.filter(sample)
    }
}

/// Direct-form II biquad.
#[pyclass(name = "Biquad")]
pub struct PyBiquad {
    inner: Biquad,
}

#[pymethods]
impl PyBiquad {
    /// Low-pass coefficients from cutoff (Hz), `quality_factor`, and sample period.
    #[staticmethod]
    fn low_pass(cutoff_hz: f64, quality_factor: f64, timestep: f64) -> PyResult<Self> {
        let coefficients = BiquadCoefficients::low_pass(cutoff_hz, quality_factor, timestep)
            .map_err(errors::signal_error)?;
        Ok(Self {
            inner: Biquad::new(coefficients),
        })
    }

    fn filter(&mut self, sample: f64) -> f64 {
        self.inner.filter(sample)
    }
}

/// Symmetric deadband around zero.
#[pyclass(name = "Deadband")]
pub struct PyDeadband {
    inner: Deadband,
}

#[pymethods]
impl PyDeadband {
    /// Half-width of the zero region.
    #[staticmethod]
    fn plain(threshold: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Deadband::plain(threshold).map_err(errors::signal_error)?,
        })
    }

    fn apply(&self, sample: f64) -> f64 {
        self.inner.apply(sample)
    }
}

/// Rate limiter on a scalar command.
#[pyclass(name = "SlewRateLimiter")]
pub struct PySlewRateLimiter {
    inner: SlewRateLimiter,
}

#[pymethods]
impl PySlewRateLimiter {
    /// Rise/fall rates per second and sample period.
    #[new]
    fn new(rise_per_second: f64, fall_per_second: f64, timestep: f64) -> PyResult<Self> {
        Ok(Self {
            inner: SlewRateLimiter::new(rise_per_second, fall_per_second, timestep)
                .map_err(errors::signal_error)?,
        })
    }

    fn filter(&mut self, target: f64) -> f64 {
        self.inner.filter(target)
    }
}

/// Moving average of window 4.
#[pyclass(name = "MovingAverage4")]
pub struct PyMovingAverage4 {
    inner: MovingAverage<4>,
}

#[pymethods]
impl PyMovingAverage4 {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self {
            inner: MovingAverage::<4>::new().map_err(errors::signal_error)?,
        })
    }

    fn filter(&mut self, sample: f64) -> f64 {
        self.inner.filter(sample)
    }
}

/// Running median of window 5.
#[pyclass(name = "RunningMedian5")]
pub struct PyRunningMedian5 {
    inner: RunningMedian<5>,
}

#[pymethods]
impl PyRunningMedian5 {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self {
            inner: RunningMedian::new().map_err(errors::signal_error)?,
        })
    }

    fn filter(&mut self, sample: f64) -> f64 {
        self.inner.filter(sample)
    }
}

pub(crate) fn register<'python>(module: &Bound<'python, PyModule>) -> PyResult<()> {
    module.add_class::<PyOnePoleLowPass>()?;
    module.add_class::<PyBiquad>()?;
    module.add_class::<PyDeadband>()?;
    module.add_class::<PySlewRateLimiter>()?;
    module.add_class::<PyMovingAverage4>()?;
    module.add_class::<PyRunningMedian5>()?;
    Ok(())
}
