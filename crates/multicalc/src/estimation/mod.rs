//! State estimation from noisy measurements.
//!
//! - [`KalmanFilter`] — the linear filter, a single Gaussian belief, over a [`KalmanModel`].
//! - [`ExtendedKalmanFilter`] — nonlinear models, differentiated for their Jacobians each step.
//! - [`UnscentedKalmanFilter`] — nonlinear models sampled at a spread of points, never
//!   differentiated.
//! - [`ErrorStateKalmanFilter`] — an IMU-driven filter tracking a body's place, motion, facing,
//!   and its own sensors' steady offsets, with the facing kept on the rotation group.
//! - [`MahonyFilter`] — a body's facing from a turn-rate sensor pulled onto an accelerometer and a
//!   magnetometer, harder the more wrong it is.
//! - [`MadgwickFilter`] — the same, nudged a fixed amount each tick instead.
//! - [`ParticleFilter`] — a cloud of weighted samples, for non-Gaussian or multi-peaked beliefs
//!   (`alloc` only).
//! - [`CovarianceUpdate`] — how the Kalman filters recompute the covariance.
//! - [`MonteCarloLocalizer`] — Monte Carlo Localization using particle filter.
//! - [`models`] — ready-made process and measurement models to drive the filters with:
//!   [`ConstantTurnAndSpeed`], [`DirectMeasurement`], and [`residual_with_wrapped_angles`].

mod attitude_correction;
mod error_state_kalman_filter;
mod extended_kalman_filter;
mod kalman_filter;
mod madgwick_filter;
mod mahony_filter;
pub mod models;
mod unscented_kalman_filter;

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
mod monte_carlo_localizer;

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
mod particle_filter;

pub use error_state_kalman_filter::{
    ErrorStateKalmanFilter, ImuNoise, NominalState, NominalStateFn,
};
pub use extended_kalman_filter::ExtendedKalmanFilter;
pub use kalman_filter::{CovarianceUpdate, KalmanFilter, KalmanModel};
pub use madgwick_filter::MadgwickFilter;
pub use mahony_filter::MahonyFilter;
pub use models::{ConstantTurnAndSpeed, DirectMeasurement, residual_with_wrapped_angles};
pub use unscented_kalman_filter::UnscentedKalmanFilter;

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub use monte_carlo_localizer::{BeamModel, InitialParticleCloud, MonteCarloLocalizer};

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub use particle_filter::{GaussianLikelihood, Likelihood, ParticleFilter, ResamplingScheme};
