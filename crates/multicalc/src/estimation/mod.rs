//! State estimation from noisy measurements.
//!
//! - [`KalmanFilter`] — the linear filter, a single Gaussian belief.
//! - [`ExtendedKalmanFilter`] — nonlinear models, differentiated for their Jacobians each step.
//! - [`ParticleFilter`] — a cloud of weighted samples, for non-Gaussian or multi-peaked beliefs
//!   (`alloc` only).
//! - [`CovarianceUpdate`] — how the Kalman filters recompute the covariance.

mod extended_kalman_filter;
mod kalman_filter;

#[cfg(feature = "alloc")]
mod particle_filter;

pub use extended_kalman_filter::ExtendedKalmanFilter;
pub use kalman_filter::{CovarianceUpdate, KalmanFilter};

#[cfg(feature = "alloc")]
pub use particle_filter::{GaussianLikelihood, Likelihood, ParticleFilter, ResamplingScheme};
