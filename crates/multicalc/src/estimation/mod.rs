//! State estimation from noisy measurements.
//!
//! - [`KalmanFilter`] — the linear filter, a single Gaussian belief, over a [`KalmanModel`].
//! - [`ExtendedKalmanFilter`] — nonlinear models, differentiated for their Jacobians each step.
//! - [`UnscentedKalmanFilter`] — nonlinear models sampled at a spread of points, never
//!   differentiated.
//! - [`ParticleFilter`] — a cloud of weighted samples, for non-Gaussian or multi-peaked beliefs
//!   (`alloc` only).
//! - [`CovarianceUpdate`] — how the Kalman filters recompute the covariance.

mod extended_kalman_filter;
mod kalman_filter;
mod unscented_kalman_filter;

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
mod particle_filter;

pub use extended_kalman_filter::ExtendedKalmanFilter;
pub use kalman_filter::{CovarianceUpdate, KalmanFilter, KalmanModel};
pub use unscented_kalman_filter::UnscentedKalmanFilter;

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub use particle_filter::{GaussianLikelihood, Likelihood, ParticleFilter, ResamplingScheme};
