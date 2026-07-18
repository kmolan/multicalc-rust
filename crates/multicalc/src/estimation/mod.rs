//! State estimation from noisy measurements.
//!
//! [`KalmanFilter`] is the linear filter: predict rolls the state forward through a matrix model,
//! update folds in a measurement. [`ExtendedKalmanFilter`] takes nonlinear process and measurement
//! models as [`VectorFn`](crate::scalar::VectorFn)s instead, differentiating them for the Jacobians
//! each step. [`CovarianceUpdate`] selects how either filter recomputes the covariance.

mod extended_kalman_filter;
mod kalman_filter;

pub use extended_kalman_filter::ExtendedKalmanFilter;
pub use kalman_filter::{CovarianceUpdate, KalmanFilter};
