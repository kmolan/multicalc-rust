//! State estimation from noisy measurements.
//!
//! [`KalmanFilter`] is the linear filter: predict rolls the state forward through a matrix model,
//! update folds in a measurement. [`CovarianceUpdate`] selects how the covariance is recomputed.

mod kalman_filter;

pub use kalman_filter::{CovarianceUpdate, KalmanFilter};
