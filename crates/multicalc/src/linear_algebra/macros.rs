//! Literal construction macros for [`Vector`](crate::Vector) and [`Matrix`](crate::Matrix).
/// Builds a [`Vector`](crate::Vector) from a comma-separated list of components.
///
/// ```
/// use multicalc::{vector, Vector};
/// let v = vector![1.0, 2.0, 3.0];
/// assert_eq!(v, Vector::new([1.0, 2.0, 3.0]));
/// ```
#[macro_export]
macro_rules! vector {
    ($($component:expr),+ $(,)?) => {
        $crate::Vector::new([$($component),+])
    };
}

/// Builds a [`Matrix`](crate::Matrix) from bracketed row literals.
///
/// ```
/// use multicalc::{matrix, Matrix};
/// let m = matrix![[1.0, 2.0], [3.0, 4.0]];
/// assert_eq!(m, Matrix::new([[1.0, 2.0], [3.0, 4.0]]));
/// ```
///
/// Uneven row lengths are rejected at compile time:
///
/// ```compile_fail
/// use multicalc::matrix;
/// let _ = matrix![[1.0, 2.0], [3.0]];
/// ```
#[macro_export]
macro_rules! matrix {
    ($([$($entry:expr),+ $(,)?]),+ $(,)?) => {
        $crate::Matrix::new([$([$($entry),+]),+])
    };
}
