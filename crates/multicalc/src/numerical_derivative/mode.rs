/// The finite-difference stencil used to approximate a derivative.
///
/// Central is the most accurate for most cases; start there and tweak the mode and step size
/// if the result needs it.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FiniteDifferenceMode {
    /// Samples at the point and one step forward.
    Forward,
    /// Samples at the point and one step backward.
    Backward,
    /// Samples one step either side of the point; most accurate.
    Central,
}

/// Default finite-difference step size.
pub const DEFAULT_STEP_SIZE: f64 = 1.0e-5;
/// Default factor the step is scaled by at each recursion level (third derivatives and higher).
pub const DEFAULT_STEP_SIZE_MULTIPLIER: f64 = 10.0;
