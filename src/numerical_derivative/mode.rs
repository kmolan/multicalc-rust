
//In most cases, Central is recommended for the highest accuracy
//If you feel unsure, start with Central and then tweak depending on results
//Not that the accuracy of results also depend on the step size
#[derive(Debug, Copy, Clone)]
pub enum FiniteDifferenceMode
{
    Forward,
    Backward,
    Central
}

///constants used by finite_difference module
pub const DEFAULT_STEP_SIZE: f64 = 1.0e-5;
pub const DEFAULT_STEP_SIZE_MULTIPLIER: f64 = 10.0;