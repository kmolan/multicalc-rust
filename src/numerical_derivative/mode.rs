
//In most cases, CentralFiniteDifference is recommended for the highest accuracy
//If you feel unsure, start with CentralFiniteDifference and then tweak depending on results
//Not that the accuracy of results also depend on the step size
#[derive(Debug, Copy, Clone)]
pub enum FiniteDifferenceMode
{
    Forward,
    Backward,
    Central
}

pub const DEFAULT_STEP_SIZE: f64 = 1.0e-5;