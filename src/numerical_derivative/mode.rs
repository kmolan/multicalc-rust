
//In most cases, CentralFixedStep is recommended for the highest accuracy
//If you feel unsure, start with CentralFixedStep and then tweak depending on results
//Not that the accuracy of results also depend on the step size
#[derive(Debug, Copy, Clone)]
pub enum FixedStepMode
{
    Forward,
    Backward,
    Central
}

pub const DEFAULT_STEP_SIZE: f64 = 1.0e-5; //default value copied from https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.approx_fprime.html