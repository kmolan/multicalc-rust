
//In most cases, CentralFixedStep is recommended for the highest accuracy
//If you feel unsure, start with CentralFixedStep and then tweak depending on results
//Not that the accuracy of results also depend on the step size
pub enum DiffMode
{
    ForwardFixedStep,
    BackwardFixedStep,
    CentralFixedStep
}