use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(_errors, CalcErrorException, PyException);
create_exception!(_errors, LinalgErrorException, PyException);
create_exception!(_errors, DiffErrorException, PyException);
create_exception!(_errors, IntegrateErrorException, PyException);
create_exception!(_errors, SolveErrorException, PyException);
create_exception!(_errors, KinematicsErrorException, PyException);
create_exception!(_errors, SpatialErrorException, PyException);
create_exception!(_errors, EstimateErrorException, PyException);
create_exception!(_errors, SignalErrorException, PyException);
create_exception!(_errors, ControlErrorException, PyException);
create_exception!(_errors, DynamicsErrorException, PyException);
create_exception!(_errors, PlantErrorException, PyException);
create_exception!(_errors, MappingErrorException, PyException);
create_exception!(_errors, MotionErrorException, PyException);
create_exception!(_errors, PolynomialErrorException, PyException);

pub(crate) fn register<'python>(module: &Bound<'python, PyModule>) -> PyResult<()> {
    let python = module.py();
    module.add("CalcError", python.get_type::<CalcErrorException>())?;
    module.add("LinalgError", python.get_type::<LinalgErrorException>())?;
    module.add("DiffError", python.get_type::<DiffErrorException>())?;
    module.add(
        "IntegrateError",
        python.get_type::<IntegrateErrorException>(),
    )?;
    module.add("SolveError", python.get_type::<SolveErrorException>())?;
    module.add(
        "KinematicsError",
        python.get_type::<KinematicsErrorException>(),
    )?;
    module.add("SpatialError", python.get_type::<SpatialErrorException>())?;
    module.add("EstimateError", python.get_type::<EstimateErrorException>())?;
    module.add("SignalError", python.get_type::<SignalErrorException>())?;
    module.add("ControlError", python.get_type::<ControlErrorException>())?;
    module.add("DynamicsError", python.get_type::<DynamicsErrorException>())?;
    module.add("PlantError", python.get_type::<PlantErrorException>())?;
    module.add("MappingError", python.get_type::<MappingErrorException>())?;
    module.add("MotionError", python.get_type::<MotionErrorException>())?;
    module.add(
        "PolynomialError",
        python.get_type::<PolynomialErrorException>(),
    )?;
    Ok(())
}

pub(crate) fn linalg_error(error: multicalc::error::LinalgError) -> PyErr {
    LinalgErrorException::new_err(format!("{error:?}"))
}

pub(crate) fn spatial_error(error: multicalc::error::SpatialError) -> PyErr {
    SpatialErrorException::new_err(format!("{error:?}"))
}

pub(crate) fn control_error(error: multicalc::error::ControlError) -> PyErr {
    ControlErrorException::new_err(format!("{error:?}"))
}

pub(crate) fn diff_error(error: multicalc::error::DiffError) -> PyErr {
    DiffErrorException::new_err(format!("{error:?}"))
}

pub(crate) fn integrate_error(error: multicalc::error::IntegrateError) -> PyErr {
    IntegrateErrorException::new_err(format!("{error:?}"))
}

pub(crate) fn solve_error(error: multicalc::error::SolveError) -> PyErr {
    SolveErrorException::new_err(format!("{error:?}"))
}

pub(crate) fn polynomial_error(error: multicalc::error::PolynomialError) -> PyErr {
    PolynomialErrorException::new_err(format!("{error:?}"))
}

pub(crate) fn estimation_error(error: multicalc::error::EstimationError) -> PyErr {
    EstimateErrorException::new_err(format!("{error:?}"))
}

pub(crate) fn signal_error(error: multicalc::error::SignalError) -> PyErr {
    SignalErrorException::new_err(format!("{error:?}"))
}

pub(crate) fn kinematics_error(error: multicalc::error::KinematicsError) -> PyErr {
    KinematicsErrorException::new_err(format!("{error:?}"))
}

pub(crate) fn motion_error(error: multicalc::error::MotionError) -> PyErr {
    MotionErrorException::new_err(format!("{error:?}"))
}

pub(crate) fn dynamics_error(error: multicalc::error::DynamicsError) -> PyErr {
    DynamicsErrorException::new_err(format!("{error:?}"))
}

pub(crate) fn plant_error(error: multicalc::error::PlantError) -> PyErr {
    PlantErrorException::new_err(format!("{error:?}"))
}

pub(crate) fn mapping_error(error: multicalc::error::MappingError) -> PyErr {
    MappingErrorException::new_err(format!("{error:?}"))
}
