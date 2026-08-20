use multicalc::error::LinalgError;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(_errors, LinalgErrorException, PyException);

pub(crate) fn register<'py>(module: &Bound<'py, PyModule>) -> PyResult<()> {
    module.add(
        "LinalgError",
        module.py().get_type::<LinalgErrorException>(),
    )?;
    Ok(())
}

pub(crate) fn linalg_error(exc: LinalgError) -> PyErr {
    LinalgErrorException::new_err(linalg_error_kind(exc))
}

fn linalg_error_kind(error: LinalgError) -> String {
    match error {
        LinalgError::Singular => "Singular".into(),
        LinalgError::IllConditioned => "IllConditioned".into(),
        LinalgError::NotPositiveDefinite => "NotPositiveDefinite".into(),
        LinalgError::Underdetermined => "Underdetermined".into(),
        LinalgError::NonFinite => "NonFinite".into(),
        LinalgError::NotSymmetric => "NotSymmetric".into(),
        LinalgError::InvalidTimestep => "InvalidTimestep".into(),
        LinalgError::DidNotConverge { iters: _ } => "DidNotConverge".into(),
        other => format!("{other:?}"),
    }
}
