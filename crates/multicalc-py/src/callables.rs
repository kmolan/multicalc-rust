use multicalc::scalar::{Numeric, ScalarFn, ScalarFnN};
use pyo3::prelude::*;

use std::cell::RefCell;

#[allow(clippy::min_ident_chars)]
mod shared {
    pub(super) use std::rc::Rc as ReferenceCounter;
}
use shared::ReferenceCounter;

use multicalc::linear_algebra::Vector;

/// First Python callback failure during a multicalc call; re-raised at the pyfunction boundary.
#[derive(Clone, Default)]
pub(crate) struct CallbackError(ReferenceCounter<RefCell<Option<PyErr>>>);

impl CallbackError {
    fn stash(&self, err: PyErr) {
        let mut slot = self.0.borrow_mut();
        if slot.is_none() {
            *slot = Some(err);
        }
    }

    pub(crate) fn take(&self) -> Option<PyErr> {
        self.0.borrow_mut().take()
    }

    pub(crate) fn finish<T>(&self, result: PyResult<T>) -> PyResult<T> {
        if let Some(err) = self.take() {
            return Err(err);
        }
        result
    }
}

fn stash_scalar_err(error: &CallbackError, err: PyErr) -> f64 {
    error.stash(err);
    f64::NAN
}

fn stash_vector_err<S: Numeric>(_error: &CallbackError) -> [S; 2] {
    [S::from_f64(f64::NAN), S::from_f64(f64::NAN)]
}

/// Reinterpret a `Numeric` scalar as `f64` for a Python callback.
///
/// Python only speaks `f64`. The finite-difference and explicit-`f64` paths in this crate
/// evaluate `ScalarFn` / `VectorFn` at `f64`. Autodiff types (`Dual`, `HyperDual`, `Jet`)
/// are wider than eight bytes and must not be passed through here.
fn read_as_f64<S: Numeric>(scalar: &S) -> f64 {
    debug_assert_eq!(core::mem::size_of::<S>(), core::mem::size_of::<f64>());
    // SAFETY: the scalar is `Copy`. These `eval` impls are only reached with `f64`, so the
    // object is a valid `f64` and the sizes match. The debug_assert above fails if an
    // autodiff scalar (larger than `f64`) is ever used.
    unsafe { core::ptr::read((scalar as *const S).cast::<f64>()) }
}

pub(crate) struct PythonScalarFn {
    callback: Py<PyAny>,
    error: CallbackError,
}

impl PythonScalarFn {
    pub(crate) fn new(callback: Py<PyAny>) -> Self {
        Self {
            callback,
            error: CallbackError::default(),
        }
    }

    pub(crate) fn finish<T>(&self, result: PyResult<T>) -> PyResult<T> {
        self.error.finish(result)
    }
}

impl ScalarFn for PythonScalarFn {
    fn eval<S: Numeric>(&self, argument: S) -> S {
        let value = read_as_f64(&argument);
        let result = Python::attach(|python| {
            self.callback
                .bind(python)
                .call1((value,))
                .and_then(|object| object.extract::<f64>())
        });
        match result {
            Ok(output) => S::from_f64(output),
            Err(err) => S::from_f64(stash_scalar_err(&self.error, err)),
        }
    }
}

pub(crate) struct PythonScalarFn2 {
    callback: Py<PyAny>,
    error: CallbackError,
}

impl PythonScalarFn2 {
    pub(crate) fn new(callback: Py<PyAny>) -> Self {
        Self {
            callback,
            error: CallbackError::default(),
        }
    }

    pub(crate) fn finish<T>(&self, result: PyResult<T>) -> PyResult<T> {
        self.error.finish(result)
    }
}

impl ScalarFnN<2> for PythonScalarFn2 {
    fn eval<S: Numeric>(&self, point: &[S; 2]) -> S {
        let first = read_as_f64(&point[0]);
        let second = read_as_f64(&point[1]);
        let result = Python::attach(|python| {
            self.callback
                .bind(python)
                .call1((first, second))
                .and_then(|object| object.extract::<f64>())
        });
        match result {
            Ok(output) => S::from_f64(output),
            Err(err) => S::from_f64(stash_scalar_err(&self.error, err)),
        }
    }
}

pub(crate) fn call_ode2(callback: &Py<PyAny>, time: f64, state: [f64; 2]) -> PyResult<[f64; 2]> {
    Python::attach(|python| {
        let output: Vec<f64> = callback
            .bind(python)
            .call1((time, state.to_vec()))?
            .extract()?;
        if output.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "ode callback must return 2 values",
            ));
        }
        Ok([output[0], output[1]])
    })
}

pub(crate) struct PythonOdeRate2 {
    callback: Py<PyAny>,
    error: CallbackError,
}

impl PythonOdeRate2 {
    pub(crate) fn new(callback: Py<PyAny>) -> Self {
        Self {
            callback,
            error: CallbackError::default(),
        }
    }

    pub(crate) fn rate(&self) -> impl Fn(f64, &Vector<2>) -> Vector<2> + '_ {
        move |clock, current| match call_ode2(&self.callback, clock, current.into_array()) {
            Ok(output) => Vector::new(output),
            Err(err) => {
                self.error.stash(err);
                Vector::new(stash_vector_err::<f64>(&self.error))
            }
        }
    }

    pub(crate) fn finish<T>(&self, result: PyResult<T>) -> PyResult<T> {
        self.error.finish(result)
    }
}

pub(crate) struct PythonVectorFn2x2 {
    callback: Py<PyAny>,
    error: CallbackError,
}

impl PythonVectorFn2x2 {
    pub(crate) fn new(callback: Py<PyAny>) -> Self {
        Self {
            callback,
            error: CallbackError::default(),
        }
    }

    pub(crate) fn finish<T>(&self, result: PyResult<T>) -> PyResult<T> {
        self.error.finish(result)
    }
}

impl multicalc::scalar::VectorFn<2, 2> for PythonVectorFn2x2 {
    fn eval<S: Numeric>(&self, point: &[S; 2]) -> [S; 2] {
        let first = read_as_f64(&point[0]);
        let second = read_as_f64(&point[1]);
        let result = Python::attach(|python| {
            self.callback
                .bind(python)
                .call1((first, second))
                .and_then(|object| object.extract::<[f64; 2]>())
        });
        match result {
            Ok(output) if output.len() == 2 => [S::from_f64(output[0]), S::from_f64(output[1])],
            Ok(_) => {
                self.error.stash(pyo3::exceptions::PyValueError::new_err(
                    "vector callback must return 2 values",
                ));
                stash_vector_err(&self.error)
            }
            Err(err) => {
                self.error.stash(err);
                stash_vector_err(&self.error)
            }
        }
    }
}
