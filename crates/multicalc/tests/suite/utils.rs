#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use core::fmt::{self, Write};
use multicalc::error::{CalcError, DiffError, IntegrateError, LinalgError, SolveError};

// Captures the single `write_str` call `Display` makes, so no allocator is needed.
struct Probe<'a> {
    expected: &'a str,
    matched: bool,
}

impl Write for Probe<'_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.matched = s == self.expected;
        Ok(())
    }
}

fn renders_as<E: fmt::Display>(error: E, expected: &str) -> bool {
    let mut probe = Probe {
        expected,
        matched: false,
    };
    write!(probe, "{error}").unwrap();
    probe.matched
}

#[test]
fn linalg_display_strings() {
    assert!(renders_as(
        LinalgError::Singular,
        "matrix is singular or rank-deficient"
    ));
    assert!(renders_as(
        LinalgError::NotPositiveDefinite,
        "matrix is not positive definite"
    ));
    assert!(renders_as(
        LinalgError::Underdetermined,
        "system is underdetermined (M < N)"
    ));
    assert!(renders_as(
        LinalgError::NonFinite,
        "matrix contained a non-finite value"
    ));
}

#[test]
fn diff_display_strings() {
    assert!(renders_as(
        DiffError::OrderZero,
        "derivative order cannot be zero"
    ));
    assert!(renders_as(
        DiffError::OrderUnsupported,
        "derivative order is not supported"
    ));
    assert!(renders_as(
        DiffError::StepSizeZero,
        "step size cannot be zero"
    ));
    assert!(renders_as(
        DiffError::IndexOutOfRange,
        "variable index out of range"
    ));
    assert!(renders_as(
        DiffError::EmptyFunctionSet,
        "function set cannot be empty"
    ));
}

#[test]
fn integrate_display_strings() {
    assert!(renders_as(
        IntegrateError::IterationsZero,
        "number of iterations cannot be zero"
    ));
    assert!(renders_as(
        IntegrateError::LimitsIllDefined,
        "lower limit must be strictly less than upper limit"
    ));
    assert!(renders_as(
        IntegrateError::QuadratureOrderOutOfRange,
        "quadrature order is out of supported range"
    ));
    assert!(renders_as(
        IntegrateError::StepSizeTooSmall,
        "adaptive step size fell below the minimum"
    ));
    assert!(renders_as(
        IntegrateError::NonFinite,
        "integrand or state contained a non-finite value"
    ));
    assert!(renders_as(
        IntegrateError::IndexOutOfRange,
        "variable index out of range"
    ));
}

#[test]
fn solve_display_strings() {
    assert!(renders_as(
        SolveError::NonFinite,
        "residual or Jacobian contained a non-finite value"
    ));
    assert!(renders_as(
        SolveError::InvalidBracket,
        "bracket endpoints must enclose a sign change"
    ));
}

// The umbrella and the solver wrappers forward to the wrapped family error's message.
#[test]
fn wrapped_errors_forward() {
    assert!(renders_as(
        CalcError::Linalg(LinalgError::Singular),
        "matrix is singular or rank-deficient"
    ));
    assert!(renders_as(
        SolveError::Linalg(LinalgError::Singular),
        "matrix is singular or rank-deficient"
    ));
    assert!(renders_as(
        SolveError::Diff(DiffError::OrderZero),
        "derivative order cannot be zero"
    ));
}

// The data-carrying variants interpolate their payload into the message.
#[test]
fn data_carrying_display() {
    assert_eq!(
        SolveError::DidNotConverge {
            iters: 7,
            residual: 0.5,
        }
        .to_string(),
        "solver did not converge after 7 iterations (residual 0.5)"
    );
    assert_eq!(
        IntegrateError::DidNotConverge { steps: 12 }.to_string(),
        "integrator did not converge within 12 steps"
    );
}
