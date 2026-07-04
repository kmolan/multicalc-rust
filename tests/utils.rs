use core::fmt::{self, Write};
use multicalc::utils::error_codes::CalcError;

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

fn renders_as(error: CalcError, expected: &str) -> bool {
    let mut probe = Probe {
        expected,
        matched: false,
    };
    write!(probe, "{error}").unwrap();
    probe.matched
}

#[test]
fn display_strings() {
    assert!(renders_as(
        CalcError::Underdetermined,
        "system is underdetermined (M < N)"
    ));
    assert!(renders_as(
        CalcError::SingularMatrix,
        "matrix is singular or rank-deficient"
    ));
    assert!(renders_as(
        CalcError::NotPositiveDefinite,
        "matrix is not positive definite"
    ));
    assert!(renders_as(
        CalcError::DidNotConverge,
        "solver did not converge within the iteration/evaluation budget"
    ));
    assert!(renders_as(
        CalcError::NonFiniteValue,
        "residual or Jacobian contained a non-finite value"
    ));
}
