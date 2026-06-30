//! Fixed-size, stack-allocated column vector.

/// A column vector of `N` components, stored inline on the stack.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct Vector<const N: usize, T = f64> {
    data: [T; N],
}
