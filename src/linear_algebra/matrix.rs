//! Fixed-size, stack-allocated matrix.

/// A `ROWS`×`COLS` matrix stored inline on the stack in row-major order.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct Matrix<const ROWS: usize, const COLS: usize, T = f64> {
    data: [[T; COLS]; ROWS],
}
