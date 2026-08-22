//! Carving one caller-supplied scratch buffer into disjoint writable views.
//!
//! Distinct from the `split_*` methods on the views themselves, which this is easy to confuse
//! with. `MatrixViewMut::split_rows_at` cuts one existing view into exactly two pieces whose
//! shapes are already fixed: one cut, two results, finished. A `Workspace` instead claims pieces
//! from a raw buffer one after another, as many as the caller asks for and in whatever shapes it
//! asks for, until the buffer runs out. Splitting divides something that already has a shape;
//! the workspace hands shape to memory that has none yet.

use super::{MatrixViewMut, VectorViewMut};

/// Carves one caller-supplied scratch buffer into disjoint writable views.
///
/// An algorithm that needs temporaries can take them from a buffer the caller owns instead of
/// sizing a stack array itself, which matters when the caller knows what its stack budget is and
/// the algorithm does not. Each `take` hands back a view borrowing a distinct piece of the
/// buffer, so several of them are usable at once; that disjointness is
/// `slice::split_at_mut`'s guarantee, not a hand-checked invariant.
///
/// The name is the one numerical linear algebra already uses: LAPACK routines take a `WORK`
/// array from the caller and publish how big it has to be, so the library never decides on the
/// caller's behalf how much memory a factorization is allowed to occupy. This is that idea with
/// the sizes checked rather than documented — a claim that does not fit returns `None` instead
/// of overrunning, and claiming nothing at all is what `remaining` and `into_remainder` report.
///
/// ```
/// use multicalc::linear_algebra::{Matrix, Workspace};
///
/// let mut scratch = [0.0; 16];
/// let mut workspace = Workspace::new(&mut scratch);
///
/// let mut left = workspace.take_matrix::<2, 3>().unwrap();
/// let mut right = workspace.take_matrix::<3, 2>().unwrap();
/// let mut residual = workspace.take_vector::<4>().unwrap();
///
/// // All three are live at the same time, over disjoint memory.
/// left.copy_from(Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).view());
/// right.copy_from(left.as_view().transposed());
/// residual.fill(1.0);
///
/// assert_eq!(right.to_matrix().into_array(), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
/// assert_eq!(workspace.remaining(), 0);
/// ```
#[derive(Debug)]
#[must_use]
pub struct Workspace<'a, T> {
    rest: &'a mut [T],
}

impl<'a, T> Workspace<'a, T> {
    /// Wraps a scratch buffer. Nothing is read from it; each `take` simply claims a span.
    #[inline]
    pub fn new(buffer: &'a mut [T]) -> Self {
        Workspace { rest: buffer }
    }

    /// How many elements are still unclaimed.
    #[inline]
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.rest.len()
    }

    /// Claims `len` elements as a plain slice, or returns `None` if too few are left. The
    /// shared primitive the two shaped claims are built on.
    #[inline]
    fn take_slice(&mut self, len: usize) -> Option<&'a mut [T]> {
        (len <= self.rest.len()).then_some(())?;
        // Moving the borrow out and putting the tail back is what lets the claimed piece carry
        // the full `'a` lifetime, so callers can hold several claims at once.
        let all = core::mem::take(&mut self.rest);
        let (claimed, rest) = all.split_at_mut(len);
        self.rest = rest;
        Some(claimed)
    }

    /// Claims a row-major `ROWS`×`COLS` matrix, or returns `None` if too few elements are left.
    #[inline]
    pub fn take_matrix<const ROWS: usize, const COLS: usize>(
        &mut self,
    ) -> Option<MatrixViewMut<'a, ROWS, COLS, T>> {
        let len = ROWS.checked_mul(COLS)?;
        let claimed = self.take_slice(len)?;
        MatrixViewMut::from_parts(claimed, 0, COLS, 1)
    }

    /// Claims `N` contiguous components, or returns `None` if too few elements are left.
    #[inline]
    pub fn take_vector<const N: usize>(&mut self) -> Option<VectorViewMut<'a, N, T>> {
        let claimed = self.take_slice(N)?;
        VectorViewMut::from_parts(claimed, 0, 1)
    }

    /// Gives back whatever is still unclaimed, ending the workspace.
    #[inline]
    #[must_use]
    pub fn into_remainder(self) -> &'a mut [T] {
        self.rest
    }
}
