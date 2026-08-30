//! Walking one flat parameter buffer, handing out a layer at a time.

use crate::error::LinalgError;
use crate::linear_algebra::{MatrixView, VectorView};
use crate::mlp_inference::{Activation, Layer};

/// A read position in a flat parameter buffer, handing out views of successive runs of it.
///
/// A trained network is exported as one block of numbers: each layer's weights row-major, then its
/// biases, in the order the layers run. Slicing that block by hand means writing the offsets out —
/// `&block[1408..1472]` — where a single wrong number produces a network that loads, runs, and
/// answers wrongly. The cursor computes them instead, from the shapes the caller declares.
///
/// Nothing is copied. Each call hands back a view of the same buffer at the next offset and steps
/// past it, so the parameters stay wherever they were stored.
///
/// ```
/// use multicalc::linear_algebra::Vector;
/// use multicalc::mlp_inference::{Activation, ParameterCursor};
///
/// // 2 -> 3 -> 1: hidden weights, hidden biases, output weights, output bias.
/// let parameters = [
///     0.5, -0.5, 1.0, 0.0, -1.0, 2.0, //
///     0.0, 1.0, -1.0, //
///     1.0, 1.0, 1.0, //
///     0.5,
/// ];
/// let mut cursor = ParameterCursor::new(&parameters);
/// let hidden = cursor.try_take_layer::<3, 2>(Activation::Relu)?;
/// let output = cursor.try_take_layer::<1, 3>(Activation::Identity)?;
/// assert!(cursor.is_empty());
///
/// let observation = Vector::new([2.0, 1.0]);
/// let action = output.forward(hidden.forward(observation.view()).view());
/// assert_eq!(action.into_array(), [4.0]);
/// # Ok::<(), multicalc::error::LinalgError>(())
///```
#[derive(Debug)]
#[must_use]
pub struct ParameterCursor<'data, T> {
    remaining: &'data [T],
}

impl<'data, T> Clone for ParameterCursor<'data, T> {
    #[inline]
    fn clone(&self) -> Self {
        ParameterCursor {
            remaining: self.remaining,
        }
    }
}

impl<'data, T> ParameterCursor<'data, T> {
    /// A cursor at the start of `parameters`.
    ///
    /// ```
    /// use multicalc::mlp_inference::ParameterCursor;
    /// let parameters = [1.0, 2.0, 3.0];
    /// assert_eq!(ParameterCursor::new(&parameters).remaining(), 3);
    /// ```
    #[inline]
    pub const fn new(parameters: &'data [T]) -> Self {
        ParameterCursor {
            remaining: parameters,
        }
    }

    /// How many values are still ahead of the read position.
    ///
    /// ```
    /// use multicalc::mlp_inference::ParameterCursor;
    /// let parameters = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let mut cursor = ParameterCursor::new(&parameters);
    /// cursor.try_take_vector::<2>()?;
    /// assert_eq!(cursor.remaining(), 3);
    /// # Ok::<(), multicalc::error::LinalgError>(())
    /// ```
    #[inline]
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.remaining.len()
    }

    /// Whether the buffer has been read to its end.
    ///
    /// Worth checking once a network is loaded: a buffer with values left over is one whose shape
    /// disagrees with the shapes just declared, which otherwise goes unnoticed because every
    /// individual read succeeded.
    ///
    /// ```
    /// use multicalc::mlp_inference::ParameterCursor;
    /// let parameters = [1.0, 2.0, 3.0];
    /// let mut cursor = ParameterCursor::new(&parameters);
    /// assert!(!cursor.is_empty());
    /// cursor.try_take_vector::<3>()?;
    /// assert!(cursor.is_empty());
    /// # Ok::<(), multicalc::error::LinalgError>(())
    /// ```
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.remaining.is_empty()
    }

    /// The next `ROWS`×`COLS` values as a row-major matrix, or `OutOfBounds` if fewer than that
    /// many are left.
    ///
    /// ```
    /// use multicalc::mlp_inference::ParameterCursor;
    /// let parameters = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let mut cursor = ParameterCursor::new(&parameters);
    /// let weights = cursor.try_take_matrix::<2, 3>()?;
    /// assert_eq!(weights.try_get(1, 0), Ok(&4.0));
    /// assert!(cursor.try_take_matrix::<2, 3>().is_err());
    /// # Ok::<(), multicalc::error::LinalgError>(())
    /// ```
    #[inline]
    pub fn try_take_matrix<const ROWS: usize, const COLS: usize>(
        &mut self,
    ) -> Result<MatrixView<'data, ROWS, COLS, T>, LinalgError> {
        let taken = self.try_split(ROWS.checked_mul(COLS).ok_or(LinalgError::OutOfBounds)?)?;
        MatrixView::try_from_row_major_slice(taken)
    }

    /// The next `N` values as a vector, or `OutOfBounds` if fewer than that many are left.
    ///
    /// ```
    /// use multicalc::mlp_inference::ParameterCursor;
    /// let parameters = [1.0, 2.0, 3.0];
    /// let mut cursor = ParameterCursor::new(&parameters);
    /// let biases = cursor.try_take_vector::<2>()?;
    /// assert_eq!(biases.try_get(1), Ok(&2.0));
    /// assert_eq!(cursor.remaining(), 1);
    /// # Ok::<(), multicalc::error::LinalgError>(())
    /// ```
    #[inline]
    pub fn try_take_vector<const N: usize>(
        &mut self,
    ) -> Result<VectorView<'data, N, T>, LinalgError> {
        let taken = self.try_split(N)?;
        VectorView::try_from_slice(taken)
    }

    /// One layer: `OUTPUT`×`INPUT` weights row-major, then `OUTPUT` biases, which is the order a
    /// trained network is normally written in.
    ///
    /// The values are taken as they come. This builds through [`Layer::new`] rather than
    /// [`Layer::try_from_slices`], so unlike that constructor it does not read the parameters, and
    /// a non-finite one passes through here unremarked.
    ///
    /// ```
    /// use multicalc::mlp_inference::{Activation, ParameterCursor};
    /// let parameters = [1.0, 0.0, 0.0, 1.0, 0.5, -0.5];
    /// let mut cursor = ParameterCursor::new(&parameters);
    /// let layer = cursor.try_take_layer::<2, 2>(Activation::Relu)?;
    /// assert!(cursor.is_empty());
    /// # Ok::<(), multicalc::error::LinalgError>(())
    /// ```
    #[inline]
    pub fn try_take_layer<const OUTPUT: usize, const INPUT: usize>(
        &mut self,
        activation: Activation,
    ) -> Result<Layer<'data, OUTPUT, INPUT, T>, LinalgError> {
        // Read into a clone so a layer whose biases run off the end leaves the position untouched,
        // rather than half past a layer that was never handed out.
        let mut lookahead = self.clone();
        let weights = lookahead.try_take_matrix::<OUTPUT, INPUT>()?;
        let biases = lookahead.try_take_vector::<OUTPUT>()?;
        *self = lookahead;
        Ok(Layer::new(weights, biases, activation))
    }

    /// Splits `count` values off the front, or `OutOfBounds` if there are fewer than that.
    #[inline]
    fn try_split(&mut self, count: usize) -> Result<&'data [T], LinalgError> {
        let (taken, rest) = self
            .remaining
            .split_at_checked(count)
            .ok_or(LinalgError::OutOfBounds)?;
        self.remaining = rest;
        Ok(taken)
    }
}
