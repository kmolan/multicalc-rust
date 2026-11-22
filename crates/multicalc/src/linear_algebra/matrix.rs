//! Fixed-size, stack-allocated matrix.

use core::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Neg, Sub, SubAssign};

use crate::error::LinalgError;
use crate::linear_algebra::Vector;
use crate::scalar::Numeric;

/// A `ROWS`×`COLS` matrix stored inline on the stack in row-major order.
///
/// ```
/// use multicalc::linear_algebra::{Matrix, Vector};
/// let a = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
/// let b = Matrix::new([[5.0, 6.0], [7.0, 8.0]]);
///
/// assert_eq!(a[(0, 1)], 2.0);
/// assert_eq!(a.get(0, 1), Some(&2.0));
/// assert_eq!((a + b).into_array(), [[6.0, 8.0], [10.0, 12.0]]);
/// assert_eq!((b - a).into_array(), [[4.0, 4.0], [4.0, 4.0]]);
/// assert_eq!((-a).into_array(), [[-1.0, -2.0], [-3.0, -4.0]]);
/// assert_eq!((a * 2.0).into_array(), [[2.0, 4.0], [6.0, 8.0]]);
/// assert_eq!((a * b).into_array(), [[19.0, 22.0], [43.0, 50.0]]);
/// assert_eq!(a * Vector::new([1.0, 1.0]), Vector::new([3.0, 7.0]));
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct Matrix<const ROWS: usize, const COLS: usize, T = f64> {
    data: [[T; COLS]; ROWS],
}

impl<const ROWS: usize, const COLS: usize, T> Matrix<ROWS, COLS, T> {
    /// Wraps a row-major array of rows into a matrix.
    #[inline]
    pub const fn new(data: [[T; COLS]; ROWS]) -> Self {
        Matrix { data }
    }

    /// Builds a matrix by calling `f` with each `(row, column)` index.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix = Matrix::<2, 2>::from_fn(|row, column| (row * 2 + column) as f64);
    /// assert_eq!(matrix.into_array(), [[0.0, 1.0], [2.0, 3.0]]);
    /// ```
    #[inline]
    pub fn from_fn(mut f: impl FnMut(usize, usize) -> T) -> Self {
        Matrix {
            data: core::array::from_fn(|row| core::array::from_fn(|column| f(row, column))),
        }
    }

    // Crate-internal panic path (also used by Index). Public: prefer `[]`; use `get` when fallible.
    #[inline]
    #[track_caller]
    #[must_use]
    pub(crate) fn get_unchecked(&self, row: usize, column: usize) -> &T {
        #[allow(clippy::indexing_slicing)]
        &self.data[row][column]
    }

    #[inline]
    #[track_caller]
    pub(crate) fn get_unchecked_mut(&mut self, row: usize, column: usize) -> &mut T {
        #[allow(clippy::indexing_slicing)]
        &mut self.data[row][column]
    }

    /// Borrows the rows.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// assert_eq!(Matrix::new([[1.0, 2.0]]).as_slice_rows(), &[[1.0, 2.0]]);
    /// ```
    #[inline]
    #[must_use]
    pub const fn as_slice_rows(&self) -> &[[T; COLS]; ROWS] {
        &self.data
    }

    /// Borrows the rows mutably.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let mut matrix = Matrix::new([[1.0, 2.0]]);
    /// matrix.as_mut_slice_rows()[0][1] = 9.0;
    /// assert_eq!(matrix[(0, 1)], 9.0);
    /// ```
    #[inline]
    pub fn as_mut_slice_rows(&mut self) -> &mut [[T; COLS]; ROWS] {
        &mut self.data
    }

    /// Returns a reference to entry `(row, column)`, or `None` if out of range.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// assert_eq!(matrix.get(1, 0), Some(&3.0));
    /// assert_eq!(matrix.get(2, 0), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn get(&self, row: usize, column: usize) -> Option<&T> {
        self.data.get(row).and_then(|row_data| row_data.get(column))
    }

    /// Returns a mutable reference to entry `(row, column)`, or `None` if out of range.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let mut matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// if let Some(x) = matrix.get_mut(0, 1) {
    ///     *x = 7.0;
    /// }
    /// assert_eq!(matrix.get(0, 1), Some(&7.0));
    /// ```
    #[inline]
    pub fn get_mut(&mut self, row: usize, column: usize) -> Option<&mut T> {
        self.data
            .get_mut(row)
            .and_then(|row_data| row_data.get_mut(column))
    }

    /// Consumes the matrix, returning its rows.
    #[inline]
    #[must_use]
    pub fn into_array(self) -> [[T; COLS]; ROWS] {
        self.data
    }
}

impl<const ROWS: usize, const COLS: usize, T: Copy> Matrix<ROWS, COLS, T> {
    /// Builds a matrix from a row-major slice, or `None` if `slice.len()` is not `ROWS * COLS`.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// assert!(Matrix::<2, 2>::try_from_row_slice(&[1.0, 2.0, 3.0, 4.0]).is_some());
    /// assert!(Matrix::<2, 2>::try_from_row_slice(&[1.0, 2.0, 3.0]).is_none());
    /// ```
    #[inline]
    #[must_use]
    pub fn try_from_row_slice(slice: &[T]) -> Option<Self> {
        // In-bounds by construction: `row < ROWS`, `column < COLS`, and the length was just checked.
        #[allow(clippy::indexing_slicing)]
        (slice.len() == ROWS * COLS)
            .then(|| Self::from_fn(|row, column| slice[row * COLS + column]))
    }

    /// Copies row `row`, or `None` if `row >= ROWS`.
    ///
    /// ```
    /// use multicalc::linear_algebra::{Matrix, Vector};
    /// let matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// assert_eq!(matrix.try_row(1), Some(Vector::new([3.0, 4.0])));
    /// assert_eq!(matrix.try_row(2), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn try_row(&self, row: usize) -> Option<Vector<COLS, T>> {
        self.data.get(row).copied().map(Vector::new)
    }

    /// Copies column `column`, or `None` if `column >= COLS`.
    ///
    /// ```
    /// use multicalc::linear_algebra::{Matrix, Vector};
    /// let matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// assert_eq!(matrix.try_column(1), Some(Vector::new([2.0, 4.0])));
    /// assert_eq!(matrix.try_column(2), None);
    /// let empty: Matrix<0, 3> = Matrix::zeros();
    /// assert_eq!(empty.try_column(0), Some(Vector::<0>::zeros()));
    /// assert_eq!(empty.try_column(3), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn try_column(&self, column: usize) -> Option<Vector<ROWS, T>> {
        (column < COLS).then(|| Vector::from_fn(|row| self.data[row][column]))
    }
}

impl<const ROWS: usize, const COLS: usize, T: Numeric> Matrix<ROWS, COLS, T> {
    /// The zero matrix.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix: Matrix<2, 3> = Matrix::zeros();
    /// assert_eq!(matrix.into_array(), [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    /// ```
    #[inline]
    pub fn zeros() -> Self {
        Matrix::from_fn(|_, _| T::ZERO)
    }

    /// Multiplies every element by `scalar`.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix = Matrix::new([[1.0, 2.0]]);
    /// let factor = 3.0;
    /// assert_eq!(matrix.scale(factor).into_array(), [[3.0, 6.0]]);
    /// ```
    #[inline]
    pub fn scale(self, scalar: T) -> Self {
        Matrix::from_fn(|row, column| self[(row, column)] * scalar)
    }

    /// The transpose, with rows and columns swapped.
    ///
    /// This copies. [`Matrix::view`] then [`transposed`](crate::linear_algebra::MatrixView::transposed)
    /// is the same reshaping with no copy, for when the result only needs reading.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix = Matrix::new([[1.0, 2.0, 3.0]]);
    /// assert_eq!(matrix.transpose().into_array(), [[1.0], [2.0], [3.0]]);
    /// ```
    #[inline]
    pub fn transpose(self) -> Matrix<COLS, ROWS, T> {
        self.view().transposed().to_matrix()
    }

    /// The Frobenius norm, sometimes called the Euclidean norm:
    /// the square root of the sum of the absolute squares of the elements.
    ///
    /// Note: this method computes the sum of the entries in row-major order from top left
    /// to bottom right, which could have an impact on the accuracy of the result
    /// in the case of floating-point types if the earlier elements are significantly
    /// larger than the later ones.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix = Matrix::new([[1.0, -2.0, 0.0], [3.0, 0.0, 4.0], [2.0, -1.0, 1.0]]);
    /// assert_eq!(matrix.frobenius_norm(), 6.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn frobenius_norm(self) -> T {
        let total = self
            .data
            .into_iter()
            .flatten()
            .fold(T::ZERO, |acc, x| acc + x * x);
        total.sqrt()
    }

    /// Returns `true` when every entry is neither infinite nor NaN.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// assert!(Matrix::new([[1.0, -2.0], [3.0, 4.0]]).is_finite());
    /// assert!(!Matrix::new([[1.0, f64::NAN], [3.0, 4.0]]).is_finite());
    /// ```
    #[inline]
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.data.iter().flatten().all(|x| x.is_finite())
    }

    /// Largest absolute entry; used to scale near-singularity checks.
    #[inline]
    #[must_use]
    pub(super) fn max_abs(self) -> T {
        let mut best = T::ZERO;
        for row in &self.data {
            for x in row {
                best = best.max(x.abs());
            }
        }
        best
    }

    /// `true` when `|det|` is at or below `EPSILON * n * scale^n`.
    #[inline]
    #[must_use]
    fn det_near_singular(det: T, scale: T, n: usize) -> bool {
        det.abs() <= T::EPSILON * T::from_usize(n) * scale.powi(n as i32)
    }
}

impl<const N: usize, T: Numeric> Matrix<N, N, T> {
    /// The `N`×`N` diagonal matrix with the given diagonal entries
    /// (all off-diagonal elements are equal to zero).
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix = Matrix::from_diagonal([1.0, 2.0, 3.0]);
    /// assert_eq!(matrix.into_array(), [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]);
    /// ```
    #[inline]
    pub fn from_diagonal(diag: [T; N]) -> Self {
        let rows = core::array::from_fn(|i| {
            let mut row = [T::ZERO; N];
            row[i] = diag[i];
            row
        });
        Matrix::new(rows)
    }

    /// Returns the diagonal entries of the matrix as an array.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// assert_eq!(matrix.diagonal(), [1.0, 4.0]);
    /// ```
    #[inline]
    #[must_use]
    pub fn diagonal(&self) -> [T; N] {
        core::array::from_fn(|i| self[(i, i)])
    }

    /// True when the matrix reads the same across the diagonal.
    ///
    /// Each pair only has to agree to within rounding, not exactly, so a matrix that has drifted a
    /// little is still accepted while a genuinely lopsided one is not. How much a pair is allowed
    /// to differ grows with how big the entries are, because bigger numbers are stored more
    /// coarsely; the floor at one stops the allowance from shrinking to exact equality for entries
    /// near zero.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// assert!(Matrix::new([[1.0, 2.0], [2.0, 3.0]]).is_symmetric());
    /// assert!(!Matrix::new([[1.0, 2.0], [-2.0, 3.0]]).is_symmetric());
    ///
    /// // A pair that has drifted by a couple of rounding steps still counts as matching.
    /// let drifted = Matrix::new([[1.0, 2.0], [2.0 + 4.0 * f64::EPSILON, 3.0]]);
    /// assert!(drifted.is_symmetric());
    /// ```
    #[inline]
    #[must_use]
    pub fn is_symmetric(self) -> bool {
        for row in 0..N {
            for column in (row + 1)..N {
                let upper = self[(row, column)];
                let lower = self[(column, row)];
                let scale = upper.abs().max(lower.abs()).max(T::ONE);
                if (upper - lower).abs() > T::EPSILON_X30 * scale {
                    return false;
                }
            }
        }
        true
    }

    /// The matrix with each pair across the diagonal replaced by their average.
    ///
    /// A matrix that should read the same across the diagonal but has drifted — a covariance after
    /// many filter updates — comes back matching exactly. Only the upper half is worked out and
    /// then mirrored, so the two halves cannot end up a rounding step apart. The diagonal is left
    /// alone.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let drifted = Matrix::new([[1.0, 2.0], [4.0, 3.0]]);
    /// let evened = drifted.symmetrized();
    /// assert_eq!(evened[(0, 1)], 3.0);
    /// assert_eq!(evened[(0, 1)], evened[(1, 0)]);
    /// assert_eq!(evened[(0, 0)], 1.0);
    /// ```
    pub fn symmetrized(self) -> Self {
        let mut result = self;
        for row in 0..N {
            for column in (row + 1)..N {
                let average = (self[(row, column)] + self[(column, row)]) * T::HALF;
                result[(row, column)] = average;
                result[(column, row)] = average;
            }
        }
        result
    }

    /// The `N`×`N` identity matrix.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix: Matrix<3, 3> = Matrix::identity();
    /// assert_eq!(matrix.into_array(), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    /// ```
    #[inline]
    pub fn identity() -> Self {
        Self::from_diagonal([T::ONE; N])
    }

    /// The determinant.
    ///
    /// Sizes up to 4×4 use a closed form; larger ones use an LU factorization. A matrix whose
    /// factorization breaks down on an all-zero pivot column is exactly singular, so its
    /// determinant is zero. A small but nonzero pivot is still included here even when it would
    /// be too ill-conditioned for [`Matrix::lu_decompose`] to expose as a reusable solve factorization.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// assert_eq!(Matrix::new([[1.0, 2.0], [3.0, 4.0]]).determinant(), -2.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn determinant(self) -> T {
        match N {
            0 => T::ONE,
            1 => self.data[0][0],
            2 => self.determinant_2x2(),
            3 => self.determinant_3x3(),
            4 => self.determinant_4x4(),
            _ => match self.lu_for_determinant() {
                Ok(factorization) => factorization.determinant(),
                Err(_) => T::ZERO,
            },
        }
    }

    /// Returns the trace of the matrix (sum of diagonal entries).
    ///
    /// Note: this method computes the sum of the entries in order from top left
    /// to bottom right, which could have an impact on the accuracy of the result
    /// in the case of floating-point types if the earlier elements are significantly
    /// larger than the later ones.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// assert_eq!(Matrix::new([[1.0, -2.0], [3.0, 4.0]]).trace(), 5.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn trace(&self) -> T {
        (0..N).fold(T::ZERO, |acc, i| acc + self[(i, i)])
    }

    /// The inverse, or [`LinalgError::Singular`] if the matrix is singular. For sizes above 4×4,
    /// returns [`LinalgError::IllConditioned`] when the matrix is invertible but a pivot is too
    /// small relative to its largest entry for a reliable solve.
    ///
    /// Sizes up to 4×4 use a closed form and reject a matrix whose `|det|` is at or below an
    /// `EPSILON`-scaled threshold. Larger ones use an LU factorization and reject one whose
    /// smallest pivot is negligible relative to the matrix's largest absolute entry.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix: Matrix<2, 2> = Matrix::new([[4.0, 7.0], [2.0, 6.0]]);
    /// let product = (matrix * matrix.inverse().unwrap());
    /// assert!((product[(0, 0)] - 1.0).abs() < 1e-12 && (product[(1, 1)] - 1.0).abs() < 1e-12);
    /// assert!(Matrix::<2, 2>::new([[1.0, 2.0], [2.0, 4.0]]).inverse().is_err());
    /// ```
    #[inline]
    pub fn inverse(self) -> Result<Self, LinalgError> {
        match N {
            0 => Ok(self),
            1 => self.inverse_1x1(),
            2 => self.inverse_2x2(),
            3 => self.inverse_3x3(),
            4 => self.inverse_4x4(),
            _ => self.inverse_lu(),
        }
    }

    #[inline]
    fn inverse_1x1(mut self) -> Result<Self, LinalgError> {
        let value = self.data[0][0];
        if Self::det_near_singular(value, value.abs(), 1) {
            return Err(LinalgError::Singular);
        }
        self.data[0][0] = T::ONE / value;
        Ok(self)
    }

    #[inline]
    #[must_use]
    fn determinant_2x2(self) -> T {
        self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
    }

    #[inline]
    fn inverse_2x2(mut self) -> Result<Self, LinalgError> {
        let determinant = self.determinant_2x2();
        if Self::det_near_singular(determinant, self.max_abs(), 2) {
            return Err(LinalgError::Singular);
        }
        let scale = T::ONE / determinant;
        let matrix = self.data;
        self.data[0][0] = matrix[1][1] * scale;
        self.data[0][1] = -matrix[0][1] * scale;
        self.data[1][0] = -matrix[1][0] * scale;
        self.data[1][1] = matrix[0][0] * scale;
        Ok(self)
    }

    #[inline]
    #[must_use]
    fn determinant_3x3(self) -> T {
        let matrix = self.data;
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
            - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
            + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    }

    #[inline]
    fn inverse_3x3(mut self) -> Result<Self, LinalgError> {
        let determinant = self.determinant_3x3();
        if Self::det_near_singular(determinant, self.max_abs(), 3) {
            return Err(LinalgError::Singular);
        }
        let scale = T::ONE / determinant;
        let matrix = self.data;
        let adjugate = [
            [
                matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1],
                matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2],
                matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1],
            ],
            [
                matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2],
                matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0],
                matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2],
            ],
            [
                matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0],
                matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1],
                matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0],
            ],
        ];
        for (row, entries) in adjugate.iter().enumerate() {
            for (column, &entry) in entries.iter().enumerate() {
                self.data[row][column] = entry * scale;
            }
        }
        Ok(self)
    }

    /// The six 2×2 minors of the top row pair (`top`) and the bottom row pair (`bottom`),
    /// indexed by column pair `01, 02, 03, 12, 13, 23`. Both the 4×4 determinant and its
    /// adjugate are built from these, so they are computed once and shared.
    #[inline]
    #[must_use]
    fn row_pair_minors(self) -> ([T; 6], [T; 6]) {
        let matrix = self.data;
        let top = [
            matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0],
            matrix[0][0] * matrix[1][2] - matrix[0][2] * matrix[1][0],
            matrix[0][0] * matrix[1][3] - matrix[0][3] * matrix[1][0],
            matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1],
            matrix[0][1] * matrix[1][3] - matrix[0][3] * matrix[1][1],
            matrix[0][2] * matrix[1][3] - matrix[0][3] * matrix[1][2],
        ];
        let bottom = [
            matrix[2][0] * matrix[3][1] - matrix[2][1] * matrix[3][0],
            matrix[2][0] * matrix[3][2] - matrix[2][2] * matrix[3][0],
            matrix[2][0] * matrix[3][3] - matrix[2][3] * matrix[3][0],
            matrix[2][1] * matrix[3][2] - matrix[2][2] * matrix[3][1],
            matrix[2][1] * matrix[3][3] - matrix[2][3] * matrix[3][1],
            matrix[2][2] * matrix[3][3] - matrix[2][3] * matrix[3][2],
        ];
        (top, bottom)
    }

    #[inline]
    #[must_use]
    fn determinant_4x4(self) -> T {
        let (top, bottom) = self.row_pair_minors();
        top[0] * bottom[5] - top[1] * bottom[4] + top[2] * bottom[3] + top[3] * bottom[2]
            - top[4] * bottom[1]
            + top[5] * bottom[0]
    }

    #[inline]
    fn inverse_4x4(mut self) -> Result<Self, LinalgError> {
        let (top, bottom) = self.row_pair_minors();
        let determinant =
            top[0] * bottom[5] - top[1] * bottom[4] + top[2] * bottom[3] + top[3] * bottom[2]
                - top[4] * bottom[1]
                + top[5] * bottom[0];
        if Self::det_near_singular(determinant, self.max_abs(), 4) {
            return Err(LinalgError::Singular);
        }
        let scale = T::ONE / determinant;
        let matrix = self.data;
        let adjugate = [
            [
                matrix[1][1] * bottom[5] - matrix[1][2] * bottom[4] + matrix[1][3] * bottom[3],
                -matrix[0][1] * bottom[5] + matrix[0][2] * bottom[4] - matrix[0][3] * bottom[3],
                matrix[3][1] * top[5] - matrix[3][2] * top[4] + matrix[3][3] * top[3],
                -matrix[2][1] * top[5] + matrix[2][2] * top[4] - matrix[2][3] * top[3],
            ],
            [
                -matrix[1][0] * bottom[5] + matrix[1][2] * bottom[2] - matrix[1][3] * bottom[1],
                matrix[0][0] * bottom[5] - matrix[0][2] * bottom[2] + matrix[0][3] * bottom[1],
                -matrix[3][0] * top[5] + matrix[3][2] * top[2] - matrix[3][3] * top[1],
                matrix[2][0] * top[5] - matrix[2][2] * top[2] + matrix[2][3] * top[1],
            ],
            [
                matrix[1][0] * bottom[4] - matrix[1][1] * bottom[2] + matrix[1][3] * bottom[0],
                -matrix[0][0] * bottom[4] + matrix[0][1] * bottom[2] - matrix[0][3] * bottom[0],
                matrix[3][0] * top[4] - matrix[3][1] * top[2] + matrix[3][3] * top[0],
                -matrix[2][0] * top[4] + matrix[2][1] * top[2] - matrix[2][3] * top[0],
            ],
            [
                -matrix[1][0] * bottom[3] + matrix[1][1] * bottom[1] - matrix[1][2] * bottom[0],
                matrix[0][0] * bottom[3] - matrix[0][1] * bottom[1] + matrix[0][2] * bottom[0],
                -matrix[3][0] * top[3] + matrix[3][1] * top[1] - matrix[3][2] * top[0],
                matrix[2][0] * top[3] - matrix[2][1] * top[1] + matrix[2][2] * top[0],
            ],
        ];
        for (row, entries) in adjugate.iter().enumerate() {
            for (column, &entry) in entries.iter().enumerate() {
                self.data[row][column] = entry * scale;
            }
        }
        Ok(self)
    }

    #[inline]
    fn inverse_lu(self) -> Result<Self, LinalgError> {
        let factorization = self.lu_decompose()?;
        Ok(factorization.inverse())
    }
}

impl<const N: usize> Matrix<N, N> {
    /// Builds a symmetric positive-definite matrix from arbitrary entries as `M·Mᵀ`, ridged so the
    /// factorization is well conditioned rather than merely non-singular.
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix = Matrix::<2, 2>::symmetric_positive_definite(&[1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(matrix[(0, 1)], matrix[(1, 0)]);
    /// assert!(matrix[(0, 0)] > 0.0 && matrix[(1, 1)] > 0.0);
    /// ```
    pub fn symmetric_positive_definite(entries: &[f64]) -> Self {
        let factor = Self::from_fn(|row, column| entries[row * N + column]);
        factor * factor.transpose() + Self::from_diagonal([0.25; N])
    }
}

impl<const ROWS: usize, const COLS: usize, T> From<[[T; COLS]; ROWS]> for Matrix<ROWS, COLS, T> {
    #[inline]
    fn from(data: [[T; COLS]; ROWS]) -> Self {
        Matrix { data }
    }
}

impl<const ROWS: usize, const COLS: usize, T> Index<(usize, usize)> for Matrix<ROWS, COLS, T> {
    type Output = T;

    /// Panics if `(row, column)` is out of range. Use [`Self::get`] when the index may be invalid.
    #[inline]
    #[track_caller]
    fn index(&self, (row, column): (usize, usize)) -> &T {
        self.get_unchecked(row, column)
    }
}

impl<const ROWS: usize, const COLS: usize, T> IndexMut<(usize, usize)> for Matrix<ROWS, COLS, T> {
    /// Panics if `(row, column)` is out of range. Use [`Self::get_mut`] when the index may be invalid.
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, (row, column): (usize, usize)) -> &mut T {
        self.get_unchecked_mut(row, column)
    }
}

impl<const ROWS: usize, const COLS: usize, T: Numeric> Add for Matrix<ROWS, COLS, T> {
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: Self) -> Self {
        self += rhs;
        self
    }
}

impl<const ROWS: usize, const COLS: usize, T: Numeric> AddAssign for Matrix<ROWS, COLS, T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for (row, rhs_row) in self.data.iter_mut().zip(&rhs.data) {
            for (a, &b) in row.iter_mut().zip(rhs_row) {
                *a += b;
            }
        }
    }
}

impl<const ROWS: usize, const COLS: usize, T: Numeric> Sub for Matrix<ROWS, COLS, T> {
    type Output = Self;

    #[inline]
    fn sub(mut self, rhs: Self) -> Self {
        self -= rhs;
        self
    }
}

impl<const ROWS: usize, const COLS: usize, T: Numeric> SubAssign for Matrix<ROWS, COLS, T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for (row, rhs_row) in self.data.iter_mut().zip(&rhs.data) {
            for (a, &b) in row.iter_mut().zip(rhs_row) {
                *a -= b;
            }
        }
    }
}

impl<const ROWS: usize, const COLS: usize, T: Numeric> Neg for Matrix<ROWS, COLS, T> {
    type Output = Self;

    #[inline]
    fn neg(mut self) -> Self {
        for row in &mut self.data {
            for x in row.iter_mut() {
                *x = -*x;
            }
        }
        self
    }
}

impl<const ROWS: usize, const COLS: usize, T: Numeric> Mul<T> for Matrix<ROWS, COLS, T> {
    type Output = Self;

    #[inline]
    fn mul(self, scalar: T) -> Self {
        self.scale(scalar)
    }
}

impl<const ROWS: usize, const COLS: usize, T: Numeric> Div<T> for Matrix<ROWS, COLS, T> {
    type Output = Self;

    #[inline]
    fn div(self, scalar: T) -> Self {
        Self::from_fn(|i, j| self[(i, j)].safe_div(scalar))
    }
}

impl<const ROWS: usize, const COLS: usize, const C2: usize, T: Numeric> Mul<Matrix<COLS, C2, T>>
    for Matrix<ROWS, COLS, T>
{
    type Output = Matrix<ROWS, C2, T>;

    #[inline]
    fn mul(self, rhs: Matrix<COLS, C2, T>) -> Matrix<ROWS, C2, T> {
        Matrix::from_fn(|row, column| {
            let mut acc = T::ZERO;
            for k in 0..COLS {
                acc += self[(row, k)] * rhs[(k, column)];
            }
            acc
        })
    }
}

impl<const ROWS: usize, const COLS: usize, T: Numeric> Mul<Vector<COLS, T>>
    for Matrix<ROWS, COLS, T>
{
    type Output = Vector<ROWS, T>;

    #[inline]
    fn mul(self, rhs: Vector<COLS, T>) -> Vector<ROWS, T> {
        Vector::from_fn(|row| {
            let mut acc = T::ZERO;
            for column in 0..COLS {
                acc += self[(row, column)] * rhs[column];
            }
            acc
        })
    }
}
