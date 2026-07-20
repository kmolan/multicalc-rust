//! Fixed-size, stack-allocated matrix.

use core::ops::{Add, AddAssign, Index, IndexMut, Mul, Neg, Sub, SubAssign};

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
    /// let m = Matrix::<2, 2>::from_fn(|r, c| (r * 2 + c) as f64);
    /// assert_eq!(m.into_array(), [[0.0, 1.0], [2.0, 3.0]]);
    /// ```
    #[inline]
    pub fn from_fn(mut f: impl FnMut(usize, usize) -> T) -> Self {
        Matrix {
            data: core::array::from_fn(|r| core::array::from_fn(|c| f(r, c))),
        }
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
        (slice.len() == ROWS * COLS).then(|| Self::from_fn(|r, c| slice[r * COLS + c]))
    }

    /// Copies row `r` into a vector. Panics if `r >= ROWS`.
    ///
    /// ```
    /// use multicalc::linear_algebra::{Matrix, Vector};
    /// let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// assert_eq!(m.row(1), Vector::new([3.0, 4.0]));
    /// ```
    #[inline]
    pub fn row(&self, r: usize) -> Vector<COLS, T> {
        Vector::new(self.data[r])
    }

    /// Copies column `c` into a vector. Panics if `c >= COLS`.
    ///
    /// ```
    /// use multicalc::linear_algebra::{Matrix, Vector};
    /// let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// assert_eq!(m.column(0), Vector::new([1.0, 3.0]));
    /// ```
    #[inline]
    pub fn column(&self, c: usize) -> Vector<ROWS, T> {
        Vector::from_fn(|r| self.data[r][c])
    }
}

impl<const ROWS: usize, const COLS: usize, T: Numeric> Matrix<ROWS, COLS, T> {
    /// The zero matrix.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let m: Matrix<2, 3> = Matrix::zeros();
    /// assert_eq!(m.into_array(), [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    /// ```
    #[inline]
    pub fn zeros() -> Self {
        Matrix::from_fn(|_, _| T::ZERO)
    }

    /// Multiplies every element by `scalar`.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// assert_eq!(Matrix::new([[1.0, 2.0]]).scale(3.0).into_array(), [[3.0, 6.0]]);
    /// ```
    #[inline]
    pub fn scale(self, scalar: T) -> Self {
        Matrix::from_fn(|r, c| self[(r, c)] * scalar)
    }

    /// The transpose, with rows and columns swapped.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let m = Matrix::new([[1.0, 2.0, 3.0]]);
    /// assert_eq!(m.transpose().into_array(), [[1.0], [2.0], [3.0]]);
    /// ```
    #[inline]
    pub fn transpose(self) -> Matrix<COLS, ROWS, T> {
        Matrix::from_fn(|r, c| self[(c, r)])
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
    fn max_abs(self) -> T {
        let mut best = T::ZERO;
        for r in 0..ROWS {
            for c in 0..COLS {
                best = best.max(self[(r, c)].abs());
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
    /// The `N`×`N` identity matrix.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let i: Matrix<3, 3> = Matrix::identity();
    /// assert_eq!(i.into_array(), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    /// ```
    #[inline]
    pub fn identity() -> Self {
        Matrix::from_fn(|r, c| if r == c { T::ONE } else { T::ZERO })
    }

    #[inline]
    #[must_use]
    pub fn determinant_general(self) -> T {
        let Ok(lu) = self.lu() else { return T::ZERO };
        lu.determinant()
    }

    pub fn inverse_general(self) -> Result<Self, LinalgError> {
        Ok(self.lu()?.inverse())
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

    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &T {
        &self.data[row][col]
    }
}

impl<const ROWS: usize, const COLS: usize, T> IndexMut<(usize, usize)> for Matrix<ROWS, COLS, T> {
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        &mut self.data[row][col]
    }
}

impl<const ROWS: usize, const COLS: usize, T: Numeric> Add for Matrix<ROWS, COLS, T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Matrix::from_fn(|r, c| self[(r, c)] + rhs[(r, c)])
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
    fn sub(self, rhs: Self) -> Self {
        Matrix::from_fn(|r, c| self[(r, c)] - rhs[(r, c)])
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
    fn neg(self) -> Self {
        Matrix::from_fn(|r, c| -self[(r, c)])
    }
}

impl<const ROWS: usize, const COLS: usize, T: Numeric> Mul<T> for Matrix<ROWS, COLS, T> {
    type Output = Self;

    #[inline]
    fn mul(self, scalar: T) -> Self {
        self.scale(scalar)
    }
}

impl<const ROWS: usize, const COLS: usize, const C2: usize, T: Numeric> Mul<Matrix<COLS, C2, T>>
    for Matrix<ROWS, COLS, T>
{
    type Output = Matrix<ROWS, C2, T>;

    #[inline]
    fn mul(self, rhs: Matrix<COLS, C2, T>) -> Matrix<ROWS, C2, T> {
        Matrix::from_fn(|r, c| {
            let mut acc = T::ZERO;
            for k in 0..COLS {
                acc += self[(r, k)] * rhs[(k, c)];
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
        Vector::from_fn(|r| self.row(r).dot(rhs))
    }
}

// The 2×2, 3×3, and 4×4 determinant and inverse are written out in closed form. These are the
// sizes seen most often, and the inline expressions keep them low-latency, sparing them the
// loops and pivoting a general factorization would need.
impl<T: Numeric> Matrix<2, 2, T> {
    /// The determinant `m[(0,0)]*m[(1,1)] - m[(0,1)]*m[(1,0)]`.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// assert_eq!(Matrix::new([[1.0, 2.0], [3.0, 4.0]]).determinant(), -2.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn determinant(self) -> T {
        self[(0, 0)] * self[(1, 1)] - self[(0, 1)] * self[(1, 0)]
    }

    /// The inverse, or [`LinalgError::Singular`] if the matrix is singular or
    /// near-singular (`|det|` at or below an `EPSILON`-scaled threshold).
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let m: Matrix<2, 2> = Matrix::new([[4.0, 7.0], [2.0, 6.0]]);
    /// let p = m * m.inverse().unwrap();
    /// assert!((p[(0, 0)] - 1.0).abs() < 1e-12 && (p[(1, 1)] - 1.0).abs() < 1e-12);
    /// assert!(Matrix::<2, 2>::new([[1.0, 2.0], [2.0, 4.0]]).inverse().is_err());
    /// ```
    #[inline]
    pub fn inverse(self) -> Result<Self, LinalgError> {
        let det = self.determinant();
        if Self::det_near_singular(det, self.max_abs(), 2) {
            return Err(LinalgError::Singular);
        }
        let inv = T::ONE / det;
        let m = self;
        Ok(Matrix::new([
            [m[(1, 1)] * inv, -m[(0, 1)] * inv],
            [-m[(1, 0)] * inv, m[(0, 0)] * inv],
        ]))
    }
}

impl<T: Numeric> Matrix<3, 3, T> {
    /// The determinant, by cofactor expansion along the first row.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let i: Matrix<3, 3> = Matrix::identity();
    /// assert_eq!(i.determinant(), 1.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn determinant(self) -> T {
        let m = self;
        m[(0, 0)] * (m[(1, 1)] * m[(2, 2)] - m[(1, 2)] * m[(2, 1)])
            - m[(0, 1)] * (m[(1, 0)] * m[(2, 2)] - m[(1, 2)] * m[(2, 0)])
            + m[(0, 2)] * (m[(1, 0)] * m[(2, 1)] - m[(1, 1)] * m[(2, 0)])
    }

    /// The inverse, or [`LinalgError::Singular`] if the matrix is singular or
    /// near-singular (`|det|` at or below an `EPSILON`-scaled threshold).
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let i: Matrix<3, 3> = Matrix::identity();
    /// assert_eq!(i.inverse().unwrap().into_array(), i.into_array());
    /// ```
    #[inline]
    pub fn inverse(self) -> Result<Self, LinalgError> {
        let det = self.determinant();
        if Self::det_near_singular(det, self.max_abs(), 3) {
            return Err(LinalgError::Singular);
        }
        let inv = T::ONE / det;
        let m = self;
        Ok(Matrix::new([
            [
                (m[(1, 1)] * m[(2, 2)] - m[(1, 2)] * m[(2, 1)]) * inv,
                (m[(0, 2)] * m[(2, 1)] - m[(0, 1)] * m[(2, 2)]) * inv,
                (m[(0, 1)] * m[(1, 2)] - m[(0, 2)] * m[(1, 1)]) * inv,
            ],
            [
                (m[(1, 2)] * m[(2, 0)] - m[(1, 0)] * m[(2, 2)]) * inv,
                (m[(0, 0)] * m[(2, 2)] - m[(0, 2)] * m[(2, 0)]) * inv,
                (m[(0, 2)] * m[(1, 0)] - m[(0, 0)] * m[(1, 2)]) * inv,
            ],
            [
                (m[(1, 0)] * m[(2, 1)] - m[(1, 1)] * m[(2, 0)]) * inv,
                (m[(0, 1)] * m[(2, 0)] - m[(0, 0)] * m[(2, 1)]) * inv,
                (m[(0, 0)] * m[(1, 1)] - m[(0, 1)] * m[(1, 0)]) * inv,
            ],
        ]))
    }
}

// The 4×4 closed form shares the twelve 2×2 row-pair minors between the determinant and the
// adjugate, so a full inverse costs little more than the determinant alone.
impl<T: Numeric> Matrix<4, 4, T> {
    /// The six 2×2 minors of the top row pair (`s`) and the bottom row pair (`c`), indexed
    /// by column pair `01, 02, 03, 12, 13, 23`. Both the determinant and the adjugate are
    /// built from these, so they are computed once and shared.
    #[inline]
    fn row_pair_minors(self) -> ([T; 6], [T; 6]) {
        let m = self;
        let s = [
            m[(0, 0)] * m[(1, 1)] - m[(0, 1)] * m[(1, 0)],
            m[(0, 0)] * m[(1, 2)] - m[(0, 2)] * m[(1, 0)],
            m[(0, 0)] * m[(1, 3)] - m[(0, 3)] * m[(1, 0)],
            m[(0, 1)] * m[(1, 2)] - m[(0, 2)] * m[(1, 1)],
            m[(0, 1)] * m[(1, 3)] - m[(0, 3)] * m[(1, 1)],
            m[(0, 2)] * m[(1, 3)] - m[(0, 3)] * m[(1, 2)],
        ];
        let c = [
            m[(2, 0)] * m[(3, 1)] - m[(2, 1)] * m[(3, 0)],
            m[(2, 0)] * m[(3, 2)] - m[(2, 2)] * m[(3, 0)],
            m[(2, 0)] * m[(3, 3)] - m[(2, 3)] * m[(3, 0)],
            m[(2, 1)] * m[(3, 2)] - m[(2, 2)] * m[(3, 1)],
            m[(2, 1)] * m[(3, 3)] - m[(2, 3)] * m[(3, 1)],
            m[(2, 2)] * m[(3, 3)] - m[(2, 3)] * m[(3, 2)],
        ];
        (s, c)
    }

    /// The determinant, as the Laplace expansion along the first two rows.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let m = Matrix::<4, 4>::new([
    ///     [2.0, 1.0, 1.0, 1.0],
    ///     [0.0, 3.0, 1.0, 1.0],
    ///     [0.0, 0.0, 4.0, 1.0],
    ///     [0.0, 0.0, 0.0, 5.0],
    /// ]);
    /// assert_eq!(m.determinant(), 120.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn determinant(self) -> T {
        let (s, c) = self.row_pair_minors();
        s[0] * c[5] - s[1] * c[4] + s[2] * c[3] + s[3] * c[2] - s[4] * c[1] + s[5] * c[0]
    }

    /// The inverse, or [`LinalgError::Singular`] if the matrix is singular or
    /// near-singular (`|det|` at or below an `EPSILON`-scaled threshold).
    ///
    /// Built from the adjugate, the transpose of the cofactor matrix, scaled by `1/det`.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let m = Matrix::<4, 4>::new([
    ///     [2.0, 1.0, 1.0, 1.0],
    ///     [0.0, 3.0, 1.0, 1.0],
    ///     [0.0, 0.0, 4.0, 1.0],
    ///     [0.0, 0.0, 0.0, 5.0],
    /// ]);
    /// let p = m * m.inverse().unwrap();
    /// for r in 0..4 {
    ///     for c in 0..4 {
    ///         let expected = if r == c { 1.0 } else { 0.0 };
    ///         assert!((p[(r, c)] - expected).abs() < 1e-12);
    ///     }
    /// }
    /// ```
    #[inline]
    pub fn inverse(self) -> Result<Self, LinalgError> {
        let (s, c) = self.row_pair_minors();
        let det = s[0] * c[5] - s[1] * c[4] + s[2] * c[3] + s[3] * c[2] - s[4] * c[1] + s[5] * c[0];
        if Self::det_near_singular(det, self.max_abs(), 4) {
            return Err(LinalgError::Singular);
        }
        let inv = T::ONE / det;
        let m = self;
        Ok(Matrix::new([
            [
                (m[(1, 1)] * c[5] - m[(1, 2)] * c[4] + m[(1, 3)] * c[3]) * inv,
                (-m[(0, 1)] * c[5] + m[(0, 2)] * c[4] - m[(0, 3)] * c[3]) * inv,
                (m[(3, 1)] * s[5] - m[(3, 2)] * s[4] + m[(3, 3)] * s[3]) * inv,
                (-m[(2, 1)] * s[5] + m[(2, 2)] * s[4] - m[(2, 3)] * s[3]) * inv,
            ],
            [
                (-m[(1, 0)] * c[5] + m[(1, 2)] * c[2] - m[(1, 3)] * c[1]) * inv,
                (m[(0, 0)] * c[5] - m[(0, 2)] * c[2] + m[(0, 3)] * c[1]) * inv,
                (-m[(3, 0)] * s[5] + m[(3, 2)] * s[2] - m[(3, 3)] * s[1]) * inv,
                (m[(2, 0)] * s[5] - m[(2, 2)] * s[2] + m[(2, 3)] * s[1]) * inv,
            ],
            [
                (m[(1, 0)] * c[4] - m[(1, 1)] * c[2] + m[(1, 3)] * c[0]) * inv,
                (-m[(0, 0)] * c[4] + m[(0, 1)] * c[2] - m[(0, 3)] * c[0]) * inv,
                (m[(3, 0)] * s[4] - m[(3, 1)] * s[2] + m[(3, 3)] * s[0]) * inv,
                (-m[(2, 0)] * s[4] + m[(2, 1)] * s[2] - m[(2, 3)] * s[0]) * inv,
            ],
            [
                (-m[(1, 0)] * c[3] + m[(1, 1)] * c[1] - m[(1, 2)] * c[0]) * inv,
                (m[(0, 0)] * c[3] - m[(0, 1)] * c[1] + m[(0, 2)] * c[0]) * inv,
                (-m[(3, 0)] * s[3] + m[(3, 1)] * s[1] - m[(3, 2)] * s[0]) * inv,
                (m[(2, 0)] * s[3] - m[(2, 1)] * s[1] + m[(2, 2)] * s[0]) * inv,
            ],
        ]))
    }
}
