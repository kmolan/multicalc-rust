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
    /// let m = Matrix::new([[1.0, -2.0, 0.0], [3.0, 0.0, 4.0], [2.0, -1.0, 1.0]]);
    /// assert_eq!(m.frobenius_norm(), 6.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn frobenius_norm(self) -> T {
        let total = self.data.into_iter().flatten().fold(T::ZERO, |acc, x| {
            // Note: implementing `|x|^2` as `x * x` is correct for real numbers,
            // however would be incorrect for complex numbers. In that case it
            // should be `x * x.conj()` (i.e. multiplying by the complex conjugate).
            // If this library is expected to work will complex numbers in the future
            // then this will need to be updated; the `Numeric` trait would also need
            // updating to include a complex conjugate operation.
            acc + x * x
        });
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
    /// The `N`×`N` diagonal matrix with the given diagonal entries
    /// (all off-diagonal elements are equal to zero).
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let m = Matrix::from_diagonal([1.0, 2.0, 3.0]);
    /// assert_eq!(m.into_array(), [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]);
    /// ```
    #[inline]
    pub fn from_diagonal(diag: [T; N]) -> Self {
        let rows = core::array::from_fn(|i| {
            let mut r = [T::ZERO; N];
            r[i] = diag[i];
            r
        });
        Matrix::new(rows)
    }

    /// The `N`×`N` identity matrix.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let i: Matrix<3, 3> = Matrix::identity();
    /// assert_eq!(i.into_array(), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    /// ```
    #[inline]
    pub fn identity() -> Self {
        Self::from_diagonal([T::ONE; N])
    }

    /// The determinant.
    ///
    /// Sizes up to 4×4 use a closed form; larger ones use an LU factorization. A matrix whose
    /// factorization breaks down on an all-zero pivot column is exactly singular, so its
    /// determinant is zero.
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
            1 => self[(0, 0)],
            2 => self.determinant_2x2(),
            3 => self.determinant_3x3(),
            4 => self.determinant_4x4(),
            _ => match self.lu() {
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

    /// The inverse, or [`LinalgError::Singular`] if the matrix is singular or near-singular.
    ///
    /// Sizes up to 4×4 use a closed form and reject a matrix whose `|det|` is at or below an
    /// `EPSILON`-scaled threshold. Larger ones use an LU factorization and reject one whose
    /// smallest pivot is negligible against its largest.
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
        let value = self[(0, 0)];
        if Self::det_near_singular(value, value.abs(), 1) {
            return Err(LinalgError::Singular);
        }
        self[(0, 0)] = T::ONE / value;
        Ok(self)
    }

    #[inline]
    fn determinant_2x2(self) -> T {
        self[(0, 0)] * self[(1, 1)] - self[(0, 1)] * self[(1, 0)]
    }

    #[inline]
    fn inverse_2x2(mut self) -> Result<Self, LinalgError> {
        let determinant = self.determinant_2x2();
        if Self::det_near_singular(determinant, self.max_abs(), 2) {
            return Err(LinalgError::Singular);
        }
        let scale = T::ONE / determinant;
        let m = self;
        self[(0, 0)] = m[(1, 1)] * scale;
        self[(0, 1)] = -m[(0, 1)] * scale;
        self[(1, 0)] = -m[(1, 0)] * scale;
        self[(1, 1)] = m[(0, 0)] * scale;
        Ok(self)
    }

    #[inline]
    fn determinant_3x3(self) -> T {
        let m = self;
        m[(0, 0)] * (m[(1, 1)] * m[(2, 2)] - m[(1, 2)] * m[(2, 1)])
            - m[(0, 1)] * (m[(1, 0)] * m[(2, 2)] - m[(1, 2)] * m[(2, 0)])
            + m[(0, 2)] * (m[(1, 0)] * m[(2, 1)] - m[(1, 1)] * m[(2, 0)])
    }

    #[inline]
    fn inverse_3x3(mut self) -> Result<Self, LinalgError> {
        let determinant = self.determinant_3x3();
        if Self::det_near_singular(determinant, self.max_abs(), 3) {
            return Err(LinalgError::Singular);
        }
        let scale = T::ONE / determinant;
        let m = self;
        let adjugate = [
            [
                m[(1, 1)] * m[(2, 2)] - m[(1, 2)] * m[(2, 1)],
                m[(0, 2)] * m[(2, 1)] - m[(0, 1)] * m[(2, 2)],
                m[(0, 1)] * m[(1, 2)] - m[(0, 2)] * m[(1, 1)],
            ],
            [
                m[(1, 2)] * m[(2, 0)] - m[(1, 0)] * m[(2, 2)],
                m[(0, 0)] * m[(2, 2)] - m[(0, 2)] * m[(2, 0)],
                m[(0, 2)] * m[(1, 0)] - m[(0, 0)] * m[(1, 2)],
            ],
            [
                m[(1, 0)] * m[(2, 1)] - m[(1, 1)] * m[(2, 0)],
                m[(0, 1)] * m[(2, 0)] - m[(0, 0)] * m[(2, 1)],
                m[(0, 0)] * m[(1, 1)] - m[(0, 1)] * m[(1, 0)],
            ],
        ];
        for (row, entries) in adjugate.iter().enumerate() {
            for (column, &entry) in entries.iter().enumerate() {
                self[(row, column)] = entry * scale;
            }
        }
        Ok(self)
    }

    /// The six 2×2 minors of the top row pair (`top`) and the bottom row pair (`bottom`),
    /// indexed by column pair `01, 02, 03, 12, 13, 23`. Both the 4×4 determinant and its
    /// adjugate are built from these, so they are computed once and shared.
    #[inline]
    fn row_pair_minors(self) -> ([T; 6], [T; 6]) {
        let m = self;
        let top = [
            m[(0, 0)] * m[(1, 1)] - m[(0, 1)] * m[(1, 0)],
            m[(0, 0)] * m[(1, 2)] - m[(0, 2)] * m[(1, 0)],
            m[(0, 0)] * m[(1, 3)] - m[(0, 3)] * m[(1, 0)],
            m[(0, 1)] * m[(1, 2)] - m[(0, 2)] * m[(1, 1)],
            m[(0, 1)] * m[(1, 3)] - m[(0, 3)] * m[(1, 1)],
            m[(0, 2)] * m[(1, 3)] - m[(0, 3)] * m[(1, 2)],
        ];
        let bottom = [
            m[(2, 0)] * m[(3, 1)] - m[(2, 1)] * m[(3, 0)],
            m[(2, 0)] * m[(3, 2)] - m[(2, 2)] * m[(3, 0)],
            m[(2, 0)] * m[(3, 3)] - m[(2, 3)] * m[(3, 0)],
            m[(2, 1)] * m[(3, 2)] - m[(2, 2)] * m[(3, 1)],
            m[(2, 1)] * m[(3, 3)] - m[(2, 3)] * m[(3, 1)],
            m[(2, 2)] * m[(3, 3)] - m[(2, 3)] * m[(3, 2)],
        ];
        (top, bottom)
    }

    #[inline]
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
        let m = self;
        let adjugate = [
            [
                m[(1, 1)] * bottom[5] - m[(1, 2)] * bottom[4] + m[(1, 3)] * bottom[3],
                -m[(0, 1)] * bottom[5] + m[(0, 2)] * bottom[4] - m[(0, 3)] * bottom[3],
                m[(3, 1)] * top[5] - m[(3, 2)] * top[4] + m[(3, 3)] * top[3],
                -m[(2, 1)] * top[5] + m[(2, 2)] * top[4] - m[(2, 3)] * top[3],
            ],
            [
                -m[(1, 0)] * bottom[5] + m[(1, 2)] * bottom[2] - m[(1, 3)] * bottom[1],
                m[(0, 0)] * bottom[5] - m[(0, 2)] * bottom[2] + m[(0, 3)] * bottom[1],
                -m[(3, 0)] * top[5] + m[(3, 2)] * top[2] - m[(3, 3)] * top[1],
                m[(2, 0)] * top[5] - m[(2, 2)] * top[2] + m[(2, 3)] * top[1],
            ],
            [
                m[(1, 0)] * bottom[4] - m[(1, 1)] * bottom[2] + m[(1, 3)] * bottom[0],
                -m[(0, 0)] * bottom[4] + m[(0, 1)] * bottom[2] - m[(0, 3)] * bottom[0],
                m[(3, 0)] * top[4] - m[(3, 1)] * top[2] + m[(3, 3)] * top[0],
                -m[(2, 0)] * top[4] + m[(2, 1)] * top[2] - m[(2, 3)] * top[0],
            ],
            [
                -m[(1, 0)] * bottom[3] + m[(1, 1)] * bottom[1] - m[(1, 2)] * bottom[0],
                m[(0, 0)] * bottom[3] - m[(0, 1)] * bottom[1] + m[(0, 2)] * bottom[0],
                -m[(3, 0)] * top[3] + m[(3, 1)] * top[1] - m[(3, 2)] * top[0],
                m[(2, 0)] * top[3] - m[(2, 1)] * top[1] + m[(2, 2)] * top[0],
            ],
        ];
        for (row, entries) in adjugate.iter().enumerate() {
            for (column, &entry) in entries.iter().enumerate() {
                self[(row, column)] = entry * scale;
            }
        }
        Ok(self)
    }

    #[inline]
    fn inverse_lu(self) -> Result<Self, LinalgError> {
        let factorization = self.lu()?;
        Ok(factorization.inverse())
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
impl<T: Numeric> Matrix<2, 2, T> {}

impl<T: Numeric> Matrix<3, 3, T> {}

// The 4×4 closed form shares the twelve 2×2 row-pair minors between the determinant and the
// adjugate, so a full inverse costs little more than the determinant alone.
impl<T: Numeric> Matrix<4, 4, T> {}
