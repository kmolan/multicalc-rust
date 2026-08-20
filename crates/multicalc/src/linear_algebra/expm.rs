//! Matrix exponential by scaling-and-squaring with a diagonal Padé approximant.

use crate::error::LinalgError;
use crate::linear_algebra::Matrix;
use crate::scalar::Numeric;

impl<const N: usize, T: Numeric> Matrix<N, N, T> {
    /// The matrix exponential `e^self`, by scaling-and-squaring with a degree-6 diagonal Padé
    /// approximant. Exact derivatives flow through: the scaling exponent is chosen from the
    /// (primal) 1-norm, an integer that is never differentiated.
    ///
    /// Returns [`LinalgError::NonFinite`] if the input or result has a non-finite entry.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// # fn main() -> Result<(), multicalc::error::LinalgError> {
    /// let exponential = Matrix::<3, 3>::zeros().expm()?; // e^0 = I
    /// assert!((exponential[(0, 0)] - 1.0).abs() < 1e-12 && exponential[(0, 1)].abs() < 1e-12);
    /// # Ok(())
    /// # }
    /// ```
    pub fn expm(self) -> Result<Matrix<N, N, T>, LinalgError> {
        if !self.is_finite() {
            return Err(LinalgError::NonFinite);
        }

        // 1-norm: the largest absolute column sum.
        let mut nrm = T::ZERO;
        for j in 0..N {
            let mut col = T::ZERO;
            for i in 0..N {
                col += self[(i, j)].abs();
            }
            if col > nrm {
                nrm = col;
            }
        }
        // Scale so ‖A / 2^s‖₁ ≤ 1/2. `squares` is that exponent and depends only on the primal, so it is AD-safe.
        let mut squares: i32 = 0;
        let mut scaled = nrm;
        while scaled > T::HALF {
            scaled *= T::HALF;
            squares += 1;
        }
        let a = self.scale(T::HALF.powi(squares));

        // Degree-6 Padé coefficients by the standard recurrence (1, 1/2, 5/44, 1/66, …).
        let mut coef = [T::ZERO; 7];
        coef[0] = T::ONE;
        for k in 1..7 {
            let k_f = k as f64;
            coef[k] = coef[k - 1] * T::from_f64((6.0 - k_f + 1.0) / (k_f * (12.0 - k_f + 1.0)));
        }
        // num = Σ c_k A^k, den = Σ c_k (−A)^k, sharing the running power A^k.
        let mut apow = Matrix::identity();
        let mut num = Matrix::zeros();
        let mut den = Matrix::zeros();
        #[allow(clippy::needless_range_loop)]
        for k in 0..7 {
            num += apow.scale(coef[k]);
            let signed = if k % 2 == 0 { coef[k] } else { -coef[k] };
            den += apow.scale(signed);
            if k < 6 {
                apow = apow * a;
            }
        }
        // e^A ≈ den⁻¹ · num, then square `squares` times to undo the scaling.
        let mut result = den.lu_decompose()?.solve_matrix(num);
        for _ in 0..squares {
            result = result * result;
        }
        if !result.is_finite() {
            return Err(LinalgError::NonFinite);
        }
        Ok(result)
    }
}
