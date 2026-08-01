//! Literal construction macros for [`Polynomial`](crate::Polynomial) and
//! [`MultivariatePolynomial`](crate::MultivariatePolynomial).

/// Builds a [`Polynomial`](crate::Polynomial) from its coefficients, lowest power first.
///
/// ```
/// use multicalc::{polynomial, Polynomial};
///
/// let p = polynomial![1.0, -2.0, 3.0]; // 1 - 2x + 3x²
/// assert_eq!(p, Polynomial::new([1.0, -2.0, 3.0]));
/// ```
#[macro_export]
macro_rules! polynomial {
    ($($coefficient:expr),+ $(,)?) => {
        $crate::Polynomial::new([$($coefficient),+])
    };
}

/// Builds a [`MultivariatePolynomial`](crate::MultivariatePolynomial) from pairs of a number and
/// the power each variable is raised to.
///
/// The result is a `Result`, since a macro cannot unwrap for you; the number of terms written is
/// what the polynomial has room for.
///
/// ```
/// use multicalc::{multivariate_polynomial, MultivariatePolynomial};
///
/// // 2.5·x²·y³ - y
/// let p: MultivariatePolynomial<2, 2> =
///     multivariate_polynomial![(2.5, [2, 3]), (-1.0, [0, 1])].unwrap();
/// assert_eq!(p.len(), 2);
/// assert!((p.evaluate(&[2.0, 1.0]) - 9.0).abs() < 1e-12);
/// ```
///
/// Power lists of differing lengths are rejected at compile time, since they would describe
/// different numbers of variables:
///
/// ```compile_fail
/// use multicalc::multivariate_polynomial;
/// let _ = multivariate_polynomial![(1.0, [1, 0]), (2.0, [0, 1, 1])];
/// ```
#[macro_export]
macro_rules! multivariate_polynomial {
    ($(($coefficient:expr, [$($exponent:expr),+ $(,)?])),+ $(,)?) => {
        $crate::MultivariatePolynomial::try_from_array([
            $($crate::MultivariateTerm::new($coefficient, [$($exponent),+])),+
        ])
    };
}
