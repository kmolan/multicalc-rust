use crate::scalar::Numeric;
use crate::utils::summation::PairwiseSum;

pub mod linear_approximation;
pub mod quadratic_approximation;

/// Goodness-of-fit metrics shared by the linear and quadratic approximators.
///
/// Returns `(mae, mse, rmse, r_squared, adjusted_r_squared)`. `num_predictors` is the
/// number of fitted terms (`p`), used only for the adjusted R². `r_squared` is `NaN` when
/// the truth is constant over `points` (zero total variance); `adjusted_r_squared` is
/// `NaN` when there are too few points (`num_points <= num_predictors + 1`).
pub(crate) fn compute_metrics<T, P, O, const NUM_VARS: usize, const NUM_POINTS: usize>(
    predict: P,
    points: &[[T; NUM_VARS]; NUM_POINTS],
    original_function: &O,
    num_predictors: usize,
) -> (T, T, T, T, T)
where
    T: Numeric,
    P: Fn(&[T; NUM_VARS]) -> T,
    O: Fn(&[T; NUM_VARS]) -> T,
{
    let n = T::from_usize(NUM_POINTS);

    let mut mean_y = PairwiseSum::new();
    for point in points {
        mean_y.add(original_function(point));
    }
    let mean_y = mean_y.total() / n;

    let mut sum_abs = PairwiseSum::new();
    let mut ss_res = PairwiseSum::new();
    let mut ss_tot = PairwiseSum::new();
    for point in points {
        let y = original_function(point);
        let residual = predict(point) - y;
        sum_abs.add(residual.abs());
        ss_res.add(residual * residual);
        ss_tot.add((y - mean_y) * (y - mean_y));
    }
    let (sum_abs, ss_res, ss_tot) = (sum_abs.total(), ss_res.total(), ss_tot.total());

    let mae = sum_abs / n;
    let mse = ss_res / n;
    let rmse = mse.sqrt();

    let r_squared = if ss_tot == T::ZERO {
        T::NAN
    } else {
        T::ONE - ss_res / ss_tot
    };

    let denominator = n - T::from_usize(num_predictors) - T::ONE;
    let adjusted_r_squared = if denominator <= T::ZERO {
        T::NAN
    } else {
        T::ONE - (T::ONE - r_squared) * (n - T::ONE) / denominator
    };

    (mae, mse, rmse, r_squared, adjusted_r_squared)
}
