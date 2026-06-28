pub mod linear_approximation;
pub mod quadratic_approximation;

#[cfg(test)]
mod test;

/// Goodness-of-fit metrics shared by the linear and quadratic approximators.
///
/// Returns `(mae, mse, rmse, r_squared, adjusted_r_squared)`. `num_predictors` is the
/// number of fitted terms (`p`), used only for the adjusted R². `r_squared` is `NaN` when
/// the truth is constant over `points` (zero total variance); `adjusted_r_squared` is
/// `NaN` when there are too few points (`num_points <= num_predictors + 1`).
pub(crate) fn compute_metrics<P, O, const NUM_VARS: usize, const NUM_POINTS: usize>(
    predict: P,
    points: &[[f64; NUM_VARS]; NUM_POINTS],
    original_function: &O,
    num_predictors: usize,
) -> (f64, f64, f64, f64, f64)
where
    P: Fn(&[f64; NUM_VARS]) -> f64,
    O: Fn(&[f64; NUM_VARS]) -> f64,
{
    let n = NUM_POINTS as f64;

    let mut mean_y = 0.0;
    for point in points {
        mean_y += original_function(point);
    }
    mean_y /= n;

    let mut sum_abs = 0.0;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for point in points {
        let y = original_function(point);
        let residual = predict(point) - y;
        sum_abs += libm::fabs(residual);
        ss_res += residual * residual;
        ss_tot += (y - mean_y) * (y - mean_y);
    }

    let mae = sum_abs / n;
    let mse = ss_res / n;
    let rmse = libm::sqrt(mse);

    let r_squared = if ss_tot == 0.0 {
        f64::NAN
    } else {
        1.0 - ss_res / ss_tot
    };

    let denominator = n - num_predictors as f64 - 1.0;
    let adjusted_r_squared = if denominator <= 0.0 {
        f64::NAN
    } else {
        1.0 - (1.0 - r_squared) * (n - 1.0) / denominator
    };

    (mae, mse, rmse, r_squared, adjusted_r_squared)
}
