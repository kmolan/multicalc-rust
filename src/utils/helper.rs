///utility to convert the transpose the matrix
///takes an input of a matrix of shape MxN, and returns the matrix as NxM
pub const fn transpose<const NUM_ROWS: usize, const NUM_COLUMNS: usize>(
    matrix: &[[f64; NUM_COLUMNS]; NUM_ROWS],
) -> [[f64; NUM_ROWS]; NUM_COLUMNS] {
    let mut result = [[0.0; NUM_ROWS]; NUM_COLUMNS];

    let mut row_index = 0;
    while row_index < NUM_COLUMNS {
        let mut col_index = 0;
        while col_index < NUM_ROWS {
            result[row_index][col_index] = matrix[col_index][row_index];
            col_index += 1;
        }
        row_index += 1;
    }

    result
}

#[inline(always)]
pub const fn powi(mut base: f64, exp: i32) -> f64 {
    if exp == 0 {
        return 1.0;
    }

    let mut result = 1.0;
    let mut e = if exp < 0 { -exp } else { exp };

    while e > 0 {
        if (e & 1) == 1 {
            result *= base;
        }
        base *= base;
        e >>= 1;
    }

    if exp < 0 {
        1.0 / result
    } else {
        result
    }
}

/// Approximates the square root of `x` using Newton-Raphson iteration.
///
/// Returns NaN for negative inputs.
///
/// Accurate to within **1e-10** compared to f64::sqrt().
pub const fn sqrt(x: f64) -> f64 {
    // Invalid input (non-positive)
    if x < 0.0 {
        return f64::NAN;
    }

    // Trivial case
    if x == 0.0 {
        return 0.0;
    }

    let mut guess = if x < 1.0 { x } else { x / 2.0 };
    let mut i = 0;

    // Perform Newton's method iterations
    while i < 20 {
        guess = 0.5 * (guess + x / guess);
        i += 1;
    }

    guess
}

/// Rounds a floating-point number to the nearest integer as a `f64` in compile time.
const fn round_const(x: f64) -> f64 {
    if x >= 0.0 {
        (x + 0.5) as i64 as f64
    } else {
        (x - 0.5) as i64 as f64
    }
}

/// Approximates e^x using a 20-term Taylor series expansion.
///
/// Special case: returns 1.0 if `x` is 0.
///
/// Accurate to within **1e-9** absolute error threshold compared to f64::exp() over the range [-10`, 10].
pub const fn exp(x: f64) -> f64 {
    const LN_2: f64 = core::f64::consts::LN_2;

    if x == 0.0 {
        return 1.0;
    }

    // Compute n = round(x / ln(2))
    let n_float = round_const(x / LN_2);
    let n = n_float as i32;

    // Compute remainder r = x - n * ln(2)
    let r = x - (n_float * LN_2);

    // Since const fn can't use loops with mutable vars well in stable,
    // unroll manually:

    let r2 = r * r;
    let r3 = r2 * r;
    let r4 = r3 * r;
    let r5 = r4 * r;
    let r6 = r5 * r;
    let r7 = r6 * r;
    let r8 = r7 * r;
    let r9 = r8 * r;
    let r10 = r9 * r;
    let r11 = r10 * r;
    let r12 = r11 * r;
    let r13 = r12 * r;
    let r14 = r13 * r;
    let r15 = r14 * r;
    let r16 = r15 * r;
    let r17 = r16 * r;
    let r18 = r17 * r;
    let r19 = r18 * r;

    let taylor = 1.0
        + r
        + r2 / 2.0
        + r3 / 6.0
        + r4 / 24.0
        + r5 / 120.0
        + r6 / 720.0
        + r7 / 5040.0
        + r8 / 40320.0
        + r9 / 362880.0
        + r10 / 3628800.0
        + r11 / 39916800.0
        + r12 / 479001600.0
        + r13 / 6227020800.0
        + r14 / 87178291200.0
        + r15 / 1307674368000.0
        + r16 / 20922789888000.0
        + r17 / 355687428096000.0
        + r18 / 6402373705728000.0
        + r19 / 121645100408832000.0;

    powi(2.0, n) * taylor
}
