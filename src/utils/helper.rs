///utility to convert the transpose the matrix
///takes an input of a matrix of shape MxN, and returns the matrix as NxM
pub const fn transpose<const ROWS: usize, const COLS: usize>(
    matrix: &[[f64; COLS]; ROWS],
) -> [[f64; ROWS]; COLS] {
    let mut result = [[0.0; ROWS]; COLS];
    let mut c = 0;
    while c < COLS {
        let mut r = 0;
        while r < ROWS {
            result[c][r] = matrix[r][c];
            r += 1;
        }
        c += 1;
    }
    result
}
