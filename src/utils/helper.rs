///utility to convert the transpose the matrix
///takes an input of a matrix of shape MxN, and returns the matrix as NxM
pub fn transpose<const NUM_ROWS: usize, const NUM_COLUMNS: usize>(
    matrix: &[[f64; NUM_COLUMNS]; NUM_ROWS],
) -> [[f64; NUM_ROWS]; NUM_COLUMNS] {
    let mut result = [[0.0; NUM_ROWS]; NUM_COLUMNS];

    for row_index in 0..NUM_COLUMNS {
        for col_index in 0..NUM_ROWS {
            result[row_index][col_index] = matrix[col_index][row_index];
        }
    }

    return result;
}
