use num_complex::ComplexFloat;

///utility to convert the transpose the matrix
///takes an input of a matrix of shape MxN, and returns the matrix as NxM
pub fn transpose<T: ComplexFloat, const NUM_ROWS: usize, const NUM_COLUMNS: usize>(matrix: &[[T; NUM_COLUMNS]; NUM_ROWS]) -> [[T; NUM_ROWS]; NUM_COLUMNS]
{
    let mut result = [[T::zero(); NUM_ROWS]; NUM_COLUMNS];

    for row_index in 0..NUM_COLUMNS
    {
        for col_index in 0..NUM_ROWS
        {
            result[row_index][col_index] = matrix[col_index][row_index];
        }
    }

    return result;
}