use crate::utils::helper;

#[test]
fn test_transpose_1() 
{
    let original = [[1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 6.0, 8.0], [1.0, 6.0, 9.0, 12.0]];

    let trans = helper::transpose(&original);

    for i in 0..original.len()
    {
        for j in 0..original[0].len()
        {
            assert!(original[i][j] == trans[j][i]);
        }
    }
}