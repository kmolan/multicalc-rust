use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::{matrix, vector};

#[test]
fn vector_macro_matches_new() {
    assert_eq!(vector![1.0, 2.0, 3.0], Vector::new([1.0, 2.0, 3.0]));

    // trailing comma is accepted
    assert_eq!(vector![1.0, 2.0,], Vector::new([1.0, 2.0,]));

    // single element
    assert_eq!(vector![42.0], Vector::new([42.0]));

    // repeat form
    assert_eq!(vector![0.0; 3], Vector::<3>::zeros());
    assert_eq!(vector![1.0; 3], Vector::new([1.0, 1.0, 1.0]));
}

#[test]
fn matrix_macro_matches_new() {
    assert_eq!(
        matrix![[1.0, 2.0], [3.0, 4.0]],
        Matrix::new([[1.0, 2.0], [3.0, 4.0]])
    );

    assert_eq!(matrix![[1.0, 2.0, 3.0]], Matrix::new([[1.0, 2.0, 3.0]]));

    assert_eq!(
        matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    );

    // repeat form
    assert_eq!(
        matrix![[1.0, 2.0, 3.0]; 2],
        Matrix::new([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    );
    assert_eq!(
        matrix![[0.0, 0.0,]; 2],
        Matrix::new([[0.0, 0.0], [0.0, 0.0]])
    );

    // nested repeat form
    assert_eq!(matrix![[0.0; 3]; 4], Matrix::<4, 3>::zeros());
    assert_eq!(matrix![[1.0; 2]; 2], Matrix::new([[1.0, 1.0], [1.0, 1.0]]));
}
