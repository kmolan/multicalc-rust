use multicalc::error::LinalgError;
use multicalc::linear_algebra::{
    Matrix, MatrixView, MatrixViewMut, Vector, VectorView, VectorViewMut,
};
use proptest::prelude::*;

// A strategy for producing matrices in property-based tests.
fn matrix_strategy<const ROWS: usize, const COLS: usize, S>(
    num_strategy: S,
) -> impl Strategy<Value = Matrix<ROWS, COLS>>
where
    S: Strategy<Value = f64>,
{
    prop::array::uniform::<_, ROWS>(prop::array::uniform::<_, COLS>(num_strategy))
        .prop_map(Matrix::new)
}

fn counted<const ROWS: usize, const COLS: usize>() -> Matrix<ROWS, COLS> {
    Matrix::from_fn(|row, column| (row * COLS + column) as f64)
}

// ----- taking a view -----

#[test]
fn view_round_trips_through_to_matrix() {
    let matrix = counted::<3, 4>();

    assert_eq!(matrix.view().to_matrix(), matrix);
    assert_eq!(matrix.view().strides(), (4, 1));
    assert_eq!((matrix.view().rows(), matrix.view().cols()), (3, 4));
    assert!(matrix.view().is_row_major());
}

#[test]
fn view_reads_the_same_entries_as_the_matrix() {
    let matrix = counted::<3, 4>();
    let view = matrix.view();

    for row in 0..3 {
        for column in 0..4 {
            assert_eq!(view.try_get(row, column).ok(), matrix.get(row, column));
        }
    }
    assert_eq!(view.try_get(3, 0), Err(LinalgError::OutOfBounds));
    assert_eq!(view.try_get(0, 4), Err(LinalgError::OutOfBounds));
}

#[test]
fn from_row_major_slice_checks_the_length() {
    let buffer = [1.0, 2.0, 3.0, 4.0, 5.0];

    let view = MatrixView::<2, 2>::try_from_row_major_slice(&buffer).unwrap();
    assert_eq!(view.to_matrix().into_array(), [[1.0, 2.0], [3.0, 4.0]]);

    // A longer buffer is fine — the tail is simply not part of the view.
    assert!(MatrixView::<1, 5>::try_from_row_major_slice(&buffer).is_ok());
    assert_eq!(
        MatrixView::<3, 2>::try_from_row_major_slice(&buffer).unwrap_err(),
        LinalgError::OutOfBounds
    );
    assert!(VectorView::<6>::try_from_slice(&buffer).is_err());
}

// ----- transpose -----

#[test]
fn transposed_only_swaps_the_strides() {
    let matrix = counted::<2, 3>();
    let transposed = matrix.view().transposed();

    assert_eq!(transposed.strides(), (1, 3));
    assert_eq!((transposed.rows(), transposed.cols()), (3, 2));
    assert!(!transposed.is_row_major());
}

#[test]
fn transposed_agrees_with_the_owning_transpose() {
    let matrix = counted::<2, 3>();

    assert_eq!(matrix.view().transposed().to_matrix(), matrix.transpose());
}

#[test]
fn transposing_twice_returns_the_original_layout() {
    let matrix = counted::<2, 3>();
    let there_and_back = matrix.view().transposed().transposed();

    assert_eq!(there_and_back.strides(), matrix.view().strides());
    assert_eq!(there_and_back, matrix.view());
}

#[test]
fn views_of_different_layouts_compare_by_element() {
    let matrix = counted::<2, 3>();
    let owned_transpose = matrix.transpose();

    // Same entries, different strides and different buffers.
    assert_eq!(matrix.view().transposed(), owned_transpose.view());
    assert_ne!(matrix.view().transposed(), counted::<3, 2>().view());
}

#[test]
fn writable_views_compare_by_element_too() {
    let mut left = counted::<2, 3>();
    let mut right = counted::<2, 3>();
    let mut different = Matrix::<2, 3>::zeros();

    assert_eq!(left.view_mut(), right.view_mut());
    assert_ne!(left.view_mut(), different.view_mut());
}

// ----- submatrix -----

#[test]
fn submatrix_reads_the_requested_block() {
    let matrix = counted::<3, 3>();

    let block = matrix.view().try_submatrix::<2, 2>(1, 1).unwrap();
    assert_eq!(block.to_matrix().into_array(), [[4.0, 5.0], [7.0, 8.0]]);
    // Only the offset moved; the strides still describe the parent.
    assert_eq!(block.strides(), (3, 1));
}

#[test]
fn submatrix_rejects_a_block_that_runs_off_an_edge() {
    let matrix = counted::<3, 3>();

    assert_eq!(
        matrix.view().try_submatrix::<2, 2>(2, 0).unwrap_err(),
        LinalgError::OutOfBounds
    );
    assert!(matrix.view().try_submatrix::<2, 2>(0, 2).is_err());
    assert!(matrix.view().try_submatrix::<4, 1>(0, 0).is_err());
    assert!(matrix.view().try_submatrix::<1, 1>(usize::MAX, 0).is_err());
}

#[test]
fn submatrix_composes_with_transpose() {
    let matrix = counted::<3, 4>();

    // Transposing then blocking must equal blocking then transposing.
    let transposed_first = matrix
        .view()
        .transposed()
        .try_submatrix::<2, 2>(1, 1)
        .unwrap();
    let blocked_first = matrix
        .view()
        .try_submatrix::<2, 2>(1, 1)
        .unwrap()
        .transposed();

    assert_eq!(transposed_first, blocked_first);
    assert_eq!(
        transposed_first.to_matrix(),
        matrix.try_submatrix::<2, 2>(1, 1).unwrap().transpose()
    );
}

#[test]
fn owning_submatrix_matches_the_view() {
    let matrix = counted::<3, 3>();

    assert_eq!(
        matrix.try_submatrix::<2, 2>(0, 1).unwrap(),
        matrix
            .view()
            .try_submatrix::<2, 2>(0, 1)
            .unwrap()
            .to_matrix()
    );
    assert_eq!(
        matrix.try_submatrix::<2, 2>(2, 2).unwrap_err(),
        LinalgError::OutOfBounds
    );
}

#[test]
fn set_submatrix_writes_a_block_and_reports_a_bad_corner() {
    let mut matrix = Matrix::<3, 3>::zeros();
    let block = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);

    assert_eq!(matrix.try_set_submatrix(1, 1, block.view()), Ok(()));
    assert_eq!(
        matrix.into_array(),
        [[0.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 3.0, 4.0]]
    );

    // A corner that would hang the block off an edge is reported, and nothing is written.
    let before = matrix;
    assert_eq!(
        matrix.try_set_submatrix(2, 2, block.view()),
        Err(LinalgError::OutOfBounds)
    );
    assert_eq!(matrix, before);
}

// ----- rows, columns, diagonals, segments -----

#[test]
fn row_and_column_views_match_the_copying_accessors() {
    let matrix = counted::<3, 4>();

    for row in 0..3 {
        assert_eq!(
            matrix.view().try_row(row).unwrap().to_vector(),
            matrix.try_row(row).unwrap()
        );
        assert_eq!(matrix.view().try_row(row).unwrap().stride(), 1);
    }
    for column in 0..4 {
        assert_eq!(
            matrix.view().try_column(column).unwrap().to_vector(),
            matrix.try_column(column).unwrap()
        );
        // A column of a row-major matrix is strided, which is what avoids the copy.
        assert_eq!(matrix.view().try_column(column).unwrap().stride(), 4);
    }
    assert!(matrix.view().try_row(3).is_err());
    assert!(matrix.view().try_column(4).is_err());
}

#[test]
fn a_transposed_view_swaps_rows_and_columns() {
    let matrix = counted::<2, 3>();

    assert_eq!(
        matrix.view().transposed().try_row(0).unwrap(),
        matrix.view().try_column(0).unwrap()
    );
}

#[test]
fn diagonal_view_walks_the_shorter_side() {
    let square = counted::<3, 3>();
    assert_eq!(
        square.view().try_diagonal::<3>().unwrap().to_vector(),
        Vector::new([0.0, 4.0, 8.0])
    );

    let wide = counted::<2, 4>();
    assert_eq!(
        wide.view().try_diagonal::<2>().unwrap().to_vector(),
        Vector::new([0.0, 5.0])
    );
    // The length has to be the shorter side.
    assert_eq!(
        wide.view().try_diagonal::<4>().unwrap_err(),
        LinalgError::OutOfBounds
    );
}

#[test]
fn a_writable_diagonal_writes_the_same_entries_the_read_only_one_reads() {
    let mut matrix = Matrix::<3, 3>::zeros();
    matrix.view_mut().try_diagonal::<3>().unwrap().fill(1.0);

    assert_eq!(matrix, Matrix::<3, 3>::identity());
    assert_eq!(
        matrix.view().try_diagonal::<3>().unwrap().to_vector(),
        Vector::new([1.0, 1.0, 1.0])
    );

    let mut wide = counted::<2, 4>();
    assert!(wide.view_mut().try_diagonal::<4>().is_err());
    wide.view_mut().try_diagonal::<2>().unwrap().fill(9.0);
    assert_eq!(
        wide.into_array(),
        [[9.0, 1.0, 2.0, 3.0], [4.0, 9.0, 6.0, 7.0]]
    );
}

#[test]
fn segment_narrows_a_vector_view() {
    let vector = Vector::new([1.0, 2.0, 3.0, 4.0]);

    assert_eq!(
        vector.view().try_segment::<2>(1).unwrap().to_vector(),
        Vector::new([2.0, 3.0])
    );
    assert_eq!(
        vector.view().try_segment::<3>(2).unwrap_err(),
        LinalgError::OutOfBounds
    );
    assert!(vector.view().try_segment::<1>(usize::MAX).is_err());
}

#[test]
fn a_writable_segment_narrows_the_same_way() {
    let mut vector = Vector::new([1.0, 2.0, 3.0, 4.0]);

    vector.view_mut().try_segment::<2>(1).unwrap().fill(0.0);
    assert_eq!(vector.into_array(), [1.0, 0.0, 0.0, 4.0]);

    assert!(vector.view_mut().try_segment::<3>(2).is_err());
    assert!(vector.view_mut().try_segment::<1>(usize::MAX).is_err());
}

#[test]
fn dot_works_across_two_different_strides() {
    let matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);

    // Column 0 is strided, row 0 is contiguous: 1*1 + 3*2.
    let column = matrix.view().try_column(0).unwrap();
    let row = matrix.view().try_row(0).unwrap();
    assert_eq!(column.dot(row), 7.0);
    assert_eq!(row.dot(column), 7.0);
}

#[test]
fn a_writable_view_dots_without_giving_up_its_write_access() {
    let mut vector = Vector::new([3.0, 4.0]);
    let mut view = vector.view_mut();

    assert_eq!(view.dot(Vector::new([1.0, 0.0]).view()), 3.0);
    // The borrow survived the dot, so the view is still writable.
    *view.try_get_mut(1).unwrap() = 0.0;

    assert_eq!(vector.into_array(), [3.0, 0.0]);
}

// ----- writing through a view -----

#[test]
fn writes_through_a_view_land_in_the_matrix() {
    let mut matrix = Matrix::<2, 2>::zeros();

    *matrix.view_mut().try_get_mut(0, 1).unwrap() = 5.0;
    assert_eq!(matrix[(0, 1)], 5.0);

    *matrix.view_mut().try_get_mut(1, 0).unwrap() = 6.0;
    assert_eq!(matrix[(1, 0)], 6.0);

    assert_eq!(
        matrix.view_mut().try_get_mut(2, 0).unwrap_err(),
        LinalgError::OutOfBounds
    );
}

#[test]
fn writing_through_a_transposed_view_hits_the_mirrored_entry() {
    let mut matrix = Matrix::<2, 3>::zeros();

    *matrix.view_mut().transposed().try_get_mut(2, 1).unwrap() = 9.0;

    assert_eq!(matrix[(1, 2)], 9.0);
}

#[test]
fn writing_through_a_submatrix_view_stays_inside_the_block() {
    let mut matrix = Matrix::<3, 3>::zeros();

    matrix
        .view_mut()
        .try_submatrix::<2, 2>(1, 1)
        .unwrap()
        .fill(1.0);

    assert_eq!(
        matrix.into_array(),
        [[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]
    );
}

#[test]
fn writing_through_a_column_view_walks_the_stride() {
    let mut matrix = Matrix::<3, 2>::zeros();

    matrix.view_mut().try_column(1).unwrap().fill(7.0);

    assert_eq!(matrix.into_array(), [[0.0, 7.0], [0.0, 7.0], [0.0, 7.0]]);
}

#[test]
fn reborrow_keeps_the_original_view_usable() {
    let mut matrix = Matrix::<2, 2>::zeros();
    let mut view = matrix.view_mut();

    *view.reborrow().transposed().try_get_mut(0, 1).unwrap() = 5.0;
    *view.try_get_mut(0, 0).unwrap() = 1.0;

    assert_eq!(matrix.into_array(), [[1.0, 0.0], [5.0, 0.0]]);
}

#[test]
fn copy_from_transposes_into_a_caller_buffer_without_an_intermediate() {
    let matrix = counted::<2, 3>();
    let mut scratch = [0.0; 6];

    let mut destination = MatrixViewMut::<3, 2>::try_from_row_major_slice(&mut scratch).unwrap();
    destination.copy_from(matrix.view().transposed());

    assert_eq!(scratch, [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
}

#[test]
fn vector_view_mut_copies_across_strides() {
    let mut matrix = Matrix::<3, 2>::zeros();
    let source = Vector::new([1.0, 2.0, 3.0]);

    matrix
        .view_mut()
        .try_column(0)
        .unwrap()
        .copy_from(source.view());

    assert_eq!(matrix.into_array(), [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]);
}

#[test]
fn vector_view_mut_writes_into_a_bare_slice() {
    let mut buffer = [0.0; 4];

    let mut view = VectorViewMut::<3>::try_from_slice(&mut buffer).unwrap();
    *view.try_get_mut(0).unwrap() = 1.0;
    view.copy_from(Vector::new([1.0, 2.0, 3.0]).view());
    assert_eq!(view.to_vector(), Vector::new([1.0, 2.0, 3.0]));

    // The fourth element is outside the view and stays untouched.
    assert_eq!(buffer, [1.0, 2.0, 3.0, 0.0]);
    assert!(VectorViewMut::<5>::try_from_slice(&mut buffer).is_err());
}

// ----- disjoint splits -----

#[test]
fn split_rows_at_yields_two_simultaneously_writable_halves() {
    let mut matrix = Matrix::<3, 2>::zeros();

    let (mut top, mut bottom) = matrix.view_mut().try_split_rows_at::<1, 2>().unwrap();
    *top.try_get_mut(0, 0).unwrap() = 1.0;
    *bottom.try_get_mut(0, 1).unwrap() = 2.0;
    *bottom.try_get_mut(1, 0).unwrap() = 3.0;

    assert_eq!(matrix.into_array(), [[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]]);
}

#[test]
fn split_rows_at_reads_the_same_blocks_as_submatrix() {
    let matrix = counted::<4, 3>();
    let mut copy = matrix;

    let (top, bottom) = copy.view_mut().try_split_rows_at::<1, 3>().unwrap();

    assert_eq!(top.to_matrix(), matrix.try_submatrix::<1, 3>(0, 0).unwrap());
    assert_eq!(
        bottom.to_matrix(),
        matrix.try_submatrix::<3, 3>(1, 0).unwrap()
    );
}

#[test]
fn split_rows_at_allows_an_empty_half() {
    let mut matrix = counted::<2, 2>();

    let (top, bottom) = matrix.view_mut().try_split_rows_at::<0, 2>().unwrap();
    assert_eq!(top.rows(), 0);
    assert_eq!(bottom.to_matrix(), counted::<2, 2>());

    let (top, bottom) = matrix.view_mut().try_split_rows_at::<2, 0>().unwrap();
    assert_eq!(top.to_matrix(), counted::<2, 2>());
    assert_eq!(bottom.rows(), 0);
}

#[test]
fn split_rows_at_rejects_a_partition_that_does_not_add_up() {
    let mut matrix = Matrix::<3, 2>::zeros();

    assert!(matrix.view_mut().try_split_rows_at::<1, 1>().is_err());
    assert!(matrix.view_mut().try_split_rows_at::<2, 2>().is_err());
    assert!(matrix.view_mut().try_split_rows_at::<0, 0>().is_err());
}

#[test]
fn split_rows_at_rejects_a_view_whose_rows_interleave() {
    let mut matrix = Matrix::<3, 3>::zeros();

    // A transposed view's rows are stripes through the buffer, so no cut of the slice
    // separates them and the split has to refuse rather than hand out aliasing views.
    assert!(!matrix.view_mut().transposed().is_row_major());
    assert_eq!(
        matrix
            .view_mut()
            .transposed()
            .try_split_rows_at::<1, 2>()
            .unwrap_err(),
        LinalgError::OutOfBounds
    );
}

#[test]
fn a_single_column_view_stays_splittable_however_it_was_reshaped() {
    // Transposing a one-row matrix gives a Cx1 view whose column stride is the parent's width.
    // That stride addresses nothing when there is one column, and the elements are still a
    // contiguous run, so the split must not refuse it.
    let mut matrix = counted::<1, 3>();
    let transposed = matrix.view_mut().transposed();
    assert_eq!(transposed.strides(), (1, 3));
    assert!(transposed.is_row_major());

    let (top, bottom) = transposed.try_split_rows_at::<1, 2>().unwrap();
    assert_eq!(top.to_matrix().into_array(), [[0.0]]);
    assert_eq!(bottom.to_matrix().into_array(), [[1.0], [2.0]]);
}

#[test]
fn split_rows_at_works_on_a_submatrix_whose_rows_stay_contiguous() {
    let mut matrix = counted::<4, 4>();

    let block = matrix.view_mut().try_submatrix::<3, 2>(1, 1).unwrap();
    assert!(block.is_row_major());

    let (top, bottom) = block.try_split_rows_at::<1, 2>().unwrap();
    assert_eq!(top.to_matrix().into_array(), [[5.0, 6.0]]);
    assert_eq!(bottom.to_matrix().into_array(), [[9.0, 10.0], [13.0, 14.0]]);
}

#[test]
fn split_cols_at_yields_two_simultaneously_writable_halves() {
    let mut matrix = Matrix::<3, 2>::zeros();

    // Columns interleave in row-major storage, so the writable column split needs the
    // column-major view that `transposed` hands back.
    let (mut left, mut right) = matrix
        .view_mut()
        .transposed()
        .try_split_cols_at::<1, 2>()
        .unwrap();
    *left.try_get_mut(0, 0).unwrap() = 1.0;
    *right.try_get_mut(1, 1).unwrap() = 2.0;

    assert_eq!(matrix.into_array(), [[1.0, 0.0], [0.0, 0.0], [0.0, 2.0]]);
}

#[test]
fn split_cols_at_rejects_a_row_major_view_and_a_bad_partition() {
    let mut matrix = Matrix::<2, 3>::zeros();

    // Row-major: the two column blocks interleave, so no cut of the slice separates them.
    assert_eq!(
        matrix.view_mut().try_split_cols_at::<1, 2>().unwrap_err(),
        LinalgError::OutOfBounds
    );
    assert!(
        matrix
            .view_mut()
            .transposed()
            .try_split_cols_at::<1, 2>()
            .is_err()
    );
}

#[test]
fn read_only_splits_need_no_layout_and_cover_every_entry() {
    let matrix = counted::<2, 3>();

    let (top, bottom) = matrix.view().try_split_rows_at::<1, 1>().unwrap();
    assert_eq!(top.to_matrix().into_array(), [[0.0, 1.0, 2.0]]);
    assert_eq!(bottom.to_matrix().into_array(), [[3.0, 4.0, 5.0]]);

    let (left, right) = matrix.view().try_split_cols_at::<1, 2>().unwrap();
    assert_eq!(left.to_matrix().into_array(), [[0.0], [3.0]]);
    assert_eq!(right.to_matrix().into_array(), [[1.0, 2.0], [4.0, 5.0]]);

    // A transposed view interleaves its rows, which stops the writable split but not this one.
    let transposed = matrix.view().transposed();
    assert!(!transposed.is_row_major());
    let (top, bottom) = transposed.try_split_rows_at::<1, 2>().unwrap();
    assert_eq!(top.to_matrix().into_array(), [[0.0, 3.0]]);
    assert_eq!(bottom.to_matrix().into_array(), [[1.0, 4.0], [2.0, 5.0]]);

    assert!(matrix.view().try_split_rows_at::<1, 2>().is_err());
    assert!(matrix.view().try_split_cols_at::<1, 1>().is_err());
}

#[test]
fn vector_split_at_yields_two_simultaneously_writable_halves() {
    let mut vector = Vector::<4>::zeros();

    let (mut head, mut tail) = vector.view_mut().try_split_at::<1, 3>().unwrap();
    *head.try_get_mut(0).unwrap() = 1.0;
    *tail.try_get_mut(2).unwrap() = 4.0;

    assert_eq!(vector.into_array(), [1.0, 0.0, 0.0, 4.0]);
}

#[test]
fn vector_split_at_rejects_a_bad_partition_or_a_strided_view() {
    let mut vector = Vector::<4>::zeros();
    assert_eq!(
        vector.view_mut().try_split_at::<1, 1>().unwrap_err(),
        LinalgError::OutOfBounds
    );

    let mut matrix = Matrix::<3, 2>::zeros();
    let column = matrix.view_mut().try_column(0).unwrap();
    assert_eq!(column.stride(), 2);
    assert!(column.try_split_at::<1, 2>().is_err());
}

#[test]
fn a_read_only_vector_split_works_at_any_stride() {
    let matrix = counted::<3, 2>();
    let column = matrix.view().try_column(0).unwrap();
    assert_eq!(column.stride(), 2);

    let (head, tail) = column.try_split_at::<1, 2>().unwrap();
    assert_eq!(head.to_vector(), Vector::new([0.0]));
    assert_eq!(tail.to_vector(), Vector::new([2.0, 4.0]));

    assert!(column.try_split_at::<1, 1>().is_err());
}

// ----- empty shapes -----

#[test]
fn empty_shapes_are_viewable_and_never_dereference() {
    let empty: Matrix<0, 3> = Matrix::zeros();
    let view = empty.view();

    assert_eq!(view.rows(), 0);
    assert_eq!(view.try_get(0, 0), Err(LinalgError::OutOfBounds));
    assert_eq!(view.to_matrix(), empty);
    assert_eq!(view.try_column(0).unwrap().len(), 0);
    assert!(view.try_column(3).is_err());

    let no_columns: Matrix<3, 0> = Matrix::zeros();
    assert_eq!(
        no_columns.view().transposed().to_matrix(),
        Matrix::<0, 3>::zeros()
    );
    assert!(Vector::<0>::zeros().view().is_empty());
}

// ----- out-of-range subscripts -----

#[test]
fn a_subscript_past_the_edge_is_rejected_even_when_it_lands_inside_the_buffer() {
    // The reason the views check row/column against ROWS/COLS instead of letting the slice
    // bound decide. This 2x2 block carries the parent's strides (3, 1), so the invalid
    // subscript (2, 0) computes flat index 2*3 + 0 = 6 -- a real element of the 9-element
    // parent buffer. A raw slice index would hand back parent entry (2, 0) and report nothing.
    let matrix = Matrix::<3, 3>::from_fn(|row, column| (row * 3 + column) as f64);
    let block = matrix.view().try_submatrix::<2, 2>(0, 0).unwrap();

    assert_eq!(block.strides(), (3, 1));
    assert_eq!(
        matrix[(2, 0)],
        6.0,
        "the entry a slice-only bound would have returned"
    );
    assert_eq!(
        block.try_get(2, 0),
        Err(LinalgError::OutOfBounds),
        "the view reports it instead"
    );
}

#[test]
fn every_out_of_range_subscript_is_reported_rather_than_panicking() {
    // The views have no `Index`, so none of these can abort: each one is an ordinary value the
    // caller can match on.
    let matrix = counted::<2, 3>();
    assert_eq!(matrix.view().try_get(2, 0), Err(LinalgError::OutOfBounds));
    assert_eq!(
        matrix.view().transposed().try_get(0, 2),
        Err(LinalgError::OutOfBounds),
        "in range for the parent buffer, out of range for the 3x2 view"
    );

    let mut writable = Matrix::<2, 2>::zeros();
    assert_eq!(
        writable.view_mut().try_get_mut(0, 2),
        Err(LinalgError::OutOfBounds)
    );

    let vector = Vector::new([1.0, 2.0]);
    assert_eq!(vector.view().try_get(2), Err(LinalgError::OutOfBounds));

    let mut writable_vector = Vector::new([1.0, 2.0]);
    assert_eq!(
        writable_vector.view_mut().try_get_mut(2),
        Err(LinalgError::OutOfBounds)
    );
}

// ----- genericity -----

#[test]
fn views_work_over_f32() {
    let matrix: Matrix<2, 3, f32> = Matrix::from_fn(|row, column| (row * 3 + column) as f32);

    assert_eq!(matrix.view().transposed().to_matrix(), matrix.transpose());
    assert_eq!(
        matrix.view().try_column(2).unwrap().to_vector(),
        Vector::new([2.0f32, 5.0])
    );

    let mut scratch = [0.0f32; 6];
    let mut written = MatrixViewMut::<3, 2, f32>::try_from_row_major_slice(&mut scratch).unwrap();
    written.copy_from(matrix.view().transposed());
    assert_eq!(written.to_matrix(), matrix.transpose());
}

#[test]
fn views_work_over_a_non_numeric_element_type() {
    // The read-only view machinery only needs `Copy` to materialize, and nothing at all to
    // reshape, so it is not tied to the numeric tower.
    let matrix = Matrix::new([['a', 'b'], ['c', 'd']]);

    assert_eq!(matrix.view().transposed().try_get(0, 1), Ok(&'c'));
    assert_eq!(
        matrix.view().transposed().to_matrix().into_array(),
        [['a', 'c'], ['b', 'd']]
    );
}

// ----- non-finite entries -----

#[test]
fn views_pass_non_finite_entries_through_untouched() {
    // The property tests below draw from `NORMAL`, since a reshaped NaN cannot be compared by
    // equality. Reshaping still has to carry one through to the right place.
    let matrix = Matrix::new([[f64::NAN, 1.0], [f64::INFINITY, f64::NEG_INFINITY]]);
    let transposed = matrix.view().transposed();

    assert!(transposed.try_get(0, 0).unwrap().is_nan());
    assert_eq!(transposed.try_get(0, 1), Ok(&f64::INFINITY));
    assert_eq!(transposed.try_get(1, 0), Ok(&1.0));
    assert_eq!(transposed.try_get(1, 1), Ok(&f64::NEG_INFINITY));
    assert!(matrix.view().try_column(0).unwrap().to_vector()[0].is_nan());
}

// ----- properties -----

fn check_transposed_view_matches_owned<const ROWS: usize, const COLS: usize>(
    matrix: Matrix<ROWS, COLS>,
) -> Result<(), TestCaseError> {
    let view = matrix.view().transposed();
    prop_assert_eq!(view.to_matrix(), matrix.transpose());
    prop_assert_eq!(view.transposed().to_matrix(), matrix);
    for row in 0..COLS {
        for column in 0..ROWS {
            prop_assert_eq!(view.try_get(row, column).ok(), matrix.get(column, row));
        }
    }
    Ok(())
}

fn check_submatrix_matches_manual_indexing<const ROWS: usize, const COLS: usize>(
    matrix: Matrix<ROWS, COLS>,
    top: usize,
    left: usize,
) -> Result<(), TestCaseError> {
    let Ok(block) = matrix.view().try_submatrix::<2, 2>(top, left) else {
        prop_assert!(top + 2 > ROWS || left + 2 > COLS);
        return Ok(());
    };
    for row in 0..2 {
        for column in 0..2 {
            prop_assert_eq!(
                block.try_get(row, column).ok(),
                matrix.get(top + row, left + column)
            );
        }
    }
    Ok(())
}

fn check_row_and_column_views<const ROWS: usize, const COLS: usize>(
    matrix: Matrix<ROWS, COLS>,
) -> Result<(), TestCaseError> {
    for row in 0..ROWS {
        prop_assert_eq!(
            matrix.view().try_row(row).unwrap().to_vector(),
            matrix.try_row(row).unwrap()
        );
    }
    for column in 0..COLS {
        prop_assert_eq!(
            matrix.view().try_column(column).unwrap().to_vector(),
            matrix.try_column(column).unwrap()
        );
    }
    Ok(())
}

fn check_split_rows_covers_every_entry<const ROWS: usize, const COLS: usize, const TOP: usize>(
    matrix: Matrix<ROWS, COLS>,
) -> Result<(), TestCaseError>
where
    Matrix<ROWS, COLS>: Copy,
{
    let mut scratch = matrix;
    let Ok((top, bottom)) = scratch.view_mut().try_split_rows_at::<TOP, 1>() else {
        prop_assert_ne!(TOP + 1, ROWS);
        return Ok(());
    };
    for row in 0..TOP {
        for column in 0..COLS {
            prop_assert_eq!(top.try_get(row, column).ok(), matrix.get(row, column));
        }
    }
    for column in 0..COLS {
        prop_assert_eq!(bottom.try_get(0, column).ok(), matrix.get(TOP, column));
    }
    Ok(())
}

fn check_split_cols_covers_every_entry<const ROWS: usize, const COLS: usize, const LEFT: usize>(
    matrix: Matrix<ROWS, COLS>,
) -> Result<(), TestCaseError> {
    let Ok((left, right)) = matrix.view().try_split_cols_at::<LEFT, 1>() else {
        prop_assert_ne!(LEFT + 1, COLS);
        return Ok(());
    };
    for row in 0..ROWS {
        for column in 0..LEFT {
            prop_assert_eq!(left.try_get(row, column).ok(), matrix.get(row, column));
        }
        prop_assert_eq!(right.try_get(row, 0).ok(), matrix.get(row, LEFT));
    }
    Ok(())
}

proptest! {
    #[test]
    fn transposed_view_matches_owned_2x3(matrix in matrix_strategy::<2, 3, _>(prop::num::f64::NORMAL)) {
        check_transposed_view_matches_owned(matrix)?;
    }

    #[test]
    fn transposed_view_matches_owned_3x2(matrix in matrix_strategy::<3, 2, _>(prop::num::f64::NORMAL)) {
        check_transposed_view_matches_owned(matrix)?;
    }

    #[test]
    fn transposed_view_matches_owned_4x4(matrix in matrix_strategy::<4, 4, _>(prop::num::f64::NORMAL)) {
        check_transposed_view_matches_owned(matrix)?;
    }

    #[test]
    fn transposed_view_matches_owned_1x5(matrix in matrix_strategy::<1, 5, _>(prop::num::f64::NORMAL)) {
        check_transposed_view_matches_owned(matrix)?;
    }

    #[test]
    fn submatrix_matches_manual_indexing_4x4(
        matrix in matrix_strategy::<4, 4, _>(prop::num::f64::NORMAL),
        top in 0usize..6,
        left in 0usize..6,
    ) {
        check_submatrix_matches_manual_indexing(matrix, top, left)?;
    }

    #[test]
    fn submatrix_matches_manual_indexing_3x5(
        matrix in matrix_strategy::<3, 5, _>(prop::num::f64::NORMAL),
        top in 0usize..5,
        left in 0usize..7,
    ) {
        check_submatrix_matches_manual_indexing(matrix, top, left)?;
    }

    #[test]
    fn row_and_column_views_3x4(matrix in matrix_strategy::<3, 4, _>(prop::num::f64::NORMAL)) {
        check_row_and_column_views(matrix)?;
    }

    #[test]
    fn row_and_column_views_4x2(matrix in matrix_strategy::<4, 2, _>(prop::num::f64::NORMAL)) {
        check_row_and_column_views(matrix)?;
    }

    #[test]
    fn split_rows_covers_every_entry_4x3(matrix in matrix_strategy::<4, 3, _>(prop::num::f64::NORMAL)) {
        check_split_rows_covers_every_entry::<4, 3, 3>(matrix)?;
    }

    #[test]
    fn split_rows_covers_every_entry_2x5(matrix in matrix_strategy::<2, 5, _>(prop::num::f64::NORMAL)) {
        check_split_rows_covers_every_entry::<2, 5, 1>(matrix)?;
    }

    #[test]
    fn split_cols_covers_every_entry_4x3(matrix in matrix_strategy::<4, 3, _>(prop::num::f64::NORMAL)) {
        check_split_cols_covers_every_entry::<4, 3, 2>(matrix)?;
    }

    #[test]
    fn split_cols_covers_every_entry_2x5(matrix in matrix_strategy::<2, 5, _>(prop::num::f64::NORMAL)) {
        check_split_cols_covers_every_entry::<2, 5, 4>(matrix)?;
    }
}
