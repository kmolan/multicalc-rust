use multicalc::error::LinalgError;
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc_testkit::tol::{
    assert_identity, assert_matrix_close, svd_moore_penrose, svd_moore_penrose_f32,
    svd_reconstructs,
};

#[test]
fn svd_reconstructs_various() {
    svd_reconstructs(
        Matrix::<3, 2>::new([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        1e-12,
    );
    svd_reconstructs(
        Matrix::<3, 3>::new([[4.0, 1.0, 2.0], [1.0, 5.0, 3.0], [2.0, 3.0, 6.0]]),
        1e-12,
    );
    svd_reconstructs(
        Matrix::<4, 3>::new([
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 1.0],
            [4.0, 1.0, 0.0],
            [2.0, 1.0, 5.0],
        ]),
        1e-12,
    );
    // Larger, well-conditioned tall matrices.
    svd_reconstructs(
        Matrix::<12, 6>::from_fn(|row, column| {
            if row == column {
                10.0
            } else {
                1.0 / (1.0 + (row + column) as f64)
            }
        }),
        1e-12,
    );
    svd_reconstructs(
        Matrix::<20, 6>::from_fn(|row, column| {
            if row == column {
                8.0
            } else {
                (row as f64 - column as f64) / (5.0 + (row + column) as f64)
            }
        }),
        1e-12,
    );
    // The same code at f32.
    svd_reconstructs(
        Matrix::<3, 3, f32>::new([[4.0, 1.0, 2.0], [1.0, 5.0, 3.0], [2.0, 3.0, 6.0]]),
        1e-4,
    );
}

#[test]
fn svd_singular_values() {
    // A symmetric matrix built from a known spectrum: A = R·diag(σ)·Rᵀ with R a proper rotation,
    // so the singular values are exactly [6, 3, 1].
    let rotation = Matrix::<3, 3>::new([
        [2.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0],
        [1.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0],
        [2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0],
    ]);
    let spectrum = Matrix::<3, 3>::new([[6.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]]);
    let singular_values = (rotation * spectrum * rotation.transpose())
        .svd()
        .unwrap()
        .singular_values();
    for (index, expected) in [6.0, 3.0, 1.0].into_iter().enumerate() {
        assert!((singular_values[index] - expected).abs() < 1e-12);
    }

    // Diagonal input: singular values are the sorted absolute diagonal.
    let diagonal_matrix = Matrix::<4, 4>::from_fn(|row, column| {
        if row == column {
            [3.0, -5.0, 2.0, -1.0][row]
        } else {
            0.0
        }
    });
    let singular_values = diagonal_matrix.svd().unwrap().singular_values();
    for (index, expected) in [5.0, 3.0, 2.0, 1.0].into_iter().enumerate() {
        assert!((singular_values[index] - expected).abs() < 1e-12);
    }
}

#[test]
fn svd_pseudo_inverse_conditions() {
    svd_moore_penrose(
        Matrix::<3, 2>::new([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        1e-10,
    );
    svd_moore_penrose(
        Matrix::<2, 3>::new([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]]),
        1e-10,
    );
    svd_moore_penrose(
        Matrix::<3, 3>::new([[4.0, 1.0, 2.0], [1.0, 5.0, 3.0], [2.0, 3.0, 6.0]]),
        1e-10,
    );
    svd_moore_penrose(
        Matrix::<12, 6>::from_fn(|row, column| {
            if row == column {
                10.0
            } else {
                1.0 / (1.0 + (row + column) as f64)
            }
        }),
        1e-10,
    );
}

#[test]
fn svd_f32_pseudo_inverse_conditions() {
    svd_moore_penrose_f32(Matrix::<3, 3, f32>::new([
        [4.0, 1.0, 2.0],
        [1.0, 5.0, 3.0],
        [2.0, 3.0, 6.0],
    ]));
    svd_moore_penrose_f32(Matrix::<4, 2, f32>::new([
        [1.0, 0.0],
        [2.0, -1.0],
        [0.5, 3.0],
        [-1.0, 2.0],
    ]));
    svd_moore_penrose_f32(Matrix::<2, 4, f32>::new([
        [1.0, 0.0, 2.0, -1.0],
        [0.0, 1.0, 1.0, 3.0],
    ]));
    // The second column is twice the first, and the third is their sum.
    svd_moore_penrose_f32(Matrix::<3, 3, f32>::new([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 6.0, 9.0],
    ]));
}

#[test]
fn svd_rank_deficient() {
    // Column 2 is twice column 1: rank 1, one exactly-zero singular value.
    let rank_one = Matrix::<3, 2>::new([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]);
    let factorization = rank_one.svd().unwrap();
    assert_eq!(factorization.rank(1e-9), 1);
    assert!(factorization.condition_number().is_infinite());

    // Truncated pseudo-inverse stays finite.
    let pseudo_inverse = factorization.pseudo_inverse();
    for row in 0..2 {
        for column in 0..3 {
            assert!(pseudo_inverse[(row, column)].is_finite());
        }
    }

    // Consistent system: the min-norm solution reproduces the range and equals A⁺·b.
    let right_hand_side = rank_one * Vector::new([1.0, 0.5]);
    let solution = factorization.solve(right_hand_side);
    assert!((rank_one * solution - right_hand_side).norm() < 1e-10);
    let pseudo_inverse_solution = pseudo_inverse * right_hand_side;
    for index in 0..2 {
        assert!((solution[index] - pseudo_inverse_solution[index]).abs() < 1e-12);
    }
}

#[test]
fn svd_error_paths() {
    // A wide matrix is rejected by the raw svd().
    let wide = Matrix::<2, 3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    assert_eq!(wide.svd().err(), Some(LinalgError::Underdetermined));

    // Non-finite entries are rejected.
    let mut with_nan = Matrix::<3, 2>::new([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    with_nan[(2, 1)] = f64::NAN;
    assert_eq!(with_nan.svd().err(), Some(LinalgError::NonFinite));
    let mut with_infinity = Matrix::<3, 2>::new([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    with_infinity[(0, 0)] = f64::INFINITY;
    assert_eq!(with_infinity.svd().err(), Some(LinalgError::NonFinite));
}

#[test]
fn svd_kabsch_rotation_recovery() {
    // A proper rotation (orthonormal, determinant +1).
    let rotation = Matrix::<3, 3>::new([
        [2.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0],
        [1.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0],
        [2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0],
    ]);
    // Body points spanning 3D.
    let body_points = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [2.0, -1.0, 0.5],
        [-1.0, 0.5, 2.0],
    ];
    // Cross-covariance H = Σ (R·p) pᵀ.
    let mut cross_covariance = Matrix::<3, 3>::zeros();
    for body_point in body_points {
        let point = Vector::new(body_point);
        let rotated_point = rotation * point;
        for row in 0..3 {
            for column in 0..3 {
                cross_covariance[(row, column)] += rotated_point[row] * point[column];
            }
        }
    }
    let factorization = cross_covariance.svd().unwrap();
    let u = factorization.u();
    let v = factorization.v();
    let mut recovered_rotation = u * v.transpose();
    // Reflection fix: guarantee a proper rotation.
    if recovered_rotation.determinant() < 0.0 {
        let mut flipped_u = u;
        for row in 0..3 {
            flipped_u[(row, 2)] = -flipped_u[(row, 2)];
        }
        recovered_rotation = flipped_u * v.transpose();
    }
    // Recovered rotation matches, is orthonormal, and has determinant +1.
    assert_matrix_close(recovered_rotation, rotation, 1e-9);
    assert_identity(recovered_rotation.transpose() * recovered_rotation, 1e-12);
    assert!((recovered_rotation.determinant() - 1.0).abs() < 1e-9);
}

#[test]
fn svd_redundant_jacobian_pseudo_inverse() {
    // A 6x7 manipulator Jacobian (7 joints -> 6-D task twist), full row rank.
    let jacobian = Matrix::<6, 7>::from_fn(|row, column| {
        if column < 6 {
            if row == column {
                2.0
            } else {
                0.3 / (1.0 + (row + column) as f64)
            }
        } else {
            0.5 * (row as f64 + 1.0)
        }
    });
    let pseudo_inverse = jacobian.pseudo_inverse().unwrap();

    // Moore–Penrose: J·J⁺·J == J and J·J⁺ symmetric.
    assert_matrix_close(jacobian * pseudo_inverse * jacobian, jacobian, 1e-9);
    let jacobian_times_pseudo_inverse = jacobian * pseudo_inverse;
    assert_matrix_close(
        jacobian_times_pseudo_inverse,
        jacobian_times_pseudo_inverse.transpose(),
        1e-12,
    );

    // Minimum-norm resolution: J⁺·v beats any other solution of J·x = v.
    let task_twist = Vector::<6>::from_fn(|index| index as f64 - 2.5);
    let minimum_norm_solution = pseudo_inverse * task_twist;
    let null_space_offset = Vector::<7>::from_fn(|index| (index as f64 - 3.0) * 0.7);
    let projected_offset = (pseudo_inverse * jacobian) * null_space_offset;
    let alternative_solution = Vector::<7>::from_fn(|index| {
        minimum_norm_solution[index] + null_space_offset[index] - projected_offset[index]
    });
    assert!((jacobian * minimum_norm_solution - task_twist).norm() < 1e-9);
    assert!((jacobian * alternative_solution - task_twist).norm() < 1e-9);
    assert!(minimum_norm_solution.norm() <= alternative_solution.norm() + 1e-9);
}

#[test]
fn svd_near_singular_jacobian() {
    // A 6x6 Jacobian at a near-singularity: column 5 nearly equals column 4.
    let mut jacobian = Matrix::<6, 6>::from_fn(|row, column| {
        if row == column {
            1.0 + 0.1 * column as f64
        } else {
            0.2 / (1.0 + (row + column) as f64)
        }
    });
    for row in 0..6 {
        let column_four_entry = jacobian[(row, 4)];
        jacobian[(row, 5)] = column_four_entry + 1e-8 * (row as f64 + 1.0);
    }
    let factorization = jacobian.svd().unwrap();
    let singular_values = factorization.singular_values();
    let tolerance = 1e-4 * singular_values[0];

    assert!(factorization.condition_number() > 1e6);
    assert_eq!(factorization.rank(tolerance), 5);

    // Truncated pseudo-inverse and solve stay finite (no blow-up on the tiny σ).
    let pseudo_inverse = factorization.pseudo_inverse_tol(tolerance);
    for row in 0..6 {
        for column in 0..6 {
            assert!(pseudo_inverse[(row, column)].is_finite());
        }
    }
    let solution = factorization.solve(Vector::from_fn(|index| index as f64 + 1.0));
    for index in 0..6 {
        assert!(solution[index].is_finite());
    }
}

#[test]
fn svd_overdetermined_least_squares() {
    // Fit a plane z = a·x + b·y + c to 30 sampled points; solve vs the normal equations.
    let sample_point = |index: usize| ((index as f64 * 0.37).sin(), (index as f64 * 0.53).cos());
    let design = Matrix::<30, 3>::from_fn(|row, column| {
        let (x, y) = sample_point(row);
        match column {
            0 => x,
            1 => y,
            _ => 1.0,
        }
    });
    let rhs = Vector::<30>::from_fn(|index| {
        let (x, y) = sample_point(index);
        0.5 * x - 1.2 * y + 2.0 + 1e-3 * (index as f64 * 1.7).sin()
    });
    let svd_solution = design.svd().unwrap().solve(rhs);
    let normal_equations_solution = (design.transpose() * design)
        .solve(design.transpose() * rhs)
        .unwrap();
    for index in 0..3 {
        assert!((svd_solution[index] - normal_equations_solution[index]).abs() < 1e-9);
    }
}
