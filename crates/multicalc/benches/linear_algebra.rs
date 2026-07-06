#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::Duration;

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use multicalc::linear_algebra::{Matrix, PivotedQr, Vector};

fn vector(crit: &mut Criterion) {
    let u: Vector<3> = Vector::new([1.0, 2.0, 3.0]);
    let v: Vector<3> = Vector::new([4.0, 5.0, 6.0]);
    let w: Vector<3> = Vector::new([7.0, 8.0, 9.0]);
    let p: Vector<2> = Vector::new([1.0, 2.0]);
    let q: Vector<2> = Vector::new([3.0, 4.0]);

    crit.bench_function("vector/dot_3", |b| {
        b.iter(|| black_box(u).dot(black_box(v)))
    });
    crit.bench_function("vector/cross_3", |b| {
        b.iter(|| black_box(u).cross(black_box(v)))
    });
    crit.bench_function("vector/cross_2d", |b| {
        b.iter(|| black_box(p).cross(black_box(q)))
    });
    crit.bench_function("vector/scalar_triple_3", |b| {
        b.iter(|| black_box(u).scalar_triple(black_box(v), black_box(w)))
    });
    crit.bench_function("vector/norm_3", |b| b.iter(|| black_box(u).norm()));
    crit.bench_function("vector/add_3", |b| b.iter(|| black_box(u) + black_box(v)));
    crit.bench_function("vector/scale_3", |b| {
        b.iter(|| black_box(u) * black_box(2.0))
    });
}

fn matrix(crit: &mut Criterion) {
    let a: Matrix<3, 3> = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]);
    let b3: Matrix<3, 3> = Matrix::new([[2.0, 0.0, 1.0], [1.0, 3.0, 2.0], [0.0, 1.0, 4.0]]);
    let a4: Matrix<4, 4> = Matrix::from_fn(|r, c| (r * 4 + c + 1) as f64);
    let b4: Matrix<4, 4> = Matrix::from_fn(|r, c| (r + c) as f64);
    let inv4: Matrix<4, 4> = Matrix::new([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 1.0, 0.0, 1.0],
        [0.0, 3.0, 1.0, 2.0],
        [1.0, 0.0, 2.0, 1.0],
    ]);
    let v: Vector<3> = Vector::new([1.0, 2.0, 3.0]);

    crit.bench_function("matrix/matmul_3x3", |b| {
        b.iter(|| black_box(a) * black_box(b3))
    });
    crit.bench_function("matrix/matmul_4x4", |b| {
        b.iter(|| black_box(a4) * black_box(b4))
    });
    crit.bench_function("matrix/mat_vec_3x3", |b| {
        b.iter(|| black_box(a) * black_box(v))
    });
    crit.bench_function("matrix/transpose_3x3", |b| {
        b.iter(|| black_box(a).transpose())
    });
    crit.bench_function("matrix/add_3x3", |b| {
        b.iter(|| black_box(a) + black_box(b3))
    });
    crit.bench_function("matrix/determinant_3x3", |b| {
        b.iter(|| black_box(a).determinant())
    });
    crit.bench_function("matrix/inverse_3x3", |b| b.iter(|| black_box(a).inverse()));
    crit.bench_function("matrix/inverse_4x4", |b| {
        b.iter(|| black_box(inv4).inverse())
    });
}

fn lu(crit: &mut Criterion) {
    // Well-conditioned, diagonally dominant systems.
    let a4: Matrix<4, 4> = Matrix::from_fn(|i, j| if i == j { 10.0 } else { (i + j) as f64 });
    let a8: Matrix<8, 8> = Matrix::from_fn(|i, j| {
        if i == j {
            20.0
        } else {
            1.0 / (i + j + 1) as f64
        }
    });
    let b8: Vector<8> = Vector::from_fn(|i| (i + 1) as f64);

    crit.bench_function("lu/decompose_4x4", |b| {
        b.iter(|| black_box(a4).lu().unwrap())
    });
    crit.bench_function("lu/decompose_8x8", |b| {
        b.iter(|| black_box(a8).lu().unwrap())
    });
    crit.bench_function("lu/decompose_solve_8x8", |b| {
        b.iter(|| black_box(a8).lu().unwrap().solve(black_box(b8)))
    });
}

fn cholesky(crit: &mut Criterion) {
    // Symmetric positive-definite (diagonally dominant, positive diagonal).
    let a4: Matrix<4, 4> = Matrix::from_fn(|i, j| if i == j { 5.0 } else { 1.0 });
    let a8: Matrix<8, 8> = Matrix::from_fn(|i, j| if i == j { 9.0 } else { 1.0 });
    let b8: Vector<8> = Vector::from_fn(|i| (i + 1) as f64);

    crit.bench_function("cholesky/decompose_4x4", |b| {
        b.iter(|| black_box(a4).cholesky().unwrap())
    });
    crit.bench_function("cholesky/decompose_8x8", |b| {
        b.iter(|| black_box(a8).cholesky().unwrap())
    });
    crit.bench_function("cholesky/decompose_solve_8x8", |b| {
        b.iter(|| black_box(a8).cholesky().unwrap().solve(black_box(b8)))
    });
}

fn qr(crit: &mut Criterion) {
    // Degree-6 polynomial least-squares fit over 20 nodes on [-1, 1] (Vandermonde design).
    let node = |i: usize| -1.0 + 2.0 * i as f64 / 19.0;
    let vandermonde = Matrix::<20, 7>::from_fn(|i, j| {
        let t = node(i);
        (0..j).fold(1.0, |acc, _| acc * t)
    });
    let vb = vandermonde * Vector::new([0.5, -1.2, 2.0, 0.3, -0.8, 1.1, -0.4]);
    crit.bench_function("qr/vandermonde_20x7_solve", |b| {
        b.iter(|| {
            PivotedQr::decompose(black_box(vandermonde))
                .unwrap()
                .solve_least_squares(black_box(vb))
                .unwrap()
        })
    });

    // Factorization of the famously ill-conditioned 8x8 Hilbert matrix.
    let hilbert = Matrix::<8, 8>::from_fn(|i, j| 1.0 / ((i + j + 1) as f64));
    crit.bench_function("qr/hilbert_8x8_decompose", |b| {
        b.iter(|| PivotedQr::decompose(black_box(hilbert)).unwrap())
    });

    // Ridge (Tikhonov) damped solve on a 15x8 Vandermonde design.
    let rnode = |i: usize| -1.0 + 2.0 * i as f64 / 14.0;
    let design = Matrix::<15, 8>::from_fn(|i, j| {
        let t = rnode(i);
        (0..j).fold(1.0, |acc, _| acc * t)
    });
    let rb = design * Vector::new([0.4, 1.0, -0.6, 0.9, -1.3, 0.5, 0.7, -0.2]);
    crit.bench_function("qr/ridge_15x8_damped_solve", |b| {
        b.iter(|| {
            PivotedQr::decompose(black_box(design))
                .unwrap()
                .into_damped(black_box(rb))
                .solve_with_diagonal(black_box(&[0.1; 8]))
        })
    });
}

fn svd(crit: &mut Criterion) {
    // Decomposition at robotics-relevant shapes.
    let a3: Matrix<3, 3> = Matrix::new([[4.0, 1.0, 2.0], [1.0, 5.0, 3.0], [2.0, 3.0, 6.0]]);
    let a6: Matrix<6, 6> = Matrix::from_fn(|i, j| {
        if i == j {
            6.0
        } else {
            1.0 / (i + j + 1) as f64
        }
    });
    let a8: Matrix<8, 8> = Matrix::from_fn(|i, j| {
        if i == j {
            8.0
        } else {
            1.0 / (i + j + 1) as f64
        }
    });
    // Tall Vandermonde design over 12 nodes on [-1, 1].
    let tall: Matrix<12, 6> = Matrix::from_fn(|i, j| {
        let t = -1.0 + 2.0 * i as f64 / 11.0;
        (0..j).fold(1.0, |acc, _| acc * t)
    });

    crit.bench_function("svd/decompose_3x3", |b| {
        b.iter(|| black_box(a3).svd().unwrap())
    });
    crit.bench_function("svd/decompose_6x6", |b| {
        b.iter(|| black_box(a6).svd().unwrap())
    });
    crit.bench_function("svd/decompose_8x8", |b| {
        b.iter(|| black_box(a8).svd().unwrap())
    });
    crit.bench_function("svd/decompose_12x6", |b| {
        b.iter(|| black_box(tall).svd().unwrap())
    });

    // Pseudo-inverse at a square and the redundant 6x7 (wide) shape.
    let j6: Matrix<6, 6> = Matrix::from_fn(|i, j| {
        if i == j {
            3.0
        } else {
            1.0 / (i + j + 1) as f64
        }
    });
    let j67: Matrix<6, 7> = Matrix::from_fn(|i, j| {
        if i == j {
            5.0
        } else {
            1.0 / (i + j + 1) as f64
        }
    });

    crit.bench_function("svd/pseudo_inverse_6x6", |b| {
        b.iter(|| black_box(j6).pseudo_inverse().unwrap())
    });
    crit.bench_function("svd/pseudo_inverse_6x7", |b| {
        b.iter(|| black_box(j67).pseudo_inverse().unwrap())
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    targets = vector, matrix, qr, lu, cholesky, svd
}
criterion_main!(benches);
