use std::time::Duration;

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use multicalc::linear_algebra::{Matrix, Vector};

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
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    targets = vector, matrix
}
criterion_main!(benches);
