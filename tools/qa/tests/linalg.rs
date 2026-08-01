#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks each decomposition against numpy/LAPACK goldens.
//!
//! Cross-implementation comparisons use only gauge-free quantities (determinant,
//! inverse, solve, least-squares solution, residual norm, singular values,
//! pseudo-inverse, the unique Cholesky factor, and a symmetric matrix's
//! eigenvalues, determinant, and condition number). Raw Q/R/U/V and eigenvector
//! matrices are verified only through multicalc's own reconstruction identities,
//! never against numpy's factors. The f64 result carries the golden; f32 re-runs
//! the same input and checks a mathematical identity only.

use multicalc::linear_algebra::{Matrix, PivotedQr};
use multicalc_qa::load::*;
use multicalc_qa::schema::*;

/// The f32 tolerance, which every linalg fixture carries.
fn f32_tolerance(fx: &Fixture) -> Tol {
    fx.tolerances
        .f32
        .expect("linalg fixtures all carry an f32 tolerance")
}

// ----- LU -----

fn run_lu<const N: usize>(fx: &Fixture) {
    let a = to_matrix::<N, N>(&fx.inputs["A"]);
    let b = to_vector::<N>(&fx.inputs["b"]);
    let t = fx.tolerances.f64;

    let f = a.lu().unwrap();
    assert_scalar(f.determinant(), &fx.expected["det"], t, "det");
    assert_vector(&f.solve(b), &fx.expected["x"], t, "x");
    assert_matrix(&f.inverse(), &fx.expected["inv"], t, "inv");
    assert_matrix_close(&(a * f.inverse()), &Matrix::identity(), t, "A*inv");

    // f32 identity only.
    let a32 = to_matrix_f32::<N, N>(&fx.inputs["A"]);
    let inv32 = a32.lu().unwrap().inverse();
    assert_matrix_close_f32(
        &(a32 * inv32),
        &Matrix::identity(),
        f32_tolerance(fx),
        "A*inv f32",
    );
}

#[test]
fn lu() {
    for fx in load_dir("linalg") {
        if fx.inputs["decomp"].as_str() != "lu" {
            continue;
        }
        let (rows, _) = fx.inputs["A"].shape();
        match rows {
            3 => run_lu::<3>(&fx),
            4 => run_lu::<4>(&fx),
            5 => run_lu::<5>(&fx),
            n => panic!("unregistered lu shape {n}"),
        }
    }
}

// ----- column-pivoted QR -----

fn run_qr<const R: usize, const C: usize>(fx: &Fixture) {
    let a = to_matrix::<R, C>(&fx.inputs["A"]);
    let b = to_vector::<R>(&fx.inputs["b"]);
    let t = fx.tolerances.f64;

    let qr = PivotedQr::decompose(a).unwrap();
    let x = qr.solve_least_squares(b).unwrap();
    assert_vector(&x, &fx.expected["x_ls"], t, "x_ls");
    assert_scalar(
        (a * x - b).norm(),
        &fx.expected["residual_norm"],
        t,
        "residual_norm",
    );

    // Self-identities from multicalc's own factors.
    let (q, r, perm) = (qr.q(), qr.r(), qr.permutation());
    assert_matrix_close(&(q.transpose() * q), &Matrix::identity(), t, "QtQ");
    let ap = Matrix::<R, C>::from_fn(|i, c| a[(i, perm[c])]);
    assert_matrix_close(&(q * r), &ap, t, "Q*R=A*P");

    // f32 identity only: reconstruct A*P.
    let a32 = to_matrix_f32::<R, C>(&fx.inputs["A"]);
    let qr32 = PivotedQr::decompose(a32).unwrap();
    let (q32, r32, perm32) = (qr32.q(), qr32.r(), qr32.permutation());
    let ap32 = Matrix::<R, C, f32>::from_fn(|i, c| a32[(i, perm32[c])]);
    assert_matrix_close_f32(&(q32 * r32), &ap32, f32_tolerance(fx), "Q*R=A*P f32");
}

#[test]
fn qr() {
    for fx in load_dir("linalg") {
        if fx.inputs["decomp"].as_str() != "qr" {
            continue;
        }
        let (rows, cols) = fx.inputs["A"].shape();
        match (rows, cols) {
            (3, 2) => run_qr::<3, 2>(&fx),
            (4, 3) => run_qr::<4, 3>(&fx),
            (3, 3) => run_qr::<3, 3>(&fx),
            (20, 7) => run_qr::<20, 7>(&fx),
            s => panic!("unregistered qr shape {s:?}"),
        }
    }
}

// ----- SVD -----

fn run_svd<const R: usize, const C: usize>(fx: &Fixture) {
    let a = to_matrix::<R, C>(&fx.inputs["A"]);
    let b = to_vector::<R>(&fx.inputs["b"]);
    let t = fx.tolerances.f64;

    let f = a.svd().unwrap();
    assert_vector(
        &f.singular_values(),
        &fx.expected["singular_values"],
        t,
        "singular_values",
    );
    assert_vector(&f.solve(b), &fx.expected["x_ls"], t, "x_ls");

    let pinv = a.pseudo_inverse().unwrap();
    assert_matrix(&pinv, &fx.expected["pinv"], t, "pinv");

    // Self-identity: A = U*diag(s)*Vt.
    let (u, s, v) = (f.u(), f.singular_values(), f.v());
    let recon = Matrix::<R, C>::from_fn(|i, j| (0..C).map(|k| u[(i, k)] * s[k] * v[(j, k)]).sum());
    assert_matrix_close(&recon, &a, t, "U*S*Vt");

    // Four Moore-Penrose conditions on the unique pseudo-inverse.
    assert_matrix_close(&(a * pinv * a), &a, t, "A*Ap*A");
    assert_matrix_close(&(pinv * a * pinv), &pinv, t, "Ap*A*Ap");
    let aap = a * pinv;
    assert_matrix_close(&aap, &aap.transpose(), t, "(A*Ap) symmetric");
    let apa = pinv * a;
    assert_matrix_close(&apa, &apa.transpose(), t, "(Ap*A) symmetric");

    // f32 identity only: reconstruct A.
    let a32 = to_matrix_f32::<R, C>(&fx.inputs["A"]);
    let f32 = a32.svd().unwrap();
    let (u32, s32, v32) = (f32.u(), f32.singular_values(), f32.v());
    let recon32 = Matrix::<R, C, f32>::from_fn(|i, j| {
        (0..C).map(|k| u32[(i, k)] * s32[k] * v32[(j, k)]).sum()
    });
    assert_matrix_close_f32(&recon32, &a32, f32_tolerance(fx), "U*S*Vt f32");
}

#[test]
fn svd() {
    for fx in load_dir("linalg") {
        if fx.inputs["decomp"].as_str() != "svd" {
            continue;
        }
        let (rows, cols) = fx.inputs["A"].shape();
        match (rows, cols) {
            (3, 2) => run_svd::<3, 2>(&fx),
            (3, 3) => run_svd::<3, 3>(&fx),
            (4, 3) => run_svd::<4, 3>(&fx),
            (12, 6) => run_svd::<12, 6>(&fx),
            (20, 6) => run_svd::<20, 6>(&fx),
            s => panic!("unregistered svd shape {s:?}"),
        }
    }
}

// ----- Cholesky -----

fn run_cholesky<const N: usize>(fx: &Fixture) {
    let a = to_matrix::<N, N>(&fx.inputs["A"]);
    let b = to_vector::<N>(&fx.inputs["b"]);
    let t = fx.tolerances.f64;

    let f = a.cholesky().unwrap();
    assert_matrix(&f.l(), &fx.expected["L"], t, "L"); // unique for positive diagonal
    assert_scalar(f.determinant(), &fx.expected["det"], t, "det");
    assert_vector(&f.solve(b), &fx.expected["x"], t, "x");
    assert_matrix_close(&(f.l() * f.l().transpose()), &a, t, "L*Lt=A");

    // f32 identity only.
    let a32 = to_matrix_f32::<N, N>(&fx.inputs["A"]);
    let l32 = a32.cholesky().unwrap().l();
    assert_matrix_close_f32(
        &(l32 * l32.transpose()),
        &a32,
        f32_tolerance(fx),
        "L*Lt=A f32",
    );
}

#[test]
fn cholesky() {
    for fx in load_dir("linalg") {
        if fx.inputs["decomp"].as_str() != "cholesky" {
            continue;
        }
        let (rows, _) = fx.inputs["A"].shape();
        match rows {
            2 => run_cholesky::<2>(&fx),
            3 => run_cholesky::<3>(&fx),
            4 => run_cholesky::<4>(&fx),
            n => panic!("unregistered cholesky shape {n}"),
        }
    }
}

// ----- symmetric eigendecomposition -----

fn run_symmetric_eigen<const N: usize>(fx: &Fixture) {
    let a = to_matrix::<N, N>(&fx.inputs["A"]);
    let t = fx.tolerances.f64;

    let f = a.symmetric_eigendecomposition().unwrap();
    assert_vector(
        &f.eigenvalues(),
        &fx.expected["eigenvalues"],
        t,
        "eigenvalues",
    );
    assert_scalar(f.determinant(), &fx.expected["det"], t, "det");
    assert_scalar(
        f.condition_number(),
        &fx.expected["condition_number"],
        t,
        "condition_number",
    );

    // Self-identities from multicalc's own factors.
    let (values, vectors) = (f.eigenvalues(), f.eigenvectors());
    let recon = Matrix::<N, N>::from_fn(|i, j| {
        (0..N)
            .map(|k| vectors[(i, k)] * values[k] * vectors[(j, k)])
            .sum()
    });
    assert_matrix_close(&recon, &a, t, "V*L*Vt");
    assert_matrix_close(
        &(vectors.transpose() * vectors),
        &Matrix::identity(),
        t,
        "VtV",
    );

    // f32 identity only.
    let a32 = to_matrix_f32::<N, N>(&fx.inputs["A"]);
    let f32_decomposition = a32.symmetric_eigendecomposition().unwrap();
    let (values32, vectors32) = (
        f32_decomposition.eigenvalues(),
        f32_decomposition.eigenvectors(),
    );
    let recon32 = Matrix::<N, N, f32>::from_fn(|i, j| {
        (0..N)
            .map(|k| vectors32[(i, k)] * values32[k] * vectors32[(j, k)])
            .sum()
    });
    assert_matrix_close_f32(&recon32, &a32, f32_tolerance(fx), "V*L*Vt f32");
}

#[test]
fn symmetric_eigen() {
    for fx in load_dir("linalg") {
        if fx.inputs["decomp"].as_str() != "symmetric_eigen" {
            continue;
        }
        let (rows, _) = fx.inputs["A"].shape();
        match rows {
            3 => run_symmetric_eigen::<3>(&fx),
            4 => run_symmetric_eigen::<4>(&fx),
            6 => run_symmetric_eigen::<6>(&fx),
            n => panic!("unregistered symmetric eigen shape {n}"),
        }
    }
}
