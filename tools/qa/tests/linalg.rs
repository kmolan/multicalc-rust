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
#[must_use]
fn f32_tolerance(fixture: &Fixture) -> Tol {
    fixture
        .tolerances
        .f32
        .expect("linalg fixtures all carry an f32 tolerance")
}

// ----- LU -----

fn run_lu<const N: usize>(fixture: &Fixture) {
    let a = to_matrix::<N, N>(&fixture.inputs["A"]);
    let b = to_vector::<N>(&fixture.inputs["b"]);
    let tolerance = fixture.tolerances.f64;

    let f = a.lu_decompose().unwrap();
    assert_scalar(f.determinant(), &fixture.expected["det"], tolerance, "det");
    assert_vector(&f.solve(b), &fixture.expected["x"], tolerance, "x");
    assert_matrix(&f.inverse(), &fixture.expected["inv"], tolerance, "inv");
    assert_matrix_close(&(a * f.inverse()), &Matrix::identity(), tolerance, "A*inv");

    // f32 identity only.
    let a32 = to_matrix_f32::<N, N>(&fixture.inputs["A"]);
    let inv32 = a32.lu_decompose().unwrap().inverse();
    assert_matrix_close_f32(
        &(a32 * inv32),
        &Matrix::identity(),
        f32_tolerance(fixture),
        "A*inv f32",
    );
}

#[test]
fn test_lu_decompose() {
    for fixture in load_dir("linalg") {
        if fixture.inputs["decomp"].as_str() != "lu" {
            continue;
        }
        let (rows, _) = fixture.inputs["A"].shape();
        match rows {
            3 => run_lu::<3>(&fixture),
            4 => run_lu::<4>(&fixture),
            5 => run_lu::<5>(&fixture),
            n => panic!("unregistered lu shape {n}"),
        }
    }
}

// ----- column-pivoted QR -----

fn run_qr<const R: usize, const C: usize>(fixture: &Fixture) {
    let a = to_matrix::<R, C>(&fixture.inputs["A"]);
    let b = to_vector::<R>(&fixture.inputs["b"]);
    let tolerance = fixture.tolerances.f64;

    let factorization = PivotedQr::decompose(a).unwrap();
    let x = factorization.solve_least_squares(b).unwrap();
    assert_vector(&x, &fixture.expected["x_ls"], tolerance, "x_ls");
    assert_scalar(
        (a * x - b).norm(),
        &fixture.expected["residual_norm"],
        tolerance,
        "residual_norm",
    );

    // Self-identities from multicalc's own factors.
    let (orthogonal, triangular, perm) = (
        factorization.orthogonal(),
        factorization.triangular(),
        factorization.permutation(),
    );
    assert_matrix_close(
        &(orthogonal.transpose() * orthogonal),
        &Matrix::identity(),
        tolerance,
        "QtQ",
    );
    let permuted = Matrix::<R, C>::from_fn(|i, col| a[(i, perm[col])]);
    assert_matrix_close(&(orthogonal * triangular), &permuted, tolerance, "Q*R=A*P");

    // f32 identity only: reconstruct A*P.
    let a32 = to_matrix_f32::<R, C>(&fixture.inputs["A"]);
    let qr32 = PivotedQr::decompose(a32).unwrap();
    let (orthogonal32, triangular32, perm32) =
        (qr32.orthogonal(), qr32.triangular(), qr32.permutation());
    let permuted32 = Matrix::<R, C, f32>::from_fn(|i, col| a32[(i, perm32[col])]);
    assert_matrix_close_f32(
        &(orthogonal32 * triangular32),
        &permuted32,
        f32_tolerance(fixture),
        "Q*R=A*P f32",
    );
}

#[test]
#[allow(clippy::min_ident_chars)]
fn qr() {
    for fixture in load_dir("linalg") {
        if fixture.inputs["decomp"].as_str() != "qr" {
            continue;
        }
        let (rows, cols) = fixture.inputs["A"].shape();
        match (rows, cols) {
            (3, 2) => run_qr::<3, 2>(&fixture),
            (4, 3) => run_qr::<4, 3>(&fixture),
            (3, 3) => run_qr::<3, 3>(&fixture),
            (20, 7) => run_qr::<20, 7>(&fixture),
            shape => panic!("unregistered qr shape {shape:?}"),
        }
    }
}

// ----- SVD -----

fn run_svd<const R: usize, const C: usize>(fixture: &Fixture) {
    let a = to_matrix::<R, C>(&fixture.inputs["A"]);
    let b = to_vector::<R>(&fixture.inputs["b"]);
    let tolerance = fixture.tolerances.f64;

    let f = a.svd().unwrap();
    assert_vector(
        &f.singular_values(),
        &fixture.expected["singular_values"],
        tolerance,
        "singular_values",
    );
    assert_vector(&f.solve(b), &fixture.expected["x_ls"], tolerance, "x_ls");

    let pinv = a.pseudo_inverse().unwrap();
    assert_matrix(&pinv, &fixture.expected["pinv"], tolerance, "pinv");

    // Self-identity: A = U*diag(s)*Vt.
    let (left, sigma, right) = (f.left(), f.singular_values(), f.right());
    let recon = Matrix::<R, C>::from_fn(|row, col| {
        (0..C)
            .map(|k| left[(row, k)] * sigma[k] * right[(col, k)])
            .sum()
    });
    assert_matrix_close(&recon, &a, tolerance, "U*S*Vt");

    // Four Moore-Penrose conditions on the unique pseudo-inverse.
    assert_matrix_close(&(a * pinv * a), &a, tolerance, "A*Ap*A");
    assert_matrix_close(&(pinv * a * pinv), &pinv, tolerance, "Ap*A*Ap");
    let aap = a * pinv;
    assert_matrix_close(&aap, &aap.transpose(), tolerance, "(A*Ap) symmetric");
    let apa = pinv * a;
    assert_matrix_close(&apa, &apa.transpose(), tolerance, "(Ap*A) symmetric");

    // f32 identity only: reconstruct A.
    let a32 = to_matrix_f32::<R, C>(&fixture.inputs["A"]);
    let f32 = a32.svd().unwrap();
    let (left32, sigma32, right32) = (f32.left(), f32.singular_values(), f32.right());
    let recon32 = Matrix::<R, C, f32>::from_fn(|row, col| {
        (0..C)
            .map(|k| left32[(row, k)] * sigma32[k] * right32[(col, k)])
            .sum()
    });
    assert_matrix_close_f32(&recon32, &a32, f32_tolerance(fixture), "U*S*Vt f32");
}

#[test]
fn svd() {
    for fixture in load_dir("linalg") {
        if fixture.inputs["decomp"].as_str() != "svd" {
            continue;
        }
        let (rows, cols) = fixture.inputs["A"].shape();
        match (rows, cols) {
            (3, 2) => run_svd::<3, 2>(&fixture),
            (3, 3) => run_svd::<3, 3>(&fixture),
            (4, 3) => run_svd::<4, 3>(&fixture),
            (12, 6) => run_svd::<12, 6>(&fixture),
            (20, 6) => run_svd::<20, 6>(&fixture),
            shape => panic!("unregistered svd shape {shape:?}"),
        }
    }
}

// ----- Cholesky -----

fn run_cholesky<const N: usize>(fixture: &Fixture) {
    let a = to_matrix::<N, N>(&fixture.inputs["A"]);
    let b = to_vector::<N>(&fixture.inputs["b"]);
    let tolerance = fixture.tolerances.f64;

    let f = a.cholesky().unwrap();
    assert_matrix(&f.lower(), &fixture.expected["L"], tolerance, "L"); // unique for positive diagonal
    assert_scalar(f.determinant(), &fixture.expected["det"], tolerance, "det");
    assert_vector(&f.solve(b), &fixture.expected["x"], tolerance, "x");
    assert_matrix_close(
        &(f.lower() * f.lower().transpose()),
        &a,
        tolerance,
        "L*Lt=A",
    );

    // f32 identity only.
    let a32 = to_matrix_f32::<N, N>(&fixture.inputs["A"]);
    let lower32 = a32.cholesky().unwrap().lower();
    assert_matrix_close_f32(
        &(lower32 * lower32.transpose()),
        &a32,
        f32_tolerance(fixture),
        "L*Lt=A f32",
    );
}

#[test]
fn cholesky() {
    for fixture in load_dir("linalg") {
        if fixture.inputs["decomp"].as_str() != "cholesky" {
            continue;
        }
        let (rows, _) = fixture.inputs["A"].shape();
        match rows {
            2 => run_cholesky::<2>(&fixture),
            3 => run_cholesky::<3>(&fixture),
            4 => run_cholesky::<4>(&fixture),
            n => panic!("unregistered cholesky shape {n}"),
        }
    }
}

// ----- symmetric eigendecomposition -----

fn run_symmetric_eigen<const N: usize>(fixture: &Fixture) {
    let a = to_matrix::<N, N>(&fixture.inputs["A"]);
    let tolerance = fixture.tolerances.f64;

    let f = a.symmetric_eigendecomposition().unwrap();
    assert_vector(
        &f.eigenvalues(),
        &fixture.expected["eigenvalues"],
        tolerance,
        "eigenvalues",
    );
    assert_scalar(f.determinant(), &fixture.expected["det"], tolerance, "det");
    assert_scalar(
        f.condition_number(),
        &fixture.expected["condition_number"],
        tolerance,
        "condition_number",
    );

    // Self-identities from multicalc's own factors.
    let (values, vectors) = (f.eigenvalues(), f.eigenvectors());
    let recon = Matrix::<N, N>::from_fn(|i, j| {
        (0..N)
            .map(|k| vectors[(i, k)] * values[k] * vectors[(j, k)])
            .sum()
    });
    assert_matrix_close(&recon, &a, tolerance, "V*L*Vt");
    assert_matrix_close(
        &(vectors.transpose() * vectors),
        &Matrix::identity(),
        tolerance,
        "VtV",
    );

    // f32 identity only.
    let a32 = to_matrix_f32::<N, N>(&fixture.inputs["A"]);
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
    assert_matrix_close_f32(&recon32, &a32, f32_tolerance(fixture), "V*L*Vt f32");
}

#[test]
fn symmetric_eigen() {
    for fixture in load_dir("linalg") {
        if fixture.inputs["decomp"].as_str() != "symmetric_eigen" {
            continue;
        }
        let (rows, _) = fixture.inputs["A"].shape();
        match rows {
            3 => run_symmetric_eigen::<3>(&fixture),
            4 => run_symmetric_eigen::<4>(&fixture),
            6 => run_symmetric_eigen::<6>(&fixture),
            n => panic!("unregistered symmetric eigen shape {n}"),
        }
    }
}
