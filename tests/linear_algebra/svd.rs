use crate::helpers::{assert_close, assert_identity, svd_moore_penrose, svd_reconstructs};
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::utils::error_codes::CalcError;

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
        Matrix::<12, 6>::from_fn(|i, j| {
            if i == j {
                10.0
            } else {
                1.0 / (1.0 + (i + j) as f64)
            }
        }),
        1e-12,
    );
    svd_reconstructs(
        Matrix::<20, 6>::from_fn(|i, j| {
            if i == j {
                8.0
            } else {
                (i as f64 - j as f64) / (5.0 + (i + j) as f64)
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
    let r = Matrix::<3, 3>::new([
        [2.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0],
        [1.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0],
        [2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0],
    ]);
    let d = Matrix::<3, 3>::new([[6.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]]);
    let s = (r * d * r.transpose()).svd().unwrap().singular_values();
    for (k, want) in [6.0, 3.0, 1.0].into_iter().enumerate() {
        assert!((s[k] - want).abs() < 1e-12);
    }

    // Diagonal input: singular values are the sorted absolute diagonal.
    let diag = Matrix::<4, 4>::from_fn(|i, j| {
        if i == j {
            [3.0, -5.0, 2.0, -1.0][i]
        } else {
            0.0
        }
    });
    let s = diag.svd().unwrap().singular_values();
    for (k, want) in [5.0, 3.0, 2.0, 1.0].into_iter().enumerate() {
        assert!((s[k] - want).abs() < 1e-12);
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
        Matrix::<12, 6>::from_fn(|i, j| {
            if i == j {
                10.0
            } else {
                1.0 / (1.0 + (i + j) as f64)
            }
        }),
        1e-10,
    );
}

#[test]
fn svd_rank_deficient() {
    // Column 2 is twice column 1: rank 1, one exactly-zero singular value.
    let a = Matrix::<3, 2>::new([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]);
    let f = a.svd().unwrap();
    assert_eq!(f.rank(1e-9), 1);
    assert!(f.condition_number().is_infinite());

    // Truncated pseudo-inverse stays finite.
    let ap = f.pseudo_inverse();
    for r in 0..2 {
        for c in 0..3 {
            assert!(ap[(r, c)].is_finite());
        }
    }

    // Consistent system: the min-norm solution reproduces the range and equals A⁺·b.
    let b = a * Vector::new([1.0, 0.5]);
    let x = f.solve(b);
    assert!((a * x - b).norm() < 1e-10);
    let x_pinv = ap * b;
    for i in 0..2 {
        assert!((x[i] - x_pinv[i]).abs() < 1e-12);
    }
}

#[test]
fn svd_error_paths() {
    // A wide matrix is rejected by the raw svd().
    let wide = Matrix::<2, 3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    assert_eq!(wide.svd().err(), Some(CalcError::Underdetermined));

    // Non-finite entries are rejected.
    let mut nan = Matrix::<3, 2>::new([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    nan[(2, 1)] = f64::NAN;
    assert_eq!(nan.svd().err(), Some(CalcError::NonFiniteValue));
    let mut inf = Matrix::<3, 2>::new([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    inf[(0, 0)] = f64::INFINITY;
    assert_eq!(inf.svd().err(), Some(CalcError::NonFiniteValue));
}

#[test]
fn svd_kabsch_rotation_recovery() {
    // A proper rotation (orthonormal, determinant +1).
    let rot = Matrix::<3, 3>::new([
        [2.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0],
        [1.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0],
        [2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0],
    ]);
    // Body points spanning 3D.
    let pts = [
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
    let mut h = Matrix::<3, 3>::zeros();
    for p in pts {
        let pv = Vector::new(p);
        let q = rot * pv;
        for i in 0..3 {
            for j in 0..3 {
                h[(i, j)] += q[i] * pv[j];
            }
        }
    }
    let f = h.svd().unwrap();
    let (u, v) = (f.u(), f.v());
    let mut rhat = u * v.transpose();
    // Reflection fix: guarantee a proper rotation.
    if rhat.determinant() < 0.0 {
        let mut uf = u;
        for i in 0..3 {
            uf[(i, 2)] = -uf[(i, 2)];
        }
        rhat = uf * v.transpose();
    }
    // Recovered rotation matches, is orthonormal, and has determinant +1.
    assert_close(rhat, rot, 1e-9);
    assert_identity(rhat.transpose() * rhat, 1e-12);
    assert!((rhat.determinant() - 1.0).abs() < 1e-9);
}

#[test]
fn svd_redundant_jacobian_pseudo_inverse() {
    // A 6x7 manipulator Jacobian (7 joints -> 6-D task twist), full row rank.
    let j = Matrix::<6, 7>::from_fn(|r, c| {
        if c < 6 {
            if r == c {
                2.0
            } else {
                0.3 / (1.0 + (r + c) as f64)
            }
        } else {
            0.5 * (r as f64 + 1.0)
        }
    });
    let jp = j.pseudo_inverse().unwrap();

    // Moore–Penrose: J·J⁺·J == J and J·J⁺ symmetric.
    assert_close(j * jp * j, j, 1e-9);
    let jjp = j * jp;
    assert_close(jjp, jjp.transpose(), 1e-12);

    // Minimum-norm resolution: J⁺·v beats any other solution of J·x = v.
    let vtwist = Vector::<6>::from_fn(|i| i as f64 - 2.5);
    let x_min = jp * vtwist;
    let n = Vector::<7>::from_fn(|i| (i as f64 - 3.0) * 0.7);
    let jpjn = (jp * j) * n;
    let x_other = Vector::<7>::from_fn(|i| x_min[i] + n[i] - jpjn[i]);
    assert!((j * x_min - vtwist).norm() < 1e-9);
    assert!((j * x_other - vtwist).norm() < 1e-9);
    assert!(x_min.norm() <= x_other.norm() + 1e-9);
}

#[test]
fn svd_near_singular_jacobian() {
    // A 6x6 Jacobian at a near-singularity: column 5 nearly equals column 4.
    let mut j = Matrix::<6, 6>::from_fn(|i, jj| {
        if i == jj {
            1.0 + 0.1 * jj as f64
        } else {
            0.2 / (1.0 + (i + jj) as f64)
        }
    });
    for r in 0..6 {
        j[(r, 5)] = j[(r, 4)] + 1e-8 * (r as f64 + 1.0);
    }
    let f = j.svd().unwrap();
    let s = f.singular_values();
    let tol = 1e-4 * s[0];

    assert!(f.condition_number() > 1e6);
    assert_eq!(f.rank(tol), 5);

    // Truncated pseudo-inverse and solve stay finite (no blow-up on the tiny σ).
    let jp = f.pseudo_inverse_tol(tol);
    for r in 0..6 {
        for c in 0..6 {
            assert!(jp[(r, c)].is_finite());
        }
    }
    let x = f.solve(Vector::from_fn(|i| i as f64 + 1.0));
    for i in 0..6 {
        assert!(x[i].is_finite());
    }
}

#[test]
fn svd_overdetermined_least_squares() {
    // Fit a plane z = a·x + b·y + c to 30 sampled points; solve vs the normal equations.
    let sample = |i: usize| ((i as f64 * 0.37).sin(), (i as f64 * 0.53).cos());
    let design = Matrix::<30, 3>::from_fn(|i, c| {
        let (x, y) = sample(i);
        match c {
            0 => x,
            1 => y,
            _ => 1.0,
        }
    });
    let rhs = Vector::<30>::from_fn(|i| {
        let (x, y) = sample(i);
        0.5 * x - 1.2 * y + 2.0 + 1e-3 * (i as f64 * 1.7).sin()
    });
    let x_svd = design.svd().unwrap().solve(rhs);
    let x_ne = (design.transpose() * design)
        .solve(design.transpose() * rhs)
        .unwrap();
    for i in 0..3 {
        assert!((x_svd[i] - x_ne[i]).abs() < 1e-9);
    }
}
