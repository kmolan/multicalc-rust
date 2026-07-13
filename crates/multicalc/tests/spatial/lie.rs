//! Lie-group tests: group laws, exp/log round trips across the θ=0 and θ=π regions, adjoint
//! identities, hat/vee, act-vs-matrix consistency, geodesic interpolation, AD-vs-FD, and f32
//! identity coverage.

use std::f64::consts::PI;

use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::scalar::{Dual, Numeric};
use multicalc::spatial::{SE2, SE3, SO2, SO3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const TOL: f64 = 1e-10;

// ---- helpers ----------------------------------------------------------------

fn rand_vec3(rng: &mut StdRng) -> Vector<3, f64> {
    Vector::new([
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
    ])
}

fn rand_unit_vec3(rng: &mut StdRng) -> Vector<3, f64> {
    loop {
        let v = rand_vec3(rng);
        let n = v.dot(v).sqrt();
        if n > 1e-3 {
            return v * n.recip();
        }
    }
}

fn rand_twist6(rng: &mut StdRng) -> Vector<6, f64> {
    Vector::new([
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
    ])
}

fn rand_so3(rng: &mut StdRng) -> SO3<f64> {
    SO3::exp(rand_unit_vec3(rng) * rng.gen_range(-2.5..2.5))
}

fn rand_se3(rng: &mut StdRng) -> SE3<f64> {
    SE3::from_parts(
        rand_so3(rng),
        Vector::new([
            rng.gen_range(-2.0..2.0),
            rng.gen_range(-2.0..2.0),
            rng.gen_range(-2.0..2.0),
        ]),
    )
}

fn rand_so2(rng: &mut StdRng) -> SO2<f64> {
    SO2::from_angle(rng.gen_range(-PI..PI))
}

fn rand_se2(rng: &mut StdRng) -> SE2<f64> {
    SE2::from_parts(
        rand_so2(rng),
        Vector::new([rng.gen_range(-2.0..2.0), rng.gen_range(-2.0..2.0)]),
    )
}

fn assert_mat_close<const R: usize, const C: usize>(
    a: Matrix<R, C, f64>,
    b: Matrix<R, C, f64>,
    tol: f64,
) {
    for i in 0..R {
        for j in 0..C {
            assert!(
                (a[(i, j)] - b[(i, j)]).abs() < tol,
                "({i},{j}): {} vs {}",
                a[(i, j)],
                b[(i, j)]
            );
        }
    }
}

fn assert_vec_close<const N: usize>(a: Vector<N, f64>, b: Vector<N, f64>, tol: f64) {
    for i in 0..N {
        assert!((a[i] - b[i]).abs() < tol, "[{i}]: {} vs {}", a[i], b[i]);
    }
}

// ---- SO(3) ------------------------------------------------------------------

#[test]
fn so3_group_laws() {
    let mut rng = StdRng::seed_from_u64(1);
    for _ in 0..200 {
        let (a, b, c) = (rand_so3(&mut rng), rand_so3(&mut rng), rand_so3(&mut rng));
        assert_mat_close(((a * b) * c).to_matrix(), (a * (b * c)).to_matrix(), TOL);
        assert_mat_close((a * SO3::identity()).to_matrix(), a.to_matrix(), TOL);
        assert_mat_close(
            (a * a.inverse()).to_matrix(),
            SO3::identity().to_matrix(),
            TOL,
        );
    }
}

#[test]
fn so3_exp_log_roundtrip() {
    let mut rng = StdRng::seed_from_u64(2);
    for _ in 0..200 {
        let axis = rand_unit_vec3(&mut rng);
        for &angle in &[1e-9, 1e-4, 0.5, 2.0, PI - 1e-6] {
            let phi = axis * angle;
            assert_vec_close(SO3::exp(phi).log(), phi, 1e-7);
        }
    }
}

#[test]
fn so3_adjoint_identity() {
    let mut rng = StdRng::seed_from_u64(3);
    for _ in 0..200 {
        let r = rand_so3(&mut rng);
        let xi = rand_vec3(&mut rng) * 0.5;
        let lhs = SO3::exp(r.adjoint() * xi).to_matrix();
        let rhs = (r * SO3::exp(xi) * r.inverse()).to_matrix();
        assert_mat_close(lhs, rhs, 1e-9);
    }
}

#[test]
fn so3_hat_vee_roundtrip() {
    let mut rng = StdRng::seed_from_u64(4);
    for _ in 0..100 {
        let phi = rand_vec3(&mut rng);
        assert_vec_close(SO3::vee(SO3::hat(phi)), phi, TOL);
    }
}

#[test]
fn so3_act_matches_matrix() {
    let mut rng = StdRng::seed_from_u64(5);
    for _ in 0..100 {
        let r = rand_so3(&mut rng);
        let p = rand_vec3(&mut rng);
        assert_vec_close(r.act(p), r.to_matrix() * p, TOL);
    }
}

#[test]
fn so3_interpolate_endpoints() {
    let mut rng = StdRng::seed_from_u64(6);
    for _ in 0..100 {
        let (a, b) = (rand_so3(&mut rng), rand_so3(&mut rng));
        assert_mat_close(a.interpolate(b, 0.0).to_matrix(), a.to_matrix(), TOL);
        assert_mat_close(a.interpolate(b, 1.0).to_matrix(), b.to_matrix(), 1e-9);
    }
}

#[test]
fn so3_left_right_jacobian_relation() {
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..100 {
        let phi = rand_vec3(&mut rng) * 1.5;
        // J_l(φ) = exp(φ) · J_r(φ)
        assert_mat_close(
            SO3::left_jacobian(phi),
            SO3::exp(phi).to_matrix() * SO3::right_jacobian(phi),
            1e-9,
        );
    }
}

// ---- SE(3) ------------------------------------------------------------------

#[test]
fn se3_group_laws() {
    let mut rng = StdRng::seed_from_u64(10);
    for _ in 0..200 {
        let (a, b, c) = (rand_se3(&mut rng), rand_se3(&mut rng), rand_se3(&mut rng));
        assert_mat_close(((a * b) * c).to_matrix(), (a * (b * c)).to_matrix(), 1e-9);
        assert_mat_close((a * SE3::identity()).to_matrix(), a.to_matrix(), TOL);
        assert_mat_close(
            (a * a.inverse()).to_matrix(),
            SE3::identity().to_matrix(),
            1e-9,
        );
    }
}

#[test]
fn se3_exp_log_roundtrip() {
    let mut rng = StdRng::seed_from_u64(11);
    for _ in 0..300 {
        let axis = rand_unit_vec3(&mut rng);
        for &angle in &[1e-9, 1e-4, 0.7, 2.0, PI - 1e-6] {
            let v = rand_vec3(&mut rng);
            let xi = Vector::new([
                v[0],
                v[1],
                v[2],
                axis[0] * angle,
                axis[1] * angle,
                axis[2] * angle,
            ]);
            assert_vec_close(SE3::exp(xi).log(), xi, 1e-6);
        }
    }
}

#[test]
fn se3_adjoint_identity() {
    let mut rng = StdRng::seed_from_u64(12);
    for _ in 0..200 {
        let x = rand_se3(&mut rng);
        let xi = rand_twist6(&mut rng) * 0.3;
        let lhs = SE3::exp(x.adjoint() * xi).to_matrix();
        let rhs = (x * SE3::exp(xi) * x.inverse()).to_matrix();
        assert_mat_close(lhs, rhs, 1e-8);
    }
}

#[test]
fn se3_act_matches_homogeneous_matrix() {
    let mut rng = StdRng::seed_from_u64(13);
    for _ in 0..100 {
        let x = rand_se3(&mut rng);
        let p = rand_vec3(&mut rng);
        let hp = Vector::new([p[0], p[1], p[2], 1.0]);
        let m = x.to_matrix() * hp;
        assert_vec_close(x.act(p), Vector::new([m[0], m[1], m[2]]), TOL);
    }
}

#[test]
fn se3_matrix_roundtrip() {
    let mut rng = StdRng::seed_from_u64(14);
    for _ in 0..100 {
        let x = rand_se3(&mut rng);
        let back = SE3::try_from_matrix(x.to_matrix()).unwrap();
        assert_mat_close(back.to_matrix(), x.to_matrix(), 1e-9);
    }
}

#[test]
fn se3_hat_vee_roundtrip() {
    let mut rng = StdRng::seed_from_u64(15);
    for _ in 0..100 {
        let xi = rand_twist6(&mut rng);
        assert_vec_close(SE3::vee(SE3::hat(xi)), xi, TOL);
    }
}

#[test]
fn se3_interpolate_endpoints() {
    let mut rng = StdRng::seed_from_u64(16);
    for _ in 0..100 {
        let (a, b) = (rand_se3(&mut rng), rand_se3(&mut rng));
        assert_mat_close(a.interpolate(b, 0.0).to_matrix(), a.to_matrix(), 1e-9);
        assert_mat_close(a.interpolate(b, 1.0).to_matrix(), b.to_matrix(), 1e-8);
    }
}

// ---- SO(2) / SE(2) ----------------------------------------------------------

#[test]
fn so2_group_and_roundtrip() {
    let mut rng = StdRng::seed_from_u64(20);
    for _ in 0..200 {
        let (a, b, c) = (rand_so2(&mut rng), rand_so2(&mut rng), rand_so2(&mut rng));
        assert_mat_close(((a * b) * c).to_matrix(), (a * (b * c)).to_matrix(), TOL);
        assert_mat_close(
            (a * a.inverse()).to_matrix(),
            SO2::identity().to_matrix(),
            TOL,
        );
        for &th in &[1e-9, 0.3, PI - 1e-6] {
            assert!((SO2::exp(th).log() - th).abs() < 1e-9);
        }
        let p = Vector::new([rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)]);
        assert_vec_close(a.act(p), a.to_matrix() * p, TOL);
    }
}

#[test]
fn se2_group_and_roundtrip() {
    let mut rng = StdRng::seed_from_u64(21);
    for _ in 0..300 {
        let (a, b, c) = (rand_se2(&mut rng), rand_se2(&mut rng), rand_se2(&mut rng));
        assert_mat_close(((a * b) * c).to_matrix(), (a * (b * c)).to_matrix(), TOL);
        assert_mat_close(
            (a * a.inverse()).to_matrix(),
            SE2::identity().to_matrix(),
            TOL,
        );
        for &th in &[1e-9, 0.4, PI - 1e-6] {
            let xi = Vector::new([rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), th]);
            assert_vec_close(SE2::exp(xi).log(), xi, 1e-7);
        }
        let p = Vector::new([rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)]);
        let hp = Vector::new([p[0], p[1], 1.0]);
        let m = a.to_matrix() * hp;
        assert_vec_close(a.act(p), Vector::new([m[0], m[1]]), TOL);
    }
}

#[test]
fn se2_adjoint_identity() {
    let mut rng = StdRng::seed_from_u64(22);
    for _ in 0..200 {
        let x = rand_se2(&mut rng);
        let xi = Vector::new([
            rng.gen_range(-0.5..0.5),
            rng.gen_range(-0.5..0.5),
            rng.gen_range(-0.5..0.5),
        ]);
        let lhs = SE2::exp(x.adjoint() * xi).to_matrix();
        let rhs = (x * SE2::exp(xi) * x.inverse()).to_matrix();
        assert_mat_close(lhs, rhs, 1e-9);
    }
}

// ---- autodiff ---------------------------------------------------------------

#[test]
fn so3_exp_ad_vs_fd() {
    // d/dφ_k of exp(φ).act(p): autodiff (Dual) vs central finite difference.
    let phi0 = [0.2_f64, -0.1, 0.35];
    let p = [0.5_f64, 0.3, -0.7];
    let h = 1e-6;
    for k in 0..3 {
        let phi = Vector::new([
            if k == 0 {
                Dual::variable(phi0[0])
            } else {
                Dual::constant(phi0[0])
            },
            if k == 1 {
                Dual::variable(phi0[1])
            } else {
                Dual::constant(phi0[1])
            },
            if k == 2 {
                Dual::variable(phi0[2])
            } else {
                Dual::constant(phi0[2])
            },
        ]);
        let pv = Vector::new([
            Dual::constant(p[0]),
            Dual::constant(p[1]),
            Dual::constant(p[2]),
        ]);
        let out = SO3::exp(phi).act(pv);

        let mut phi_p = phi0;
        let mut phi_m = phi0;
        phi_p[k] += h;
        phi_m[k] -= h;
        let fp = SO3::exp(Vector::new(phi_p)).act(Vector::new(p));
        let fm = SO3::exp(Vector::new(phi_m)).act(Vector::new(p));
        for i in 0..3 {
            let fd = (fp[i] - fm[i]) / (2.0 * h);
            assert!(
                (out[i].deriv - fd).abs() < 1e-6,
                "k={k} i={i}: {} vs {}",
                out[i].deriv,
                fd
            );
        }
    }
}

#[test]
fn so3_exp_derivative_finite_at_zero() {
    // At φ = 0 the exp map is smooth; a naive sqrt-based path would give a NaN derivative here.
    let phi = Vector::new([
        Dual::variable(0.0),
        Dual::constant(0.0),
        Dual::constant(0.0),
    ]);
    let pv = Vector::new([
        Dual::constant(1.0),
        Dual::constant(0.0),
        Dual::constant(0.0),
    ]);
    let out = SO3::exp(phi).act(pv);
    for i in 0..3 {
        assert!(out[i].deriv.is_finite());
    }
}

// ---- f32 identity coverage --------------------------------------------------

#[test]
fn f32_identity_coverage() {
    let phi = Vector::new([0.2_f32, -0.3, 0.5]);
    let back = SO3::exp(phi).log();
    for i in 0..3 {
        assert!((back[i] - phi[i]).abs() < 1e-4);
    }

    let xi = Vector::new([0.1_f32, -0.2, 0.3, 0.2, -0.3, 0.5]);
    let back6 = SE3::exp(xi).log();
    for i in 0..6 {
        assert!((back6[i] - xi[i]).abs() < 1e-4);
    }

    // Rotation matrix orthonormality: RᵀR = I.
    let r = SO3::exp(phi).to_matrix();
    let rtr = r.transpose() * r;
    for i in 0..3 {
        for j in 0..3 {
            let expect = if i == j { 1.0 } else { 0.0 };
            assert!((rtr[(i, j)] - expect).abs() < 1e-5);
        }
    }
}

// ---- value goldens (exact, scipy-equivalent) --------------------------------

#[test]
fn so3_exp_goldens() {
    // exp(θ·axis).to_matrix() == R.from_rotvec([...]).as_matrix()
    let rz = SO3::<f64>::exp(Vector::new([0.0, 0.0, PI / 2.0]));
    assert_mat_close(
        rz.to_matrix(),
        Matrix::new([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        1e-12,
    );
    let rx = SO3::<f64>::exp(Vector::new([PI / 2.0, 0.0, 0.0]));
    assert_mat_close(
        rx.to_matrix(),
        Matrix::new([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
        1e-12,
    );
    let ry = SO3::<f64>::exp(Vector::new([0.0, PI / 2.0, 0.0]));
    assert_mat_close(
        ry.to_matrix(),
        Matrix::new([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
        1e-12,
    );
}

#[test]
fn se3_to_matrix_goldens() {
    // Pure translation: identity rotation, translation (1, 2, 3).
    let t = SE3::<f64>::exp(Vector::new([1.0, 2.0, 3.0, 0.0, 0.0, 0.0]));
    assert_mat_close(
        t.to_matrix(),
        Matrix::new([
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]),
        1e-12,
    );
    // Pure rotation twist: Rz(90°), zero translation (J_l · 0 = 0).
    let r = SE3::<f64>::exp(Vector::new([0.0, 0.0, 0.0, 0.0, 0.0, PI / 2.0]));
    assert_mat_close(
        r.to_matrix(),
        Matrix::new([
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]),
        1e-12,
    );
}
