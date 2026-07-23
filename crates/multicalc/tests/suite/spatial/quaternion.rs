//! Quaternion tests: comparison against offline-generated scipy values, algebraic identities, exp/ln round trips,
//! AD-vs-FD (including a finite-derivative regression at zero rotation), and f32 identity coverage.

use std::f64::consts::PI;

use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::scalar::{Dual, Numeric};
use multicalc::spatial::Quaternion;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const TOL: f64 = 1e-12;

// ---- helpers ----------------------------------------------------------------

fn rand_quat(rng: &mut StdRng) -> Quaternion<f64> {
    Quaternion::new(
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
    )
}

fn rand_unit(rng: &mut StdRng) -> Quaternion<f64> {
    loop {
        let q = rand_quat(rng);
        if q.norm() > 1e-3 {
            return q.normalized();
        }
    }
}

fn rand_unit_vec(rng: &mut StdRng) -> Vector<3, f64> {
    loop {
        let v = Vector::new([
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        ]);
        let n = v.dot(v).sqrt();
        if n > 1e-3 {
            return v * n.recip();
        }
    }
}

fn assert_close_q(a: Quaternion<f64>, b: Quaternion<f64>, tol: f64) {
    for (x, y) in a.as_array().iter().zip(b.as_array().iter()) {
        assert!((x - y).abs() < tol, "{x} vs {y}");
    }
}

fn assert_close_v(a: Vector<3, f64>, b: Vector<3, f64>, tol: f64) {
    for i in 0..3 {
        let (ai, bi) = (a.as_array()[i], b.as_array()[i]);
        assert!((ai - bi).abs() < tol, "{ai} vs {bi}");
    }
}

/// Compares two rotations (sign-insensitive) via their rotation matrices.
fn assert_rot_close(a: Quaternion<f64>, b: Quaternion<f64>, tol: f64) {
    let (ra, rb) = (a.to_rotation_matrix(), b.to_rotation_matrix());
    for r in 0..3 {
        for c in 0..3 {
            assert!((ra.get(r, c).copied().unwrap() - rb.get(r, c).copied().unwrap()).abs() < tol);
        }
    }
}

// ---- value goldens ----------------------------------------------------------

// Reference matrices are generated offline with scipy and converted at the boundary from its
// scalar-last [x, y, z, w] to our scalar-first [w, x, y, z]:
//
//   from scipy.spatial.transform import Rotation as R
//   q_xyzw = R.from_matrix(M).as_quat()
//   our_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
//
// The cases below are exact, so they are written inline rather than as a fixture file.
#[test]
fn rotation_matrix_goldens() {
    let s = std::f64::consts::FRAC_1_SQRT_2; // sin/cos of 45°

    // 90° about x.
    let qx = Quaternion::from_axis_angle(Vector::new([1.0, 0.0, 0.0]), PI / 2.0);
    assert_close_q(qx, Quaternion::new(s, s, 0.0, 0.0), TOL);
    let expect_x = Matrix::new([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]);
    for r in 0..3 {
        for c in 0..3 {
            assert!(
                (qx.to_rotation_matrix().get(r, c).copied().unwrap()
                    - expect_x.get(r, c).copied().unwrap())
                .abs()
                    < TOL
            );
        }
    }

    // The 120° rotation q = (0.5, 0.5, 0.5, 0.5): a cyclic axis permutation.
    let qc = Quaternion::new(0.5, 0.5, 0.5, 0.5).normalized();
    let expect_c = Matrix::new([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    for r in 0..3 {
        for c in 0..3 {
            assert!(
                (qc.to_rotation_matrix().get(r, c).copied().unwrap()
                    - expect_c.get(r, c).copied().unwrap())
                .abs()
                    < TOL
            );
        }
    }
}

#[test]
fn euler_golden_yaw_90() {
    // 90° about z is pure yaw in ZYX.
    let q = Quaternion::from_euler_zyx(0.0, 0.0, PI / 2.0);
    let s = std::f64::consts::FRAC_1_SQRT_2;
    assert_close_q(q, Quaternion::new(s, 0.0, 0.0, s), TOL);
    let (roll, pitch, yaw) = q.to_euler_zyx();
    assert!(roll.abs() < TOL);
    assert!(pitch.abs() < TOL);
    assert!((yaw - PI / 2.0).abs() < TOL);
}

#[test]
fn euler_gimbal_lock_poles() {
    // At pitch = ±π/2 the roll/yaw split is not unique; check the clamp branch reports the pole
    // exactly and that the reconstructed rotation still matches.
    for &sign in &[1.0, -1.0] {
        let q = Quaternion::from_euler_zyx(0.3, sign * PI / 2.0, 0.7);
        let (r2, p2, y2) = q.to_euler_zyx();
        assert!((p2 - sign * PI / 2.0).abs() < 1e-9);
        assert_rot_close(Quaternion::from_euler_zyx(r2, p2, y2), q, 1e-9);
    }
}

// ---- algebraic identities ---------------------------------------------------

#[test]
fn hamilton_associative() {
    let mut rng = StdRng::seed_from_u64(1);
    for _ in 0..2000 {
        let (a, b, c) = (
            rand_quat(&mut rng),
            rand_quat(&mut rng),
            rand_quat(&mut rng),
        );
        assert_close_q((a * b) * c, a * (b * c), 1e-11);
    }
}

#[test]
fn conjugate_antihomomorphism() {
    let mut rng = StdRng::seed_from_u64(2);
    for _ in 0..2000 {
        let (a, b) = (rand_quat(&mut rng), rand_quat(&mut rng));
        assert_close_q((a * b).conjugate(), b.conjugate() * a.conjugate(), 1e-12);
    }
}

#[test]
fn norm_multiplicative() {
    let mut rng = StdRng::seed_from_u64(3);
    for _ in 0..2000 {
        let (a, b) = (rand_quat(&mut rng), rand_quat(&mut rng));
        assert!(((a * b).norm() - a.norm() * b.norm()).abs() < 1e-12);
    }
}

#[test]
fn rotation_matrix_orthonormal() {
    let mut rng = StdRng::seed_from_u64(4);
    for _ in 0..2000 {
        let q = rand_unit(&mut rng);
        let m = q.to_rotation_matrix();
        let should_be_i = m.transpose() * m;
        for r in 0..3 {
            for c in 0..3 {
                let expect = if r == c { 1.0 } else { 0.0 };
                assert!((should_be_i.get(r, c).copied().unwrap() - expect).abs() < 1e-12);
            }
        }
        assert!((m.determinant() - 1.0).abs() < 1e-12);
    }
}

#[test]
fn matrix_roundtrip() {
    let mut rng = StdRng::seed_from_u64(5);
    for _ in 0..2000 {
        let q = rand_unit(&mut rng);
        let back = Quaternion::try_from_rotation_matrix(q.to_rotation_matrix()).unwrap();
        assert_rot_close(q, back, 1e-11);
    }
}

#[test]
fn scaled_axis_roundtrip() {
    let mut rng = StdRng::seed_from_u64(6);
    for _ in 0..2000 {
        let theta = rng.gen_range(0.0..0.99 * PI);
        let phi = rand_unit_vec(&mut rng) * theta;
        assert_close_v(
            Quaternion::from_scaled_axis(phi).to_scaled_axis(),
            phi,
            1e-10,
        );
    }
    // Small-angle branch.
    let tiny = Vector::new([1e-8, -2e-8, 0.5e-8]);
    assert_close_v(
        Quaternion::from_scaled_axis(tiny).to_scaled_axis(),
        tiny,
        1e-18,
    );
}

#[test]
fn axis_angle_roundtrip() {
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..2000 {
        let axis = rand_unit_vec(&mut rng);
        let angle = rng.gen_range(0.01..0.99 * PI);
        let (a2, t2) = Quaternion::from_axis_angle(axis, angle).to_axis_angle();
        assert!((t2 - angle).abs() < 1e-10);
        assert_close_v(a2, axis, 1e-9);
    }
}

#[test]
fn euler_roundtrip() {
    let mut rng = StdRng::seed_from_u64(8);
    for _ in 0..2000 {
        let roll = rng.gen_range(-0.99 * PI..0.99 * PI); // clear of the ±π wrap boundary
        let pitch = rng.gen_range(-0.49 * PI..0.49 * PI); // clear of gimbal lock
        let yaw = rng.gen_range(-0.99 * PI..0.99 * PI);
        let (r2, p2, y2) = Quaternion::from_euler_zyx(roll, pitch, yaw).to_euler_zyx();
        assert!((r2 - roll).abs() < 1e-10);
        assert!((p2 - pitch).abs() < 1e-10);
        assert!((y2 - yaw).abs() < 1e-10);
    }
}

// ---- exponential and logarithm ----------------------------------------------

#[test]
fn exp_ln_roundtrip() {
    let mut rng = StdRng::seed_from_u64(20);
    for _ in 0..2000 {
        // ln ∘ exp on a general (small) quaternion. Keep the vector part below π so exp stays
        // inside the principal branch that ln inverts.
        let q = Quaternion::new(
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        );
        assert_close_q(q.exp().ln(), q, 1e-10);

        // exp ∘ ln on a quaternion with positive real part (clear of the negative-real branch cut).
        let p = Quaternion::new(
            rng.gen_range(0.2..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        );
        assert_close_q(p.ln().exp(), p, 1e-10);
    }
}

#[test]
fn exp_matches_from_scaled_axis() {
    let mut rng = StdRng::seed_from_u64(21);
    for _ in 0..2000 {
        // For a pure-vector quaternion v, exp(v) is the rotation from_scaled_axis(2v).
        let theta = rng.gen_range(0.0..0.49 * PI);
        let v = rand_unit_vec(&mut rng) * theta;
        let [vx, vy, vz] = *v.as_array();
        let qv = Quaternion::new(0.0, vx, vy, vz);
        assert_rot_close(qv.exp(), Quaternion::from_scaled_axis(v * 2.0), 1e-11);
    }
    // Both branches stay finite and unit at exactly zero.
    let z = Quaternion::<f64>::new(0.0, 0.0, 0.0, 0.0);
    assert!((z.exp().norm() - 1.0).abs() < TOL);
}

// ---- interpolation and point action ----------------------------------------

#[test]
fn slerp_endpoints_shortest_path() {
    let mut rng = StdRng::seed_from_u64(9);
    for _ in 0..2000 {
        let (a, b) = (rand_unit(&mut rng), rand_unit(&mut rng));
        // Endpoints reproduce the input rotation even when dot < 0 (sign-flip branch).
        assert_rot_close(a.slerp(b, 0.0), a, 1e-10);
        assert_rot_close(a.slerp(b, 1.0), b, 1e-10);
    }
}

#[test]
fn slerp_constant_rate() {
    let mut rng = StdRng::seed_from_u64(10);
    for _ in 0..1000 {
        let a = rand_unit(&mut rng);
        let delta = Quaternion::from_axis_angle(rand_unit_vec(&mut rng), rng.gen_range(0.1..2.0));
        let b = a * delta;
        // The midpoint of the geodesic is half the relative rotation.
        let mid = a.slerp(b, 0.5);
        let half = a * Quaternion::from_scaled_axis(delta.to_scaled_axis() * 0.5);
        assert_rot_close(mid, half, 1e-9);
    }
}

#[test]
fn slerp_small_angle_finite() {
    let a = Quaternion::<f64>::identity();
    let b = Quaternion::from_axis_angle(Vector::new([0.0, 0.0, 1.0]), 1e-9);
    let m = a.slerp(b, 0.5);
    assert!(m.norm().is_finite());
    assert!((m.norm() - 1.0).abs() < 1e-12);
}

#[test]
fn transform_point_properties() {
    let mut rng = StdRng::seed_from_u64(11);
    for _ in 0..2000 {
        let q = rand_unit(&mut rng);
        let v = Vector::new([
            rng.gen_range(-5.0..5.0),
            rng.gen_range(-5.0..5.0),
            rng.gen_range(-5.0..5.0),
        ]);
        let rv = q.transform_point(v);
        // Norm-preserving.
        assert!((rv.dot(rv).sqrt() - v.dot(v).sqrt()).abs() < 1e-11);
        // Inverse undoes it.
        assert_close_v(q.inverse().transform_point(rv), v, 1e-10);
        // Matches the matrix action.
        assert_close_v(rv, q.to_rotation_matrix() * v, 1e-11);
    }
}

// ---- autodiff ---------------------------------------------------------------

/// Rotates x̂ by `angle` about ẑ and returns the x-component, which is `cos(angle)`.
fn rotated_component<T: Numeric>(angle: T) -> T {
    let q = Quaternion::from_axis_angle(Vector::new([T::ZERO, T::ZERO, T::ONE]), angle);
    q.transform_point(Vector::new([T::ONE, T::ZERO, T::ZERO]))
        .as_array()[0]
}

#[test]
fn ad_matches_finite_difference() {
    let mut rng = StdRng::seed_from_u64(12);
    for _ in 0..1000 {
        let a = rng.gen_range(-PI..PI);
        let ad = rotated_component(Dual::variable(a)).deriv;
        let h = 1e-6;
        let fd = (rotated_component(a + h) - rotated_component(a - h)) / (2.0 * h);
        assert!((ad + a.sin()).abs() < 1e-9); // exact derivative is -sin(a)
        assert!((ad - fd).abs() < 1e-6); // FD's own error floor
    }
}

/// Rotates x̂ by `from_scaled_axis(t·ẑ)` and returns the y-component, which is `sin(t)` (chosen so
/// the derivative at `t = 0` is a non-zero 1). Differentiating here exercises the small-angle
/// branch that once produced NaN via `sqrt(0)`.
fn scaled_axis_component<T: Numeric>(t: T) -> T {
    let phi = Vector::new([T::ZERO, T::ZERO, T::ONE]) * t;
    let q = Quaternion::from_scaled_axis(phi);
    q.transform_point(Vector::new([T::ONE, T::ZERO, T::ZERO]))
        .as_array()[1]
}

#[test]
fn ad_through_scaled_axis_finite_at_zero() {
    // Regression: from_scaled_axis must give a finite derivative at and near ω = 0. The mapped
    // component is sin(t) about ẑ, whose derivative at 0 is 1.
    let d0 = scaled_axis_component(Dual::variable(0.0)).deriv;
    assert!(d0.is_finite());
    assert!((d0 - 1.0).abs() < 1e-12);

    for &t in &[1e-9, 1e-6, 1e-3, 0.5] {
        let ad = scaled_axis_component(Dual::variable(t)).deriv;
        assert!(ad.is_finite());
        assert!((ad - t.cos()).abs() < 1e-6);
    }
}

// ---- f32 identity coverage --------------------------------------------------

#[test]
fn f32_identities() {
    let mut rng = StdRng::seed_from_u64(13);
    for _ in 0..1000 {
        let axis = Vector::new([
            rng.gen_range(-1.0f32..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        ]);
        let angle = rng.gen_range(0.01f32..0.99 * std::f32::consts::PI);
        let q = Quaternion::from_axis_angle(axis, angle);

        // Orthonormal rotation matrix.
        let m = q.to_rotation_matrix();
        let i = m.transpose() * m;
        for r in 0..3 {
            for c in 0..3 {
                let expect = if r == c { 1.0 } else { 0.0 };
                assert!((i.get(r, c).copied().unwrap() - expect).abs() < 1e-4);
            }
        }

        // from_scaled_axis ∘ to_scaled_axis round trip within f32 tolerance.
        let back = Quaternion::from_scaled_axis(q.to_scaled_axis());
        let (ra, rb) = (q.to_rotation_matrix(), back.to_rotation_matrix());
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (ra.get(r, c).copied().unwrap() - rb.get(r, c).copied().unwrap()).abs() < 1e-4
                );
            }
        }
    }
}
