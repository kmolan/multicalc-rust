//! Quaternion tests: comparison against offline-generated scipy values, algebraic identities, exp/ln round trips,
//! AD-vs-FD (including a finite-derivative regression at zero rotation), and f32 identity coverage.

use std::f64::consts::PI;

use multicalc::linear_algebra::{Matrix, Vector, Vector3D};
use multicalc::scalar::{Dual, Numeric};
use multicalc::spatial::Quaternion;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const TOL: f64 = 1e-12;

// ---- helpers ----------------------------------------------------------------

fn random_quaternion(rng: &mut StdRng) -> Quaternion<f64> {
    Quaternion::new(
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
    )
}

fn random_unit_quaternion(rng: &mut StdRng) -> Quaternion<f64> {
    loop {
        let quaternion = random_quaternion(rng);
        if quaternion.norm() > 1e-3 {
            return quaternion.normalized();
        }
    }
}

fn random_unit_vector(rng: &mut StdRng) -> Vector3D {
    loop {
        let vector = Vector::new([
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        ]);
        let norm = vector.dot(vector).sqrt();
        if norm > 1e-3 {
            return vector * norm.recip();
        }
    }
}

fn assert_quaternions_close(first: Quaternion<f64>, second: Quaternion<f64>, tolerance: f64) {
    for (left, right) in first.as_array().iter().zip(second.as_array().iter()) {
        assert!((left - right).abs() < tolerance, "{left} vs {right}");
    }
}

fn assert_vectors_close(first: Vector3D, second: Vector3D, tolerance: f64) {
    for index in 0..3 {
        let left = first[index];
        let right = second[index];
        assert!((left - right).abs() < tolerance, "{left} vs {right}");
    }
}

/// Compares two rotations (sign-insensitive) via their rotation matrices.
fn assert_rotations_close(first: Quaternion<f64>, second: Quaternion<f64>, tolerance: f64) {
    let left_matrix = first.to_rotation_matrix();
    let right_matrix = second.to_rotation_matrix();
    for row in 0..3 {
        for column in 0..3 {
            assert!((left_matrix[(row, column)] - right_matrix[(row, column)]).abs() < tolerance);
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
    let half_root_two = std::f64::consts::FRAC_1_SQRT_2; // sin/cos of 45°

    // 90° about x.
    let quarter_turn_about_x = Quaternion::from_axis_angle(Vector::new([1.0, 0.0, 0.0]), PI / 2.0);
    assert_quaternions_close(
        quarter_turn_about_x,
        Quaternion::new(half_root_two, half_root_two, 0.0, 0.0),
        TOL,
    );
    let expected_x = Matrix::new([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]);
    for row in 0..3 {
        for column in 0..3 {
            assert!(
                (quarter_turn_about_x.to_rotation_matrix()[(row, column)]
                    - expected_x[(row, column)])
                    .abs()
                    < TOL
            );
        }
    }

    // The 120° rotation q = (0.5, 0.5, 0.5, 0.5): a cyclic axis permutation.
    let cyclic_permutation = Quaternion::new(0.5, 0.5, 0.5, 0.5).normalized();
    let expected_cyclic = Matrix::new([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    for row in 0..3 {
        for column in 0..3 {
            assert!(
                (cyclic_permutation.to_rotation_matrix()[(row, column)]
                    - expected_cyclic[(row, column)])
                    .abs()
                    < TOL
            );
        }
    }
}

#[test]
fn euler_golden_yaw_90() {
    // 90° about z is pure yaw in ZYX.
    let quaternion = Quaternion::from_euler_zyx(0.0, 0.0, PI / 2.0);
    let half_root_two = std::f64::consts::FRAC_1_SQRT_2;
    assert_quaternions_close(
        quaternion,
        Quaternion::new(half_root_two, 0.0, 0.0, half_root_two),
        TOL,
    );
    let (roll, pitch, yaw) = quaternion.to_euler_zyx();
    assert!(roll.abs() < TOL);
    assert!(pitch.abs() < TOL);
    assert!((yaw - PI / 2.0).abs() < TOL);
}

#[test]
fn euler_gimbal_lock_poles() {
    // At pitch = ±π/2 the roll/yaw split is not unique; check the clamp branch reports the pole
    // exactly and that the reconstructed rotation still matches.
    for &sign in &[1.0, -1.0] {
        let quaternion = Quaternion::from_euler_zyx(0.3, sign * PI / 2.0, 0.7);
        let (recovered_roll, recovered_pitch, recovered_yaw) = quaternion.to_euler_zyx();
        assert!((recovered_pitch - sign * PI / 2.0).abs() < 1e-9);
        assert_rotations_close(
            Quaternion::from_euler_zyx(recovered_roll, recovered_pitch, recovered_yaw),
            quaternion,
            1e-9,
        );
    }
}

// ---- algebraic identities ---------------------------------------------------

#[test]
fn hamilton_associative() {
    let mut rng = StdRng::seed_from_u64(1);
    for _ in 0..2000 {
        let first = random_quaternion(&mut rng);
        let second = random_quaternion(&mut rng);
        let third = random_quaternion(&mut rng);
        assert_quaternions_close((first * second) * third, first * (second * third), 1e-11);
    }
}

#[test]
fn conjugate_antihomomorphism() {
    let mut rng = StdRng::seed_from_u64(2);
    for _ in 0..2000 {
        let first = random_quaternion(&mut rng);
        let second = random_quaternion(&mut rng);
        assert_quaternions_close(
            (first * second).conjugate(),
            second.conjugate() * first.conjugate(),
            1e-12,
        );
    }
}

#[test]
fn norm_multiplicative() {
    let mut rng = StdRng::seed_from_u64(3);
    for _ in 0..2000 {
        let first = random_quaternion(&mut rng);
        let second = random_quaternion(&mut rng);
        assert!(((first * second).norm() - first.norm() * second.norm()).abs() < 1e-12);
    }
}

#[test]
fn rotation_matrix_orthonormal() {
    let mut rng = StdRng::seed_from_u64(4);
    for _ in 0..2000 {
        let quaternion = random_unit_quaternion(&mut rng);
        let rotation_matrix = quaternion.to_rotation_matrix();
        let should_be_identity = rotation_matrix.transpose() * rotation_matrix;
        for row in 0..3 {
            for column in 0..3 {
                let expected = if row == column { 1.0 } else { 0.0 };
                assert!((should_be_identity[(row, column)] - expected).abs() < 1e-12);
            }
        }
        assert!((rotation_matrix.determinant() - 1.0).abs() < 1e-12);
    }
}

#[test]
fn matrix_roundtrip() {
    let mut rng = StdRng::seed_from_u64(5);
    for _ in 0..2000 {
        let quaternion = random_unit_quaternion(&mut rng);
        let recovered =
            Quaternion::try_from_rotation_matrix(quaternion.to_rotation_matrix()).unwrap();
        assert_rotations_close(quaternion, recovered, 1e-11);
    }
}

#[test]
fn scaled_axis_roundtrip() {
    let mut rng = StdRng::seed_from_u64(6);
    for _ in 0..2000 {
        let angle = rng.gen_range(0.0..0.99 * PI);
        let rotation_vector = random_unit_vector(&mut rng) * angle;
        assert_vectors_close(
            Quaternion::from_scaled_axis(rotation_vector).to_scaled_axis(),
            rotation_vector,
            1e-10,
        );
    }
    // Small-angle branch.
    let tiny_rotation = Vector::new([1e-8, -2e-8, 0.5e-8]);
    assert_vectors_close(
        Quaternion::from_scaled_axis(tiny_rotation).to_scaled_axis(),
        tiny_rotation,
        1e-18,
    );
}

#[test]
fn axis_angle_roundtrip() {
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..2000 {
        let axis = random_unit_vector(&mut rng);
        let angle = rng.gen_range(0.01..0.99 * PI);
        let (recovered_axis, recovered_angle) =
            Quaternion::from_axis_angle(axis, angle).to_axis_angle();
        assert!((recovered_angle - angle).abs() < 1e-10);
        assert_vectors_close(recovered_axis, axis, 1e-9);
    }
}

#[test]
fn euler_roundtrip() {
    let mut rng = StdRng::seed_from_u64(8);
    for _ in 0..2000 {
        let roll = rng.gen_range(-0.99 * PI..0.99 * PI); // clear of the ±π wrap boundary
        let pitch = rng.gen_range(-0.49 * PI..0.49 * PI); // clear of gimbal lock
        let yaw = rng.gen_range(-0.99 * PI..0.99 * PI);
        let (recovered_roll, recovered_pitch, recovered_yaw) =
            Quaternion::from_euler_zyx(roll, pitch, yaw).to_euler_zyx();
        assert!((recovered_roll - roll).abs() < 1e-10);
        assert!((recovered_pitch - pitch).abs() < 1e-10);
        assert!((recovered_yaw - yaw).abs() < 1e-10);
    }
}

// ---- exponential and logarithm ----------------------------------------------

#[test]
fn exp_ln_roundtrip() {
    let mut rng = StdRng::seed_from_u64(20);
    for _ in 0..2000 {
        // ln ∘ exp on a general (small) quaternion. Keep the vector part below π so exp stays
        // inside the principal branch that ln inverts.
        let quaternion = Quaternion::new(
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        );
        assert_quaternions_close(quaternion.exp().ln(), quaternion, 1e-10);

        // exp ∘ ln on a quaternion with positive real part (clear of the negative-real branch cut).
        let positive_real = Quaternion::new(
            rng.gen_range(0.2..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        );
        assert_quaternions_close(positive_real.ln().exp(), positive_real, 1e-10);
    }
}

#[test]
fn exp_matches_from_scaled_axis() {
    let mut rng = StdRng::seed_from_u64(21);
    for _ in 0..2000 {
        // For a pure-vector quaternion v, exp(v) is the rotation from_scaled_axis(2v).
        let angle = rng.gen_range(0.0..0.49 * PI);
        let vector = random_unit_vector(&mut rng) * angle;
        let pure_vector = Quaternion::new(0.0, vector[0], vector[1], vector[2]);
        assert_rotations_close(
            pure_vector.exp(),
            Quaternion::from_scaled_axis(vector * 2.0),
            1e-11,
        );
    }
    // Both branches stay finite and unit at exactly zero.
    let zero_quaternion = Quaternion::<f64>::new(0.0, 0.0, 0.0, 0.0);
    assert!((zero_quaternion.exp().norm() - 1.0).abs() < TOL);
}

// ---- interpolation and point action ----------------------------------------

#[test]
fn slerp_endpoints_shortest_path() {
    let mut rng = StdRng::seed_from_u64(9);
    for _ in 0..2000 {
        let first = random_unit_quaternion(&mut rng);
        let second = random_unit_quaternion(&mut rng);
        // Endpoints reproduce the input rotation even when dot < 0 (sign-flip branch).
        assert_rotations_close(first.slerp(second, 0.0), first, 1e-10);
        assert_rotations_close(first.slerp(second, 1.0), second, 1e-10);
    }
}

#[test]
fn slerp_constant_rate() {
    let mut rng = StdRng::seed_from_u64(10);
    for _ in 0..1000 {
        let first = random_unit_quaternion(&mut rng);
        let delta =
            Quaternion::from_axis_angle(random_unit_vector(&mut rng), rng.gen_range(0.1..2.0));
        let second = first * delta;
        // The midpoint of the geodesic is half the relative rotation.
        let midpoint = first.slerp(second, 0.5);
        let half_rotation = first * Quaternion::from_scaled_axis(delta.to_scaled_axis() * 0.5);
        assert_rotations_close(midpoint, half_rotation, 1e-9);
    }
}

#[test]
fn slerp_small_angle_finite() {
    let first = Quaternion::<f64>::identity();
    let second = Quaternion::from_axis_angle(Vector::new([0.0, 0.0, 1.0]), 1e-9);
    let midpoint = first.slerp(second, 0.5);
    assert!(midpoint.norm().is_finite());
    assert!((midpoint.norm() - 1.0).abs() < 1e-12);
}

#[test]
fn transform_point_properties() {
    let mut rng = StdRng::seed_from_u64(11);
    for _ in 0..2000 {
        let quaternion = random_unit_quaternion(&mut rng);
        let vector = Vector::new([
            rng.gen_range(-5.0..5.0),
            rng.gen_range(-5.0..5.0),
            rng.gen_range(-5.0..5.0),
        ]);
        let rotated = quaternion.transform_point(vector);
        // Norm-preserving.
        assert!((rotated.dot(rotated).sqrt() - vector.dot(vector).sqrt()).abs() < 1e-11);
        // Inverse undoes it.
        assert_vectors_close(quaternion.inverse().transform_point(rotated), vector, 1e-10);
        // Matches the matrix action.
        assert_vectors_close(rotated, quaternion.to_rotation_matrix() * vector, 1e-11);
    }
}

#[test]
fn inverse_transform_point_undoes_transform_point() {
    let mut rng = StdRng::seed_from_u64(12);

    for _ in 0..2000 {
        let quaternion = random_unit_quaternion(&mut rng);
        let vector = Vector::new([
            rng.gen_range(-5.0..5.0),
            rng.gen_range(-5.0..5.0),
            rng.gen_range(-5.0..5.0),
        ]);

        // Round trip in both orders.
        let forward = quaternion.transform_point(vector);
        assert_vectors_close(quaternion.inverse_transform_point(forward), vector, 1e-10);

        let backward = quaternion.inverse_transform_point(vector);
        assert_vectors_close(quaternion.transform_point(backward), vector, 1e-10);
        // Same as rotating by the conjugate explicitly, which is what it saves the caller writing.
        assert_vectors_close(
            backward,
            quaternion.conjugate().transform_point(vector),
            TOL,
        );
        // Matches the transposed matrix action.
        assert_vectors_close(
            backward,
            quaternion.to_rotation_matrix().transpose() * vector,
            1e-11,
        );
    }
}

// ---- angle between rotations ------------------------------------------------

#[test]
fn rotation_angle_to_recovers_relative_rotation() {
    let mut rng = StdRng::seed_from_u64(13);

    for _ in 0..2000 {
        let first = random_unit_quaternion(&mut rng);
        let angle = rng.gen_range(0.0..PI);
        let second = first * Quaternion::from_axis_angle(random_unit_vector(&mut rng), angle);

        assert!((first.rotation_angle_to(second) - angle).abs() < 1e-10);
        // Symmetric, and blind to the double cover on either side.
        assert!((second.rotation_angle_to(first) - angle).abs() < 1e-10);
        assert!((first.rotation_angle_to(-second) - angle).abs() < 1e-10);
        assert!(((-first).rotation_angle_to(second) - angle).abs() < 1e-10);
    }
}

#[test]
fn rotation_angle_to_takes_the_shortest_path() {
    let mut rng = StdRng::seed_from_u64(14);

    for _ in 0..1000 {
        let first = random_unit_quaternion(&mut rng);
        // Beyond pi the relative rotation is reported the short way round, as 2pi - angle.
        let angle = rng.gen_range(PI..2.0 * PI);
        let second = first * Quaternion::from_axis_angle(random_unit_vector(&mut rng), angle);
        let measured = first.rotation_angle_to(second);

        assert!((0.0..=PI + 1e-12).contains(&measured), "{measured}");
        assert!((measured - (2.0 * PI - angle)).abs() < 1e-10);
    }
}

#[test]
fn rotation_angle_to_self_is_zero_and_accurate_when_small() {
    let mut rng = StdRng::seed_from_u64(15);

    for _ in 0..1000 {
        let quaternion = random_unit_quaternion(&mut rng);
        assert!(quaternion.rotation_angle_to(quaternion) < TOL);
        // `q` and `-q` are the same rotation.
        assert!(quaternion.rotation_angle_to(-quaternion) < TOL);
    }

    // atan2 measures these to an absolute error of a few epsilon, the floor set by rounding in the
    // quaternion product itself. `acos` of the dot product cannot: 1 - cos(theta/2) drops below
    // the double epsilon for theta this small, costing about half the significant digits.
    let base = Quaternion::from_euler_zyx(0.3, -0.7, 1.2);
    for exponent in 4..=10 {
        let angle = 10f64.powi(-exponent);
        let rotated = base * Quaternion::from_axis_angle(Vector::new([0.0, 0.0, 1.0]), angle);
        let error = (base.rotation_angle_to(rotated) - angle).abs();

        assert!(error < 1e-15 + angle * 1e-12, "{angle}: {error}");
    }
}

// ---- rotation between two directions ----------------------------------------

#[test]
fn from_two_vectors_maps_one_direction_onto_the_other() {
    let mut rng = StdRng::seed_from_u64(16);

    for _ in 0..2000 {
        // Unnormalized inputs: only the directions matter.
        let from = random_unit_vector(&mut rng) * rng.gen_range(0.1..10.0);
        let to = random_unit_vector(&mut rng) * rng.gen_range(0.1..10.0);
        let rotation = Quaternion::from_two_vectors(from, to);

        assert!((rotation.norm() - 1.0).abs() < TOL);
        assert_vectors_close(
            rotation.transform_point(from.normalized()),
            to.normalized(),
            1e-10,
        );
        // Shortest: the rotation angle is exactly the angle between the two directions.
        let separation = from
            .normalized()
            .dot(to.normalized())
            .clamp(-1.0, 1.0)
            .acos();

        assert!((Quaternion::identity().rotation_angle_to(rotation) - separation).abs() < 1e-7);
    }
}

#[test]
fn from_two_vectors_handles_opposite_directions() {
    let mut rng = StdRng::seed_from_u64(17);

    for _ in 0..2000 {
        let from = random_unit_vector(&mut rng);
        let rotation = Quaternion::from_two_vectors(from, -from);

        assert!((rotation.norm() - 1.0).abs() < TOL);
        // The axis is undetermined, but every valid choice still lands on -from by a pi turn.
        assert_vectors_close(rotation.transform_point(from), -from, 1e-10);
        assert!((Quaternion::identity().rotation_angle_to(rotation) - PI).abs() < 1e-10);
    }
}

#[test]
fn from_two_vectors_near_opposite_directions() {
    let mut rng = StdRng::seed_from_u64(18);
    // Just off antiparallel, down to where the perturbation stops being representable at all.
    // There is no band of degraded accuracy in front of pi, so one flat tolerance covers the whole
    // sweep. That is the regression guard on both halves of how the rotation is built.
    //
    // On taking the half-angle from norm(a + b): with the common sqrt(2(1 + a.b)) form, `1 + a.b`
    // is delta^2/2 computed with an absolute error of eps, and the error grows as eps/delta^2
    // rather than staying flat. It reaches 6e-4 at delta = 1e-6, which this tolerance rejects.
    //
    // On taking the axis from a x (a + b): that is perpendicular to `from` however much error
    // a + b carries, so `from` can only be mislanded within the correct plane, and the eps/delta
    // relative error of a + b is scaled back down by the delta it turns through. Dividing the axis
    // by norm(a + b) up front instead of normalizing the quaternion at the end would let the
    // rounding of a small norm through, and costs the unit-norm assertion below from delta = 1e-9.
    for exponent in 1..=17 {
        let perturbation = 10f64.powi(-exponent);
        for _ in 0..200 {
            let from = random_unit_vector(&mut rng);
            let nudge = from.cross(random_unit_vector(&mut rng)).normalized();
            let to = (-from + nudge * perturbation).normalized();
            let rotation = Quaternion::from_two_vectors(from, to);

            assert!((rotation.norm() - 1.0).abs() < TOL);
            assert_vectors_close(rotation.transform_point(from), to, 1e-14);
        }
    }
}

#[test]
fn from_two_vectors_non_finite_inputs_yield_identity() {
    let unit = Vector::new([0.0, 0.0, 1.0]);

    // Documented contract: an input that is not finite carries no direction, so there is no
    // rotation to report and the identity comes back. Previously these normalized to a NaN
    // vector and every component of the result was NaN.
    for bad in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        for polluted in [
            Vector::new([bad, 0.0, 0.0]),
            Vector::new([1.0, bad, 0.0]),
            Vector::new([bad, bad, bad]),
        ] {
            assert_quaternions_close(
                Quaternion::from_two_vectors(polluted, unit),
                Quaternion::identity(),
                TOL,
            );
            assert_quaternions_close(
                Quaternion::from_two_vectors(unit, polluted),
                Quaternion::identity(),
                TOL,
            );
        }
    }
}

#[test]
fn from_two_vectors_extreme_magnitudes() {
    // Only the direction matters, so scaling either input by any finite amount must not move the
    // answer, including where the squared norm is not representable. Normalizing through norm()
    // squares first: at 1e200 that overflows to an infinite norm and the direction collapses to
    // the zero vector (which used to give a quaternion of norm 0.5), and at 1e-200 it underflows
    // to a zero norm and the direction is rejected outright (which used to give the identity).
    let reference = Quaternion::from_two_vectors(
        Vector::<3, f64>::new([1.0, 0.0, 0.0]),
        Vector::new([0.0, 1.0, 0.0]),
    );

    for scale in [
        1e200,
        1e-200,
        f64::MAX,
        f64::MIN_POSITIVE,
        5e-324, // subnormal
        1.0,
    ] {
        for (from, to) in [
            (
                Vector::<3, f64>::new([scale, 0.0, 0.0]),
                Vector::new([0.0, 1.0, 0.0]),
            ),
            (Vector::new([1.0, 0.0, 0.0]), Vector::new([0.0, scale, 0.0])),
            (
                Vector::new([scale, 0.0, 0.0]),
                Vector::new([0.0, scale, 0.0]),
            ),
        ] {
            let rotation = Quaternion::from_two_vectors(from, to);

            assert!(
                (rotation.norm() - 1.0).abs() < TOL,
                "scale {scale:e} gave a non-unit quaternion of norm {}",
                rotation.norm()
            );
            assert_quaternions_close(rotation, reference, TOL);
        }
    }
}

#[test]
fn from_two_vectors_degenerate_inputs() {
    let unit = Vector::new([0.0, 0.0, 1.0]);
    let zero = Vector::<3, f64>::zeros();

    // A direction that does not exist cannot pick out a rotation;
    // identity, as from_axis_angle does for a zero-length axis.
    assert_quaternions_close(
        Quaternion::from_two_vectors(zero, unit),
        Quaternion::identity(),
        TOL,
    );
    assert_quaternions_close(
        Quaternion::from_two_vectors(unit, zero),
        Quaternion::identity(),
        TOL,
    );
    // Already aligned: no rotation, whatever the magnitudes.
    assert_quaternions_close(
        Quaternion::from_two_vectors(unit, unit * 7.5),
        Quaternion::identity(),
        TOL,
    );
}

#[test]
fn from_two_vectors_f32_including_opposite() {
    let from = Vector::<3, f32>::new([0.0, 1.0, 0.0]);

    for to in [
        Vector::new([1.0f32, 0.0, 0.0]),
        Vector::new([0.0f32, -1.0, 0.0]),
        Vector::new([0.0f32, 1.0, 0.0]),
    ] {
        let rotation = Quaternion::from_two_vectors(from, to);

        assert!((rotation.norm() - 1.0).abs() < 1e-6);

        let rotated = rotation.transform_point(from);
        for index in 0..3 {
            assert!((rotated[index] - to[index]).abs() < 1e-6);
        }
    }
}

// ---- autodiff ---------------------------------------------------------------

/// Rotates x̂ by `angle` about ẑ and returns the x-component, which is `cos(angle)`.
fn rotated_component<T: Numeric>(angle: T) -> T {
    let quaternion = Quaternion::from_axis_angle(Vector::new([T::ZERO, T::ZERO, T::ONE]), angle);
    quaternion.transform_point(Vector::new([T::ONE, T::ZERO, T::ZERO]))[0]
}

#[test]
fn ad_matches_finite_difference() {
    let mut rng = StdRng::seed_from_u64(12);
    for _ in 0..1000 {
        let angle = rng.gen_range(-PI..PI);
        let autodiff = rotated_component(Dual::variable(angle)).deriv;
        let step = 1e-6;
        let finite_difference =
            (rotated_component(angle + step) - rotated_component(angle - step)) / (2.0 * step);
        assert!((autodiff + angle.sin()).abs() < 1e-9); // exact derivative is -sin(angle)
        assert!((autodiff - finite_difference).abs() < 1e-6); // FD's own error floor
    }
}

/// Rotates x̂ by `from_scaled_axis(angle·ẑ)` and returns the y-component, which is `sin(angle)`
/// (chosen so the derivative at `angle = 0` is a non-zero 1). Differentiating here exercises the
/// small-angle branch that once produced NaN via `sqrt(0)`.
fn scaled_axis_component<T: Numeric>(angle: T) -> T {
    let rotation_vector = Vector::new([T::ZERO, T::ZERO, T::ONE]) * angle;
    let quaternion = Quaternion::from_scaled_axis(rotation_vector);
    quaternion.transform_point(Vector::new([T::ONE, T::ZERO, T::ZERO]))[1]
}

#[test]
fn ad_through_scaled_axis_finite_at_zero() {
    // Regression: from_scaled_axis must give a finite derivative at and near ω = 0. The mapped
    // component is sin(angle) about ẑ, whose derivative at 0 is 1.
    let derivative_at_zero = scaled_axis_component(Dual::variable(0.0)).deriv;
    assert!(derivative_at_zero.is_finite());
    assert!((derivative_at_zero - 1.0).abs() < 1e-12);

    for &angle in &[1e-9, 1e-6, 1e-3, 0.5] {
        let autodiff = scaled_axis_component(Dual::variable(angle)).deriv;
        assert!(autodiff.is_finite());
        assert!((autodiff - angle.cos()).abs() < 1e-6);
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
        let quaternion = Quaternion::from_axis_angle(axis, angle);

        // Orthonormal rotation matrix.
        let rotation_matrix = quaternion.to_rotation_matrix();
        let should_be_identity = rotation_matrix.transpose() * rotation_matrix;
        for row in 0..3 {
            for column in 0..3 {
                let expected = if row == column { 1.0 } else { 0.0 };
                assert!((should_be_identity[(row, column)] - expected).abs() < 1e-4);
            }
        }

        // from_scaled_axis ∘ to_scaled_axis round trip within f32 tolerance.
        let recovered = Quaternion::from_scaled_axis(quaternion.to_scaled_axis());
        let left_matrix = quaternion.to_rotation_matrix();
        let right_matrix = recovered.to_rotation_matrix();
        for row in 0..3 {
            for column in 0..3 {
                assert!((left_matrix[(row, column)] - right_matrix[(row, column)]).abs() < 1e-4);
            }
        }
    }
}

//---- near-zero identities (f32 + f64) ----------------------------------------

fn near_zero_scaled_axis_roundtrip<T: Numeric>() {
    let zero = Vector::new([T::ZERO, T::ZERO, T::ZERO]);
    let q0 = Quaternion::from_scaled_axis(zero);
    assert!((q0.norm() - T::ONE).abs() < T::from_f64(1e-5));
    let back0 = q0.to_scaled_axis();
    for &comp in back0.as_array() {
        assert!(comp.abs() < T::from_f64(1e-5));
    }

    // Sub threshold angle stays on Taylor path; round-trip stays finite/unit.
    let tiny = Vector::new([T::from_f64(1e-9), T::ZERO, T::ZERO]);
    let q = Quaternion::from_scaled_axis(tiny);
    assert!(q.norm().is_finite());
    assert!((q.norm() - T::ONE).abs() < T::from_f64(1e-4));
    let back = Quaternion::from_scaled_axis(q.to_scaled_axis());
    for (a, b) in q.as_array().iter().zip(back.as_array().iter()) {
        assert!((*a - *b).abs() < T::from_f64(1e-4));
    }

    let [tx, ty, tz] = *tiny.as_array();
    let q_exp = Quaternion::new(T::ZERO, tx, ty, tz).exp();
    let q_ln = q_exp.ln();
    assert!(q_ln.w().is_finite() && q_ln.x().is_finite());
}

#[test]
fn near_zero_identities_f64() {
    near_zero_scaled_axis_roundtrip::<f64>();
}

#[test]
fn near_zero_identities_f32() {
    near_zero_scaled_axis_roundtrip::<f32>();
}
