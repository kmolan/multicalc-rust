//! Lie-group tests: group laws, exp/log round trips across the θ=0 and θ=π regions, adjoint
//! identities, hat/vee, act-vs-matrix consistency, geodesic interpolation, AD-vs-FD, and f32
//! identity coverage.

use std::f64::consts::PI;

use multicalc::linear_algebra::{Matrix, Matrix3D, Vector, Vector2D, Vector3D, Vector6D};
use multicalc::scalar::{Dual, Numeric};
use multicalc::spatial::{SE2, SE3, SO2, SO3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const TOL: f64 = 1e-10;

// ---- helpers ----------------------------------------------------------------

fn random_vector3(rng: &mut StdRng) -> Vector3D {
    Vector::new([
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
    ])
}

fn random_unit_vector3(rng: &mut StdRng) -> Vector3D {
    loop {
        let vector = random_vector3(rng);
        let norm = vector.dot(vector).sqrt();
        if norm > 1e-3 {
            return vector * norm.recip();
        }
    }
}

fn random_twist6(rng: &mut StdRng) -> Vector6D {
    Vector::new([
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
    ])
}

fn random_so3(rng: &mut StdRng) -> SO3<f64> {
    SO3::exp(random_unit_vector3(rng) * rng.gen_range(-2.5..2.5))
}

fn random_se3(rng: &mut StdRng) -> SE3<f64> {
    SE3::from_parts(
        random_so3(rng),
        Vector::new([
            rng.gen_range(-2.0..2.0),
            rng.gen_range(-2.0..2.0),
            rng.gen_range(-2.0..2.0),
        ]),
    )
}

fn random_so2(rng: &mut StdRng) -> SO2<f64> {
    SO2::from_angle(rng.gen_range(-PI..PI))
}

fn random_se2(rng: &mut StdRng) -> SE2<f64> {
    SE2::from_parts(
        random_so2(rng),
        Vector::new([rng.gen_range(-2.0..2.0), rng.gen_range(-2.0..2.0)]),
    )
}

/// Translation part of every `Q(ρ, φ)` reference below. Must match `RHO` in
/// `tools/qa/src/bin/spatial_small_angle.py`, which generated those matrices.
const RHO: Vector3D = Vector::new([0.5, 0.25, -0.75]);

/// Barfoot's `Q(ρ, φ)`: the top-right 3×3 block of the SE(3) left Jacobian. `q_matrix_se3` is
/// `pub(crate)`, so assembling the twist and slicing the block out is the only route to it from
/// the test crate. The tangent ordering is `[ρ; φ]`, linear part first.
fn q_block(rho: Vector3D, phi: Vector3D) -> Matrix3D {
    let xi = Vector::new([rho[0], rho[1], rho[2], phi[0], phi[1], phi[2]]);
    let jacobian = SE3::left_jacobian(xi);
    Matrix::from_fn(|row, column| jacobian[(row, column + 3)])
}

fn assert_entries_close<const R: usize, const C: usize>(
    first: Matrix<R, C, f64>,
    second: Matrix<R, C, f64>,
    tolerance: f64,
) {
    for row in 0..R {
        for column in 0..C {
            let left = first[(row, column)];
            let right = second[(row, column)];
            assert!(
                (left - right).abs() < tolerance,
                "({row},{column}): {left} vs {right}"
            );
        }
    }
}

fn assert_components_close<const N: usize>(
    first: Vector<N, f64>,
    second: Vector<N, f64>,
    tolerance: f64,
) {
    for index in 0..N {
        let left = first[index];
        let right = second[index];
        assert!(
            (left - right).abs() < tolerance,
            "[{index}]: {left} vs {right}"
        );
    }
}

// ---- SO(3) ------------------------------------------------------------------

#[test]
fn so3_group_laws() {
    let mut rng = StdRng::seed_from_u64(1);
    for _ in 0..200 {
        let first = random_so3(&mut rng);
        let second = random_so3(&mut rng);
        let third = random_so3(&mut rng);
        assert_entries_close(
            ((first * second) * third).to_matrix(),
            (first * (second * third)).to_matrix(),
            TOL,
        );
        assert_entries_close(
            (first * SO3::identity()).to_matrix(),
            first.to_matrix(),
            TOL,
        );
        assert_entries_close(
            (first * first.inverse()).to_matrix(),
            SO3::identity().to_matrix(),
            TOL,
        );
    }
}

#[test]
fn so3_exp_log_roundtrip() {
    let mut rng = StdRng::seed_from_u64(2);
    for _ in 0..200 {
        let axis = random_unit_vector3(&mut rng);
        for &angle in &[1e-9, 1e-4, 0.5, 2.0, PI - 1e-6] {
            let rotation_vector = axis * angle;
            assert_components_close(SO3::exp(rotation_vector).log(), rotation_vector, 1e-7);
        }
    }
}

#[test]
fn so3_adjoint_identity() {
    let mut rng = StdRng::seed_from_u64(3);
    for _ in 0..200 {
        let rotation = random_so3(&mut rng);
        let twist = random_vector3(&mut rng) * 0.5;
        let left_side = SO3::exp(rotation.adjoint() * twist).to_matrix();
        let right_side = (rotation * SO3::exp(twist) * rotation.inverse()).to_matrix();
        assert_entries_close(left_side, right_side, 1e-9);
    }
}

#[test]
fn so3_hat_vee_roundtrip() {
    let mut rng = StdRng::seed_from_u64(4);
    for _ in 0..100 {
        let rotation_vector = random_vector3(&mut rng);
        assert_components_close(SO3::vee(SO3::hat(rotation_vector)), rotation_vector, TOL);
    }
}

#[test]
fn so3_act_matches_matrix() {
    let mut rng = StdRng::seed_from_u64(5);
    for _ in 0..100 {
        let rotation = random_so3(&mut rng);
        let point = random_vector3(&mut rng);
        assert_components_close(rotation.act(point), rotation.to_matrix() * point, TOL);
    }
}

#[test]
fn so3_interpolate_endpoints() {
    let mut rng = StdRng::seed_from_u64(6);
    for _ in 0..100 {
        let first = random_so3(&mut rng);
        let second = random_so3(&mut rng);
        assert_entries_close(
            first.interpolate(second, 0.0).to_matrix(),
            first.to_matrix(),
            TOL,
        );
        assert_entries_close(
            first.interpolate(second, 1.0).to_matrix(),
            second.to_matrix(),
            1e-9,
        );
    }
}

#[test]
fn so3_left_right_jacobian_relation() {
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..100 {
        let rotation_vector = random_vector3(&mut rng) * 1.5;
        // J_l(φ) = exp(φ) · J_r(φ)
        assert_entries_close(
            SO3::left_jacobian(rotation_vector),
            SO3::exp(rotation_vector).to_matrix() * SO3::right_jacobian(rotation_vector),
            1e-9,
        );
    }
}

// ---- SE(3) ------------------------------------------------------------------

#[test]
fn se3_group_laws() {
    let mut rng = StdRng::seed_from_u64(10);
    for _ in 0..200 {
        let first = random_se3(&mut rng);
        let second = random_se3(&mut rng);
        let third = random_se3(&mut rng);
        assert_entries_close(
            ((first * second) * third).to_matrix(),
            (first * (second * third)).to_matrix(),
            1e-9,
        );
        assert_entries_close(
            (first * SE3::identity()).to_matrix(),
            first.to_matrix(),
            TOL,
        );
        assert_entries_close(
            (first * first.inverse()).to_matrix(),
            SE3::identity().to_matrix(),
            1e-9,
        );
    }
}

#[test]
fn se3_exp_log_roundtrip() {
    let mut rng = StdRng::seed_from_u64(11);
    for _ in 0..300 {
        let axis = random_unit_vector3(&mut rng);
        for &angle in &[1e-9, 1e-4, 0.7, 2.0, PI - 1e-6] {
            let translation = random_vector3(&mut rng);
            let twist = Vector::new([
                translation[0],
                translation[1],
                translation[2],
                axis[0] * angle,
                axis[1] * angle,
                axis[2] * angle,
            ]);
            assert_components_close(SE3::exp(twist).log(), twist, 1e-6);
        }
    }
}

#[test]
fn se3_adjoint_identity() {
    let mut rng = StdRng::seed_from_u64(12);
    for _ in 0..200 {
        let pose = random_se3(&mut rng);
        let twist = random_twist6(&mut rng) * 0.3;
        let left_side = SE3::exp(pose.adjoint() * twist).to_matrix();
        let right_side = (pose * SE3::exp(twist) * pose.inverse()).to_matrix();
        assert_entries_close(left_side, right_side, 1e-8);
    }
}

#[test]
fn se3_act_matches_homogeneous_matrix() {
    let mut rng = StdRng::seed_from_u64(13);
    for _ in 0..100 {
        let pose = random_se3(&mut rng);
        let point = random_vector3(&mut rng);
        let homogeneous = Vector::new([point[0], point[1], point[2], 1.0]);
        let product = pose.to_matrix() * homogeneous;
        assert_components_close(
            pose.act(point),
            Vector::new([product[0], product[1], product[2]]),
            TOL,
        );
    }
}

#[test]
fn se3_matrix_roundtrip() {
    let mut rng = StdRng::seed_from_u64(14);
    for _ in 0..100 {
        let pose = random_se3(&mut rng);
        let recovered = SE3::try_from_matrix(pose.to_matrix()).unwrap();
        assert_entries_close(recovered.to_matrix(), pose.to_matrix(), 1e-9);
    }
}

#[test]
fn se3_matrix_constructor_rejects_non_homogeneous_bottom_row() {
    for column in 0..4 {
        let mut projective = SE3::<f64>::identity().to_matrix();
        projective[(3, column)] = if column == 3 { 0.5 } else { 0.25 };
        assert!(SE3::try_from_matrix(projective).is_none());
    }

    let mut non_finite = SE3::<f64>::identity().to_matrix();
    non_finite[(3, 0)] = f64::NAN;
    assert!(SE3::try_from_matrix(non_finite).is_none());
}

#[test]
fn so3_matrix_constructor_rejects_non_rotations() {
    let reflection = Matrix::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]);
    assert!(SO3::try_from_matrix(reflection).is_none());

    let shear = Matrix::new([[1.0, 0.25, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    assert!(SO3::try_from_matrix(shear).is_none());

    let scaled = Matrix::new([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]);
    assert!(SO3::try_from_matrix(scaled).is_none());

    let non_finite = Matrix::new([[1.0, 0.0, f64::NAN], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    assert!(SO3::try_from_matrix(non_finite).is_none());
}

#[test]
fn se3_matrix_constructor_rejects_invalid_rotation_and_translation() {
    let mut reflection = SE3::<f64>::identity().to_matrix();
    reflection[(2, 2)] = -1.0;
    assert!(SE3::try_from_matrix(reflection).is_none());

    for row in 0..3 {
        let mut non_finite = SE3::<f64>::identity().to_matrix();
        non_finite[(row, 3)] = match row {
            0 => f64::NAN,
            1 => f64::INFINITY,
            _ => f64::NEG_INFINITY,
        };
        assert!(SE3::try_from_matrix(non_finite).is_none());
    }
}

#[test]
fn so2_matrix_constructor_rejects_reflection() {
    let reflection = Matrix::new([[1.0, 0.0], [0.0, -1.0]]);
    assert!(SO2::try_from_matrix(reflection).is_none());
}

#[test]
fn so2_matrix_constructor_rejects_shear() {
    let shear = Matrix::new([[1.0, 0.25], [0.0, 1.0]]);
    assert!(SO2::try_from_matrix(shear).is_none());
}

#[test]
fn so2_matrix_constructor_rejects_scale_and_non_finite_columns() {
    let scaled = Matrix::new([[2.0, 0.0], [0.0, 2.0]]);
    assert!(SO2::try_from_matrix(scaled).is_none());

    let non_finite = Matrix::new([[1.0, f64::NAN], [0.0, 1.0]]);
    assert!(SO2::try_from_matrix(non_finite).is_none());

    let extreme = Matrix::new([[f64::MAX, 0.0], [f64::MAX, 1.0]]);
    assert!(SO2::try_from_matrix(extreme).is_none());
}

#[test]
fn se2_matrix_constructor_validates_rotation_block() {
    let pose = SE2::from_parts(SO2::from_angle(0.3), Vector2D::new([1.0, 2.0]));
    assert!(SE2::try_from_matrix(pose.to_matrix()).is_some());

    let reflection = Matrix::new([[1.0, 0.0, 1.0], [0.0, -1.0, 2.0], [0.0, 0.0, 1.0]]);
    assert!(SE2::try_from_matrix(reflection).is_none());
}

#[test]
fn se2_matrix_constructor_rejects_invalid_homogeneous_components() {
    for column in 0..3 {
        let mut projective = SE2::<f64>::identity().to_matrix();
        projective[(2, column)] = if column == 2 { 0.5 } else { 0.25 };
        assert!(SE2::try_from_matrix(projective).is_none());
    }

    let mut non_finite_bottom_row = SE2::<f64>::identity().to_matrix();
    non_finite_bottom_row[(2, 0)] = f64::NAN;
    assert!(SE2::try_from_matrix(non_finite_bottom_row).is_none());

    for row in 0..2 {
        let mut non_finite = SE2::<f64>::identity().to_matrix();
        non_finite[(row, 2)] = if row == 0 { f64::NAN } else { f64::INFINITY };
        assert!(SE2::try_from_matrix(non_finite).is_none());
    }
}

#[test]
fn matrix_constructors_accept_valid_f32_roundtrips() {
    let rotation = SO2::<f32>::from_angle(0.3);
    assert!(SO2::try_from_matrix(rotation.to_matrix()).is_some());

    let rotation = SO3::<f32>::exp(Vector::new([0.1, -0.2, 0.3]));
    assert!(SO3::try_from_matrix(rotation.to_matrix()).is_some());

    let pose = SE3::<f32>::identity();
    assert!(SE3::try_from_matrix(pose.to_matrix()).is_some());
}

#[test]
fn matrix_constructors_accept_roundoff_sized_drift() {
    let tolerance = <f64 as Numeric>::EPSILON_X4;

    let mut rotation = SO2::<f64>::from_angle(0.3).to_matrix();
    rotation[(0, 0)] += tolerance;
    assert!(SO2::try_from_matrix(rotation).is_some());

    let mut rotation = SO3::<f64>::exp(Vector::new([0.1, -0.2, 0.3])).to_matrix();
    rotation[(0, 0)] += tolerance;
    assert!(SO3::try_from_matrix(rotation).is_some());

    let mut pose = SE2::<f64>::identity().to_matrix();
    pose[(2, 0)] = tolerance;
    assert!(SE2::try_from_matrix(pose).is_some());

    let mut pose = SE3::<f64>::identity().to_matrix();
    pose[(3, 0)] = tolerance;
    assert!(SE3::try_from_matrix(pose).is_some());
}

#[test]
fn se3_hat_vee_roundtrip() {
    let mut rng = StdRng::seed_from_u64(15);
    for _ in 0..100 {
        let twist = random_twist6(&mut rng);
        assert_components_close(SE3::vee(SE3::hat(twist)), twist, TOL);
    }
}

#[test]
fn se3_interpolate_endpoints() {
    let mut rng = StdRng::seed_from_u64(16);
    for _ in 0..100 {
        let first = random_se3(&mut rng);
        let second = random_se3(&mut rng);
        assert_entries_close(
            first.interpolate(second, 0.0).to_matrix(),
            first.to_matrix(),
            1e-9,
        );
        assert_entries_close(
            first.interpolate(second, 1.0).to_matrix(),
            second.to_matrix(),
            1e-8,
        );
    }
}

// ---- SO(2) / SE(2) ----------------------------------------------------------

#[test]
fn so2_group_and_roundtrip() {
    let mut rng = StdRng::seed_from_u64(20);
    for _ in 0..200 {
        let first = random_so2(&mut rng);
        let second = random_so2(&mut rng);
        let third = random_so2(&mut rng);
        assert_entries_close(
            ((first * second) * third).to_matrix(),
            (first * (second * third)).to_matrix(),
            TOL,
        );
        assert_entries_close(
            (first * first.inverse()).to_matrix(),
            SO2::identity().to_matrix(),
            TOL,
        );
        assert_entries_close(
            SO2::try_from_matrix(first.to_matrix()).unwrap().to_matrix(),
            first.to_matrix(),
            TOL,
        );
        for &angle in &[1e-9, 0.3, PI - 1e-6] {
            assert!((SO2::exp(angle).log() - angle).abs() < 1e-9);
        }
        let point = Vector::new([rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)]);
        assert_components_close(first.act(point), first.to_matrix() * point, TOL);
    }
}

#[test]
fn se2_group_and_roundtrip() {
    let mut rng = StdRng::seed_from_u64(21);
    for _ in 0..300 {
        let first = random_se2(&mut rng);
        let second = random_se2(&mut rng);
        let third = random_se2(&mut rng);
        assert_entries_close(
            ((first * second) * third).to_matrix(),
            (first * (second * third)).to_matrix(),
            TOL,
        );
        assert_entries_close(
            (first * first.inverse()).to_matrix(),
            SE2::identity().to_matrix(),
            TOL,
        );
        for &angle in &[1e-9, 0.4, PI - 1e-6] {
            let twist = Vector::new([rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), angle]);
            assert_components_close(SE2::exp(twist).log(), twist, 1e-7);
        }
        let point = Vector::new([rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)]);
        let homogeneous = Vector::new([point[0], point[1], 1.0]);
        let product = first.to_matrix() * homogeneous;
        assert_components_close(first.act(point), Vector::new([product[0], product[1]]), TOL);
    }
}

#[test]
fn se2_adjoint_identity() {
    let mut rng = StdRng::seed_from_u64(22);
    for _ in 0..200 {
        let pose = random_se2(&mut rng);
        let twist = Vector::new([
            rng.gen_range(-0.5..0.5),
            rng.gen_range(-0.5..0.5),
            rng.gen_range(-0.5..0.5),
        ]);
        let left_side = SE2::exp(pose.adjoint() * twist).to_matrix();
        let right_side = (pose * SE2::exp(twist) * pose.inverse()).to_matrix();
        assert_entries_close(left_side, right_side, 1e-9);
    }
}

#[test]
fn so2_normalized_removes_composition_drift() {
    let step = SO2::<f64>::from_angle(0.3);
    let mut rotation = SO2::<f64>::identity();

    // Repeated rotation to accumulate drift.
    for _ in 0..10_000 {
        rotation = rotation.compose(step);
    }

    let (drifted_c, drifted_s) = rotation.cos_sin();
    let drifted_norm = rotation.norm();
    let tolerance = <f64 as multicalc::Numeric>::EPSILON;

    // The accumulated rotation has drifted away from the unit circle.
    assert!((drifted_norm - 1.0).abs() > tolerance);

    let normalized = rotation.normalized();
    let (normalized_c, normalized_s) = normalized.cos_sin();
    let normalized_norm = normalized.norm();

    // Normalization pulls SO2 back onto the unit circle.
    assert!((normalized_norm - 1.0).abs() <= tolerance);

    // Scaling the normalized components to reconstruct the input.
    assert!((normalized_c * drifted_norm - drifted_c).abs() <= tolerance);
    assert!((normalized_s * drifted_norm - drifted_s).abs() <= tolerance);
}

#[test]
fn so2_norm() {
    let tolerance = <f64 as multicalc::Numeric>::EPSILON;

    // The identity has an exact unit norm.
    assert_eq!(SO2::<f64>::identity().norm(), 1.0);

    // Different rotations have norm 1.
    for angle in [-3.0, -0.3, 0.0, 0.3, 3.0] {
        let rotation = SO2::<f64>::from_angle(angle);
        assert!((rotation.norm() - 1.0).abs() <= tolerance);
    }

    // The norm remains stable for a very small angle.
    let rotation = SO2::<f64>::from_angle(<f64 as multicalc::Numeric>::EPSILON_X30);
    assert!((rotation.norm() - 1.0).abs() <= tolerance);
}

// ---- autodiff ---------------------------------------------------------------

#[test]
fn so3_exp_ad_vs_fd() {
    // d/dφ_k of exp(φ).act(p): autodiff (Dual) vs central finite difference.
    let base_rotation_vector = [0.2_f64, -0.1, 0.35];
    let point = [0.5_f64, 0.3, -0.7];
    let step = 1e-6;
    for variable_index in 0..3 {
        let rotation_vector = Vector::new([
            if variable_index == 0 {
                Dual::variable(base_rotation_vector[0])
            } else {
                Dual::constant(base_rotation_vector[0])
            },
            if variable_index == 1 {
                Dual::variable(base_rotation_vector[1])
            } else {
                Dual::constant(base_rotation_vector[1])
            },
            if variable_index == 2 {
                Dual::variable(base_rotation_vector[2])
            } else {
                Dual::constant(base_rotation_vector[2])
            },
        ]);
        let dual_point = Vector::new([
            Dual::constant(point[0]),
            Dual::constant(point[1]),
            Dual::constant(point[2]),
        ]);
        let outputs = SO3::exp(rotation_vector).act(dual_point);

        let mut plus = base_rotation_vector;
        let mut minus = base_rotation_vector;
        plus[variable_index] += step;
        minus[variable_index] -= step;
        let outputs_at_plus = SO3::exp(Vector::new(plus)).act(Vector::new(point));
        let outputs_at_minus = SO3::exp(Vector::new(minus)).act(Vector::new(point));
        for output_index in 0..3 {
            let finite_difference =
                (outputs_at_plus[output_index] - outputs_at_minus[output_index]) / (2.0 * step);
            assert!(
                (outputs[output_index].deriv - finite_difference).abs() < 1e-6,
                "variable {variable_index} output {output_index}: {} vs {}",
                outputs[output_index].deriv,
                finite_difference
            );
        }
    }
}

#[test]
fn so3_exp_derivative_finite_at_zero() {
    // At φ = 0 the exp map is smooth; a naive sqrt-based path would give a NaN derivative here.
    let rotation_vector = Vector::new([
        Dual::variable(0.0),
        Dual::constant(0.0),
        Dual::constant(0.0),
    ]);
    let dual_point = Vector::new([
        Dual::constant(1.0),
        Dual::constant(0.0),
        Dual::constant(0.0),
    ]);
    let outputs = SO3::exp(rotation_vector).act(dual_point);
    for output_index in 0..3 {
        assert!(outputs[output_index].deriv.is_finite());
    }
}

// ---- f32 identity coverage --------------------------------------------------

#[test]
fn f32_identity_coverage() {
    let rotation_vector = Vector::new([0.2_f32, -0.3, 0.5]);
    let recovered = SO3::exp(rotation_vector).log();
    for index in 0..3 {
        assert!((recovered[index] - rotation_vector[index]).abs() < 1e-4);
    }

    let twist = Vector::new([0.1_f32, -0.2, 0.3, 0.2, -0.3, 0.5]);
    let recovered_twist = SE3::exp(twist).log();
    for index in 0..6 {
        assert!((recovered_twist[index] - twist[index]).abs() < 1e-4);
    }

    // Rotation matrix orthonormality: RᵀR = I.
    let rotation = SO3::exp(rotation_vector).to_matrix();
    let should_be_identity = rotation.transpose() * rotation;
    for row in 0..3 {
        for column in 0..3 {
            let expected = if row == column { 1.0 } else { 0.0 };
            assert!((should_be_identity[(row, column)] - expected).abs() < 1e-5);
        }
    }
}

// ---- value goldens (exact, scipy-equivalent) --------------------------------

#[test]
fn so3_exp_goldens() {
    // exp(θ·axis).to_matrix() == R.from_rotvec([...]).as_matrix()
    let rotation_about_z = SO3::<f64>::exp(Vector::new([0.0, 0.0, PI / 2.0]));
    assert_entries_close(
        rotation_about_z.to_matrix(),
        Matrix::new([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        1e-12,
    );
    let rotation_about_x = SO3::<f64>::exp(Vector::new([PI / 2.0, 0.0, 0.0]));
    assert_entries_close(
        rotation_about_x.to_matrix(),
        Matrix::new([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
        1e-12,
    );
    let rotation_about_y = SO3::<f64>::exp(Vector::new([0.0, PI / 2.0, 0.0]));
    assert_entries_close(
        rotation_about_y.to_matrix(),
        Matrix::new([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
        1e-12,
    );
}

// ---- Lie Jacobians (added) --------------------------------------------------

#[test]
fn so3_jacobian_inverse_roundtrip() {
    let mut rng = StdRng::seed_from_u64(20);
    for _ in 0..200 {
        let axis = random_unit_vector3(&mut rng);
        for &angle in &[1e-9, 1e-4, 0.5, 2.0, PI - 1e-6] {
            let rotation_vector = axis * angle;
            assert_entries_close(
                SO3::left_jacobian(rotation_vector) * SO3::left_jacobian_inverse(rotation_vector),
                Matrix::identity(),
                1e-9,
            );
            assert_entries_close(
                SO3::right_jacobian(rotation_vector) * SO3::right_jacobian_inverse(rotation_vector),
                Matrix::identity(),
                1e-9,
            );
        }
    }
}

#[test]
fn se3_jacobian_identities() {
    let mut rng = StdRng::seed_from_u64(21);
    for _ in 0..200 {
        let twist = random_twist6(&mut rng);
        // J_l · J_l⁻¹ = I and J_r = J_l(−ξ).
        assert_entries_close(
            SE3::left_jacobian(twist) * SE3::left_jacobian_inverse(twist),
            Matrix::identity(),
            1e-8,
        );
        assert_entries_close(SE3::right_jacobian(twist), SE3::left_jacobian(-twist), TOL);
        // Adjoint identity: Ad_{exp(ξ)} = J_l(ξ) · J_r(ξ)⁻¹.
        assert_entries_close(
            SE3::exp(twist).adjoint(),
            SE3::left_jacobian(twist) * SE3::right_jacobian_inverse(twist),
            1e-8,
        );
    }
}

#[test]
fn se3_left_jacobian_matches_finite_difference() {
    // Defining property: log(exp(ξ + h·eᵢ) · exp(ξ)⁻¹)/h → column i of J_l(ξ).
    let twist = Vector::new([0.2_f64, -0.1, 0.3, 0.25, -0.15, 0.4]);
    let left_jacobian = SE3::left_jacobian(twist);
    let step = 1e-6;
    for column in 0..6 {
        let mut plus = twist;
        plus[column] += step;
        let difference = (SE3::exp(plus) * SE3::exp(twist).inverse()).log() * (1.0 / step);
        for row in 0..6 {
            assert!(
                (difference[row] - left_jacobian[(row, column)]).abs() < 1e-4,
                "({row},{column})"
            );
        }
    }
}

#[test]
fn se3_left_jacobian_finite_at_zero_under_dual() {
    let twist = Vector::new([
        Dual::variable(0.0),
        Dual::constant(0.0),
        Dual::constant(0.0),
        Dual::constant(0.0),
        Dual::constant(0.0),
        Dual::constant(0.0),
    ]);
    let left_jacobian = SE3::left_jacobian(twist);
    for row in 0..6 {
        for column in 0..6 {
            let cell = left_jacobian[(row, column)];
            assert!(cell.value.is_finite() && cell.deriv.is_finite());
        }
    }
}

#[test]
fn se2_jacobian_identities_and_fd() {
    let mut rng = StdRng::seed_from_u64(22);
    for _ in 0..200 {
        let twist = Vector::new([
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-2.5..2.5),
        ]);
        assert_entries_close(
            SE2::left_jacobian(twist) * SE2::left_jacobian_inverse(twist),
            Matrix::identity(),
            1e-9,
        );
        assert_entries_close(
            SE2::exp(twist).adjoint(),
            SE2::left_jacobian(twist) * SE2::right_jacobian_inverse(twist),
            1e-8,
        );
    }
    // Finite-difference ground truth on one configuration.
    let twist = Vector::new([0.3_f64, -0.4, 0.5]);
    let left_jacobian = SE2::left_jacobian(twist);
    let step = 1e-6;
    for column in 0..3 {
        let mut plus = twist;
        plus[column] += step;
        let difference = (SE2::exp(plus) * SE2::exp(twist).inverse()).log() * (1.0 / step);
        for row in 0..3 {
            assert!(
                (difference[row] - left_jacobian[(row, column)]).abs() < 1e-4,
                "({row},{column})"
            );
        }
    }
}

#[test]
fn so2_jacobians_are_one() {
    assert!((SO2::left_jacobian(0.7_f64) - 1.0).abs() < 1e-15);
    assert!((SO2::right_jacobian(0.7_f64) - 1.0).abs() < 1e-15);
    assert!((SO2::left_jacobian_inverse(-0.3_f64) - 1.0).abs() < 1e-15);
    assert!((SO2::right_jacobian_inverse(-0.3_f64) - 1.0).abs() < 1e-15);
}

#[test]
fn se3_to_matrix_goldens() {
    // Pure translation: identity rotation, translation (1, 2, 3).
    let translation_only = SE3::<f64>::exp(Vector::new([1.0, 2.0, 3.0, 0.0, 0.0, 0.0]));
    assert_entries_close(
        translation_only.to_matrix(),
        Matrix::new([
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]),
        1e-12,
    );
    // Pure rotation twist: Rz(90°), zero translation (J_l · 0 = 0).
    let rotation = SE3::<f64>::exp(Vector::new([0.0, 0.0, 0.0, 0.0, 0.0, PI / 2.0]));
    assert_entries_close(
        rotation.to_matrix(),
        Matrix::new([
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]),
        1e-12,
    );
}

#[test]
fn near_zero_exp_log_and_jacobians_f64() {
    let z3 = Vector::new([0.0_f64, 0.0, 0.0]);
    assert_components_close(SO3::exp(z3).log(), z3, 1e-14);
    assert_entries_close(
        SO3::left_jacobian(z3) * SO3::left_jacobian_inverse(z3),
        Matrix::identity(),
        1e-14,
    );

    let tiny3 = Vector::new([1e-9_f64, 0.0, 0.0]);
    assert_components_close(SO3::exp(tiny3).log(), tiny3, 1e-12);
    assert_entries_close(
        SO3::left_jacobian(tiny3) * SO3::left_jacobian_inverse(tiny3),
        Matrix::identity(),
        1e-9,
    );

    let z6 = Vector::new([0.0_f64; 6]);
    assert_components_close(SE3::exp(z6).log(), z6, 1e-14);
    let tiny6 = Vector::new([0.0, 0.0, 0.0, 1e-9_f64, 0.0, 0.0]);
    assert_components_close(SE3::exp(tiny6).log(), tiny6, 1e-12);

    let z3_se2 = Vector::new([0.0_f64, 0.0, 0.0]);
    assert_components_close(SE2::exp(z3_se2).log(), z3_se2, 1e-14);
    let tiny_se2 = Vector::new([0.0, 0.0, 1e-9_f64]);
    assert_components_close(SE2::exp(tiny_se2).log(), tiny_se2, 1e-12);
}

#[test]
fn near_zero_exp_log_and_jacobians_f32() {
    let z3 = Vector::new([0.0_f32, 0.0, 0.0]);
    let back = SO3::exp(z3).log();
    for i in 0..3 {
        assert!(back[i].abs() < 1e-5);
    }
    let jl = SO3::left_jacobian(z3) * SO3::left_jacobian_inverse(z3);
    for i in 0..3 {
        for j in 0..3 {
            let expect = if i == j { 1.0 } else { 0.0 };
            assert!((jl[(i, j)] - expect).abs() < 1e-4);
        }
    }

    let tiny3 = Vector::new([1e-9_f32, 0.0, 0.0]);
    let back_tiny = SO3::exp(tiny3).log();
    for i in 0..3 {
        assert!((back_tiny[i] - tiny3[i]).abs() < 1e-5);
    }
    let jl_tiny = SO3::left_jacobian(tiny3) * SO3::left_jacobian_inverse(tiny3);
    for i in 0..3 {
        for j in 0..3 {
            let expect = if i == j { 1.0 } else { 0.0 };
            assert!((jl_tiny[(i, j)] - expect).abs() < 1e-4);
        }
    }

    let z6 = Vector::new([0.0_f32; 6]);
    let back6 = SE3::exp(z6).log();
    for i in 0..6 {
        assert!(back6[i].abs() < 1e-5);
    }
    let tiny6 = Vector::new([0.0, 0.0, 0.0, 1e-9_f32, 0.0, 0.0]);
    let back6_tiny = SE3::exp(tiny6).log();
    for i in 0..6 {
        assert!((back6_tiny[i] - tiny6[i]).abs() < 1e-5);
    }

    let z3_se2 = Vector::new([0.0_f32, 0.0, 0.0]);
    let back_se2 = SE2::exp(z3_se2).log();
    for i in 0..3 {
        assert!(back_se2[i].abs() < 1e-5);
    }
    let tiny_se2 = Vector::new([0.0, 0.0, 1e-9_f32]);
    let back_se2_tiny = SE2::exp(tiny_se2).log();
    for i in 0..3 {
        assert!((back_se2_tiny[i] - tiny_se2[i]).abs() < 1e-5);
    }
}

#[test]
fn so2_default_is_identity() {
    let default_so2 = SO2::default();
    let so2 = SO2::<f64>::from_angle(0.3);

    assert_eq!(default_so2, SO2::identity());
    assert_eq!(so2 * default_so2, so2);
    assert_eq!(default_so2 * so2, so2);
}

#[test]
fn so3_default_is_identity() {
    let default_so3 = SO3::default();
    let so3 = SO3::<f64>::from_quaternion(multicalc::Quaternion::from_array([1.0, 2.0, 3.0, 4.0]));

    assert_eq!(default_so3, SO3::identity());
    assert_eq!(so3 * default_so3, so3);
    assert_eq!(default_so3 * so3, so3);
}

#[test]
fn se2_default_is_identity() {
    let default_se2 = SE2::default();
    let se2 = SE2::from_parts(SO2::from_angle(0.3), Vector2D::new([1.0, 2.0]));

    assert_eq!(default_se2, SE2::identity());
    assert_eq!(default_se2 * se2, se2);
    assert_eq!(se2 * default_se2, se2);
}

#[test]
fn se3_default_is_identity() {
    let default_se3 = SE3::default();
    let se3 = SE3::<f64>::from_parts(
        SO3::from_quaternion(multicalc::Quaternion::from_array([1.0, 2.0, 3.0, 4.0])),
        Vector3D::new([1.0, 2.0, 3.0]),
    );

    assert_eq!(default_se3, SE3::identity());
    assert_eq!(se3 * default_se3, se3);
    assert_eq!(default_se3 * se3, se3);
}

// ---- two-direction attitude -------------------------------------------------

#[test]
fn two_direction_pairs_recovers_a_known_rotation() {
    let down_in_world = Vector::new([0.0, 0.0, -1.0]);
    let north_in_world = Vector::new([1.0, 0.0, 0.0]);

    let mut rng = StdRng::seed_from_u64(20260802);
    for _ in 0..200 {
        let rotation_vector = Vector::new([
            rng.gen_range(-2.0..2.0),
            rng.gen_range(-2.0..2.0),
            rng.gen_range(-2.0..2.0),
        ]);
        let truth = SO3::exp(rotation_vector);

        let down_in_body = truth.inverse().act(down_in_world);
        let north_in_body = truth.inverse().act(north_in_world);
        let recovered = SO3::from_two_direction_pairs(
            down_in_body,
            north_in_body,
            down_in_world,
            north_in_world,
        )
        .unwrap();

        assert_entries_close(recovered.to_matrix(), truth.to_matrix(), TOL);
    }
}

#[test]
fn two_direction_pairs_keeps_the_primary_exact_under_noise() {
    let down_in_world = Vector::new([0.0, 0.0, -1.0]);
    let north_in_world = Vector::new([1.0, 0.0, 0.0]);
    let truth = SO3::exp(Vector::new([0.3, -0.2, 0.5]));

    let down_in_body = truth.inverse().act(down_in_world);
    let north_in_body = truth.inverse().act(north_in_world);

    let noise = Vector::new([0.05, 0.05, 0.05]);
    let recovered = SO3::from_two_direction_pairs(
        down_in_body,
        north_in_body + noise,
        down_in_world,
        north_in_world,
    )
    .unwrap();

    // The primary pair is trusted completely, so it comes back with nothing left over.
    let recovered_down = recovered.act(down_in_body);
    assert!((recovered_down - down_in_world).norm() < TOL);

    // The secondary pair absorbs the noise, so it only lands close.
    let recovered_north = recovered.act(north_in_body);
    let north_error = (recovered_north - north_in_world).norm();
    assert!(
        north_error > TOL,
        "the noise should have moved it: {north_error}"
    );
    assert!(north_error < 0.1, "but not by much: {north_error}");
}

#[test]
fn two_direction_pairs_ignores_length() {
    let down_in_world = Vector::new([0.0, 0.0, -1.0]);
    let north_in_world = Vector::new([1.0, 0.0, 0.0]);
    let truth = SO3::exp(Vector::new([0.4, 0.7, -0.3]));
    let down_in_body = truth.inverse().act(down_in_world);
    let north_in_body = truth.inverse().act(north_in_world);

    let unscaled =
        SO3::from_two_direction_pairs(down_in_body, north_in_body, down_in_world, north_in_world)
            .unwrap();

    let mut rng = StdRng::seed_from_u64(20260803);
    for _ in 0..50 {
        let scaled = SO3::from_two_direction_pairs(
            down_in_body * rng.gen_range(0.1..10.0),
            north_in_body * rng.gen_range(0.1..10.0),
            down_in_world * rng.gen_range(0.1..10.0),
            north_in_world * rng.gen_range(0.1..10.0),
        )
        .unwrap();

        assert_entries_close(scaled.to_matrix(), unscaled.to_matrix(), TOL);
    }
}

#[test]
fn two_direction_pairs_rejects_parallel_directions() {
    let down = Vector::new([0.0, 0.0, -1.0]);
    let north = Vector::new([1.0, 0.0, 0.0]);
    let down_again = down * 2.0;
    let up = -down;

    // The two observed directions point the same way, so the spin about them is unsettled.
    assert!(SO3::from_two_direction_pairs(down, down_again, down, north).is_none());
    // Opposite directions leave it just as unsettled.
    assert!(SO3::from_two_direction_pairs(down, up, down, north).is_none());
    // The same holds when it is the reference pair that is parallel.
    assert!(SO3::from_two_direction_pairs(down, north, down, down_again).is_none());
}

#[test]
fn two_direction_pairs_rejects_degenerate_input() {
    let down = Vector::new([0.0, 0.0, -1.0]);
    let north = Vector::new([1.0, 0.0, 0.0]);
    let zero = Vector::new([0.0, 0.0, 0.0]);
    let not_a_number = Vector::new([f64::NAN, 0.0, 0.0]);
    let unbounded = Vector::new([f64::INFINITY, 0.0, 0.0]);

    for degenerate in [zero, not_a_number, unbounded] {
        assert!(SO3::from_two_direction_pairs(degenerate, north, down, north).is_none());
        assert!(SO3::from_two_direction_pairs(down, degenerate, down, north).is_none());
        assert!(SO3::from_two_direction_pairs(down, north, degenerate, north).is_none());
        assert!(SO3::from_two_direction_pairs(down, north, down, degenerate).is_none());
    }
}

#[test]
fn two_direction_pairs_f32_round_trip() {
    const F32_TOL: f32 = 1e-4;
    let down_in_world = Vector::new([0.0_f32, 0.0, -1.0]);
    let north_in_world = Vector::new([1.0_f32, 0.0, 0.0]);

    let rotation_vectors = [
        Vector::new([0.0_f32, 0.0, 0.0]),
        Vector::new([0.3_f32, -0.2, 0.5]),
        Vector::new([1.2_f32, 0.9, -1.5]),
    ];
    for rotation_vector in rotation_vectors {
        let truth = SO3::exp(rotation_vector);
        let down_in_body = truth.inverse().act(down_in_world);
        let north_in_body = truth.inverse().act(north_in_world);

        let recovered = SO3::from_two_direction_pairs(
            down_in_body,
            north_in_body,
            down_in_world,
            north_in_world,
        )
        .unwrap();

        let difference = recovered.to_matrix() - truth.to_matrix();
        for row in 0..3 {
            for column in 0..3 {
                assert!(difference[(row, column)].abs() < F32_TOL);
            }
        }
    }
}

#[test]
fn jacobian_branches_agree_across_small_angle_thresholds() {
    let mut rng = StdRng::seed_from_u64(2);
    let axis = random_unit_vector3(&mut rng);

    let thresholds = [
        (360.0_f64 * f64::EPSILON).powf(1.0 / 6.0),     // so3 c1
        (2520.0_f64 * f64::EPSILON).powf(1.0 / 6.0),    // so3 c2, q c2
        (15_120.0_f64 * f64::EPSILON).powf(1.0 / 6.0),  // inv so3 c3
        (20_160.0_f64 * f64::EPSILON).powf(1.0 / 8.0),  // q c3
        (181_440.0_f64 * f64::EPSILON).powf(1.0 / 8.0), // q c5
    ];

    // Translation part of the twist. `Q` is linear in ρ, so any fixed non-degenerate
    // value works; it only has to be nonzero for the ρ-carrying terms to show up.
    let rho = random_unit_vector3(&mut rng);

    // Straddles each seam. δ·θ ≈ 5e-12 of genuine variation, well under TOL, while a
    // sign-flipped or mispaired series shows up around 1e-4.
    let delta: f64 = 1e-10;
    const TOL: f64 = 1e-9;

    let twist = |theta: f64| {
        let phi = axis.scale(theta);
        Vector::new([rho[0], rho[1], rho[2], phi[0], phi[1], phi[2]])
    };

    for t in thresholds {
        let (lo, hi) = (t * (1.0 - delta), t * (1.0 + delta));

        // c1, c2
        let d = SO3::left_jacobian(axis.scale(hi)) - SO3::left_jacobian(axis.scale(lo));
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    d[(r, c)].abs() < TOL,
                    "SO3::left_jacobian θ={t:e} row={r} col={c} jump={:e}",
                    d[(r, c)]
                );
            }
        }

        // inv c3
        let d =
            SO3::left_jacobian_inverse(axis.scale(hi)) - SO3::left_jacobian_inverse(axis.scale(lo));
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    d[(r, c)].abs() < TOL,
                    "SO3::left_jacobian_inverse θ={t:e} row={r} col={c} jump={:e}",
                    d[(r, c)]
                );
            }
        }

        // q c2, c3, c5 in the top-right block, alongside c1/c2 on the diagonal
        let d = SE3::left_jacobian(twist(hi)) - SE3::left_jacobian(twist(lo));
        for r in 0..6 {
            for c in 0..6 {
                assert!(
                    d[(r, c)].abs() < TOL,
                    "SE3::left_jacobian θ={t:e} row={r} col={c} jump={:e}",
                    d[(r, c)]
                );
            }
        }
    }
}

/// Left Jacobian SO3
/// Threshold for theta_sq 360e^1/3
/// 0.0065633231517825491692449761238662810296049408728717656257768315763825522720280114 +/- 1e-09
/// Computed using mpmath at 80 digits.
#[test]
fn left_jacobian_so3_c1_threshold() {
    let t_hi: Vector<3, f64> = Vector::new([
        0.0018752354719378712,
        -0.002812853207906807,
        0.005625706415813614,
    ]);
    let matrix_hi = Matrix::new([
        [
            0.9999934065615603,
            -0.0028137222355543025,
            -0.0014046633049639315,
        ],
        [
            0.002811963985303727,
            0.9999941391658315,
            -0.0009402517455188686,
        ],
        [
            0.0014081798054650832,
            0.0009349769947671412,
            0.9999980952288952,
        ],
    ]);

    let t_lo: Vector<3, f64> = Vector::new([
        0.0018752349005092999,
        -0.00281285235076395,
        0.0056257047015279,
    ]);
    let matrix_lo = Matrix::new([
        [
            0.9999934065655787,
            -0.002813721377884897,
            -0.0014046628774686768,
        ],
        [
            0.0028119631287058795,
            0.9999941391694033,
            -0.0009402514582003222,
        ],
        [
            0.0014081793758267116,
            0.00093497671066327,
            0.9999980952300561,
        ],
    ]);

    assert_entries_close(SO3::left_jacobian(t_hi), matrix_hi, 1e-13);

    assert_entries_close(SO3::left_jacobian(t_lo), matrix_lo, 1e-13);
}

/// Left Jacobian SO3
/// Threshold for theta_sq 2,025e^1/3
/// 0.0090776505658726734101113986465827468133032549568677515285705797365192293938527642 +/- 1e-09
/// Computed using mpmath at 80 digits.
#[test]
fn left_jacobian_so3_c2_threshold() {
    let t_hi: Vector<3, f64> = Vector::new([
        0.002593614733106478,
        -0.0038904220996597173,
        0.0077808441993194345,
    ]);
    let matrix_hi = Matrix::new([
        [
            0.9999873872318725,
            -0.003892077086700015,
            -0.0019418342873075028,
        ],
        [
            0.0038887136818660114,
            0.9999887886505533,
            -0.0013018435686786764,
        ],
        [
            0.0019485610969755102,
            0.0012917533541766655,
            0.9999963563114298,
        ],
    ]);
    let t_lo: Vector<3, f64> = Vector::new([
        0.002593614161677907,
        -0.00389042124251686,
        0.00778084248503372,
    ]);
    let matrix_lo = Matrix::new([
        [
            0.9999873872374302,
            -0.003892076228833789,
            -0.0019418338602269565,
        ],
        [
            0.0038887128254818386,
            0.9999887886554935,
            -0.0013018432807471972,
        ],
        [
            0.0019485606669308576,
            0.0012917530706913457,
            0.9999963563130354,
        ],
    ]);

    assert_entries_close(SO3::left_jacobian(t_hi), matrix_hi, 1e-13);

    assert_entries_close(SO3::left_jacobian(t_lo), matrix_lo, 1e-13);
}

/// Inverse Left Jacobian SO3
/// Threshold for theta_sq 15,120e^1/3
/// 0.01223672883207982409248517788125746974275923799237944986778430043546338826342771 +/- 1e-09
/// Computed using mpmath at 80 digits.
#[test]
fn inverse_left_jacobian_so3_c3_threshold() {
    let t_hi: Vector<3, f64> = Vector::new([
        0.0034962085234513784,
        -0.005244312785177068,
        0.010488625570354135,
    ]);
    let matrix_hi = Matrix::new([
        [
            0.9999885404644893,
            0.00524278484710897,
            0.002625212268724729,
        ],
        [
            -0.005245840723245165,
            0.9999898137462127,
            0.001743520447521396,
        ],
        [
            -0.0026191005164523384,
            -0.0017526880759299824,
            0.9999966894675191,
        ],
    ]);
    let t_lo: Vector<3, f64> = Vector::new([
        0.003496207952022807,
        -0.00524431192803421,
        0.01048862385606842,
    ]);
    let matrix_lo = Matrix::new([
        [
            0.9999885404682353,
            0.005242783990465573,
            0.002625211839154379,
        ],
        [
            -0.005245839865602847,
            0.9999898137495424,
            0.0017435201633054927,
        ],
        [
            -0.002619100088879831,
            -0.001752687788717314,
            0.9999966894686013,
        ],
    ]);

    assert_entries_close(SO3::left_jacobian_inverse(t_hi), matrix_hi, 1e-13);

    assert_entries_close(SO3::left_jacobian_inverse(t_lo), matrix_lo, 1e-13);
}

/// Matrix Q SE3
/// Threshold for theta_sq 2,025e^1/3
/// 0.0090776505658726734101113986465827468133032549568677515285705797365192293938527642 +/- 1e-09
/// Computed using mpmath at 80 digits.
#[test]
fn q_matrix_se3_c2_threshold() {
    let t_hi: Vector<3, f64> = Vector::new([
        0.002593614733106478,
        -0.0038904220996597173,
        0.0077808441993194345,
    ]);
    let matrix_hi = Matrix::new([
        [
            0.0022693965896371064,
            0.3747777166794713,
            0.1253215571831013,
        ],
        [
            -0.37520998587434706,
            0.0015129295149932092,
            -0.2491889759799066,
        ],
        [
            -0.12467315246392878,
            0.25080997294809526,
            -0.00010806884348379557,
        ],
    ]);
    let t_lo: Vector<3, f64> = Vector::new([
        0.002593614161677907,
        -0.00389042124251686,
        0.00778084248503372,
    ]);
    let matrix_lo = Matrix::new([
        [
            0.002269396089647881,
            0.37477771672979976,
            0.12532155711283796,
        ],
        [
            -0.3752099858294374,
            0.0015129291816680802,
            -0.24918897615870556,
        ],
        [
            -0.12467315253652327,
            0.2508099727697594,
            -0.00010806881967323877,
        ],
    ]);

    assert_entries_close(q_block(RHO, t_hi), matrix_hi, 1e-13);
    assert_entries_close(q_block(RHO, t_lo), matrix_lo, 1e-13);
}

/// Matrix Q SE3
/// Threshold for theta_sq 20,160e^1/4
/// 0.038138740273514832356063423923705899849787335895792428065353256457527059271825147 +/- 1e-09
/// Computed using mpmath at 80 digits.
#[test]
fn q_matrix_se3_c3_threshold() {
    let t_hi: Vector<3, f64> = Vector::new([
        0.010896783221004238,
        -0.016345174831506357,
        0.032690349663012715,
    ]);
    let matrix_hi = Matrix::new([
        [
            0.009533476406376695,
            0.37398340468332747,
            0.1263154481158103,
        ],
        [
            -0.37579954060639215,
            0.0063555363829814185,
            -0.24658593196723697,
        ],
        [
            -0.12359117549845144,
            0.25339551378644487,
            -0.0004541485353692165,
        ],
    ]);
    let t_lo: Vector<3, f64> =
        Vector::new([0.010896782649575667, -0.0163451739743635, 0.032690347948727]);
    let matrix_lo = Matrix::new([
        [
            0.009533475906566877,
            0.37398340474232844,
            0.12631544804927197,
        ],
        [
            -0.3757995405701542,
            0.006355536049792893,
            -0.24658593214671007,
        ],
        [
            -0.12359117557478233,
            0.2533955136089179,
            -0.00045414851154146005,
        ],
    ]);

    assert_entries_close(q_block(RHO, t_hi), matrix_hi, 1e-13);
    assert_entries_close(q_block(RHO, t_lo), matrix_lo, 1e-13);
}

#[test]
/// Matrix Q SE3
/// Threshold for theta_sq 181,440e^1/4
/// 0.050193404960717505336016995289832652425413397500753684377067048913466264320148131 +/- 1e-09
/// Computed using mpmath at 80 digits.
fn q_matrix_se3_c5_threshold() {
    let t_hi: Vector<3, f64> = Vector::new([
        0.014340973131633574,
        -0.021511459697450358,
        0.043022919394900716,
    ]);
    let matrix_hi = Matrix::new([
        [0.012545595866446456, 0.3736169484295461, 0.1267118476660283],
        [
            -0.37600712288784666,
            0.008363469462850288,
            -0.24550341123485972,
        ],
        [
            -0.12312642930970909,
            0.2544644504237632,
            -0.0005978047293558165,
        ],
    ]);
    let t_lo: Vector<3, f64> =
        Vector::new([0.014340972560205001, -0.0215114588403075, 0.043022917680615]);
    let matrix_lo = Matrix::new([
        [
            0.012545595366775837,
            0.3736169484921431,
            0.12671184760103235,
        ],
        [
            -0.37600712285520405,
            0.008363469129767754,
            -0.2455034114145882,
        ],
        [
            -0.1231264293875912,
            0.25446445024659614,
            -0.0005978047055147155,
        ],
    ]);

    assert_entries_close(q_block(RHO, t_hi), matrix_hi, 1e-13);
    assert_entries_close(q_block(RHO, t_lo), matrix_lo, 1e-13);
}
