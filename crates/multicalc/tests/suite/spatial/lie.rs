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

fn random_vector3(rng: &mut StdRng) -> Vector<3, f64> {
    Vector::new([
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
    ])
}

fn random_unit_vector3(rng: &mut StdRng) -> Vector<3, f64> {
    loop {
        let vector = random_vector3(rng);
        let norm = vector.dot(vector).sqrt();
        if norm > 1e-3 {
            return vector * norm.recip();
        }
    }
}

fn random_twist6(rng: &mut StdRng) -> Vector<6, f64> {
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
