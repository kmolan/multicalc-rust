//! Spatial-algebra tests: the cross products against their matrix forms, the Plücker actions
//! against the adjoint matrices, and the invariants a change of frame has to leave alone.

use multicalc::linear_algebra::{Matrix, Matrix3D, Matrix6D, Vector, Vector3D};
use multicalc::scalar::Dual;
use multicalc::spatial::{SE3, Twist, Wrench};

/// The 3×3 skew matrix, written out here because the crate's own is not public.
fn skew(vector: Vector3D<f64>) -> Matrix3D<f64> {
    Matrix::new([
        [0.0, -vector[2], vector[1]],
        [vector[2], 0.0, -vector[0]],
        [-vector[1], vector[0], 0.0],
    ])
}

/// The 6×6 form of `Twist::cross`, in `[v; ω]` blocks: `[[[ω]×, [v]×], [0, [ω]×]]`.
fn motion_cross_matrix(twist: Twist<f64>) -> Matrix6D<f64> {
    let angular = skew(twist.angular());
    let linear = skew(twist.linear());
    let mut entries = Matrix::zeros();
    for i in 0..3 {
        for j in 0..3 {
            entries[(i, j)] = angular[(i, j)];
            entries[(i, j + 3)] = linear[(i, j)];
            entries[(i + 3, j + 3)] = angular[(i, j)];
        }
    }
    entries
}

/// Four poses with rotation and translation both non-trivial.
fn sample_poses() -> [SE3<f64>; 4] {
    [
        SE3::exp(Vector::new([0.4, -0.2, 0.7, 0.3, 0.9, -0.5])),
        SE3::exp(Vector::new([-1.1, 0.6, 0.2, -0.8, 0.4, 1.2])),
        SE3::exp(Vector::new([0.05, 0.9, -1.3, 1.5, -0.7, 0.25])),
        SE3::exp(Vector::new([2.0, -1.4, 0.3, -0.6, -1.1, 0.85])),
    ]
}

fn first_twist() -> Twist<f64> {
    Twist::from_array([0.3, -0.7, 1.1, 0.5, 0.2, -0.9])
}

fn second_twist() -> Twist<f64> {
    Twist::from_array([-1.2, 0.4, 0.6, -0.3, 0.8, 0.1])
}

fn third_twist() -> Twist<f64> {
    Twist::from_array([0.9, 0.1, -0.4, 1.4, -0.6, 0.3])
}

fn sample_wrench() -> Wrench<f64> {
    Wrench::from_array([2.0, -1.5, 0.7, 0.9, -0.2, 1.3])
}

#[test]
fn motion_cross_matches_its_matrix_form() {
    let got = first_twist().cross(second_twist()).to_vector();
    let want = motion_cross_matrix(first_twist()) * second_twist().to_vector();
    for component in 0..6 {
        assert!((got[component] - want[component]).abs() < 1e-12);
    }
}

#[test]
fn force_cross_is_the_negative_transpose_of_the_motion_cross() {
    let got = first_twist().cross_wrench(sample_wrench()).to_vector();
    let want = -(motion_cross_matrix(first_twist()).transpose()) * sample_wrench().to_vector();
    for component in 0..6 {
        assert!((got[component] - want[component]).abs() < 1e-12);
    }
}

#[test]
fn motion_cross_flips_sign_and_kills_itself() {
    let (a, b) = (first_twist(), second_twist());
    assert!((a.cross(b) + b.cross(a)).to_vector().norm() < 1e-12);
    assert!(a.cross(a).to_vector().norm() < 1e-12);
}

#[test]
fn motion_cross_satisfies_the_jacobi_identity() {
    let (first, second, third) = (first_twist(), second_twist(), third_twist());
    let sum = first.cross(second.cross(third))
        + second.cross(third.cross(first))
        + third.cross(first.cross(second));
    assert!(sum.to_vector().norm() < 1e-11);
}

#[test]
fn motion_action_matches_the_adjoint_matrix() {
    let twist = first_twist();
    for pose in sample_poses() {
        let got = pose.act_twist(twist).to_vector();
        let want = pose.adjoint() * twist.to_vector();
        for component in 0..6 {
            assert!((got[component] - want[component]).abs() < 1e-12);
        }
    }
}

#[test]
fn force_action_matches_the_force_adjoint_matrix() {
    let wrench = sample_wrench();
    for pose in sample_poses() {
        let got = pose.act_wrench(wrench).to_vector();
        let want = pose.force_adjoint() * wrench.to_vector();
        for component in 0..6 {
            assert!((got[component] - want[component]).abs() < 1e-12);
        }
    }
}

#[test]
fn force_adjoint_is_the_inverse_transpose_of_the_adjoint() {
    for pose in sample_poses() {
        let product = pose.adjoint().transpose() * pose.force_adjoint();
        for row in 0..6 {
            for col in 0..6 {
                let want = if row == col { 1.0 } else { 0.0 };
                assert!((product[(row, col)] - want).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn actions_round_trip() {
    let twist = first_twist();
    let wrench = sample_wrench();
    for pose in sample_poses() {
        let motion = pose.inverse_act_twist(pose.act_twist(twist));
        assert!((motion - twist).to_vector().norm() < 1e-12);

        let force = pose.inverse_act_wrench(pose.act_wrench(wrench));
        assert!((force - wrench).to_vector().norm() < 1e-12);
    }
}

#[test]
fn actions_compose() {
    let twist = first_twist();
    let wrench = sample_wrench();
    for first in sample_poses() {
        for second in sample_poses() {
            let composed = first * second;

            let motion = composed.act_twist(twist);
            let stepwise = first.act_twist(second.act_twist(twist));
            assert!((motion - stepwise).to_vector().norm() < 1e-11);

            let force = composed.act_wrench(wrench);
            let stepwise = first.act_wrench(second.act_wrench(wrench));
            assert!((force - stepwise).to_vector().norm() < 1e-11);
        }
    }
}

#[test]
fn power_survives_a_change_of_frame() {
    let twist = first_twist();
    let wrench = sample_wrench();
    let here = twist.dot_wrench(wrench);
    for pose in sample_poses() {
        let there = pose.act_twist(twist).dot_wrench(pose.act_wrench(wrench));
        assert!((here - there).abs() < 1e-11);
    }
}

#[test]
fn cross_products_survive_a_change_of_frame() {
    let (a, b) = (first_twist(), second_twist());
    let wrench = sample_wrench();
    for pose in sample_poses() {
        let carried = pose.act_twist(a.cross(b));
        let crossed = pose.act_twist(a).cross(pose.act_twist(b));
        assert!((carried - crossed).to_vector().norm() < 1e-11);

        let carried = pose.act_wrench(a.cross_wrench(wrench));
        let crossed = pose.act_twist(a).cross_wrench(pose.act_wrench(wrench));
        assert!((carried - crossed).to_vector().norm() < 1e-11);
    }
}

#[test]
fn algebra_is_transparent_to_dual_scalars() {
    // Seed ω_z with a unit derivative. The first linear component of a × b is
    // (ω_a × v_b)_x + (v_a × ω_b)_x, and only the -ω_z·v_by term depends on it.
    let spin_z = 0.5;
    let seeded = Twist::new(
        Vector::new([
            Dual::constant(0.3),
            Dual::constant(-0.7),
            Dual::constant(1.1),
        ]),
        Vector::new([
            Dual::constant(0.5),
            Dual::constant(0.2),
            Dual::variable(spin_z),
        ]),
    );
    let other = Twist::from_array([
        Dual::constant(-1.2),
        Dual::constant(0.4),
        Dual::constant(0.6),
        Dual::constant(-0.3),
        Dual::constant(0.8),
        Dual::constant(0.1),
    ]);

    let crossed = seeded.cross(other);
    let linear_x = crossed.linear()[0];

    let plain = Twist::from_array([0.3, -0.7, 1.1, 0.5, 0.2, spin_z]);
    let plain_crossed = plain.cross(second_twist());
    assert!((linear_x.value - plain_crossed.linear()[0]).abs() < 1e-12);

    // ∂/∂ω_z of (ω_ay·v_bz − ω_az·v_by) is −v_by.
    assert!((linear_x.deriv - -0.4).abs() < 1e-12);
}

#[test]
fn algebra_works_in_single_precision() {
    let twist = Twist::from_array([0.3_f32, -0.7, 1.1, 0.5, 0.2, -0.9]);
    let pose = SE3::exp(Vector::new([0.4_f32, -0.2, 0.7, 0.3, 0.9, -0.5]));

    let got = pose.act_twist(twist).to_vector();
    let want = pose.adjoint() * twist.to_vector();
    for component in 0..6 {
        assert!((got[component] - want[component]).abs() < 1e-5);
    }
}
