//! Rigid-body inertia tests: constructor checks, moving the reference point, combining bodies,
//! the momentum and energy a motion carries, and scalar-generic behaviour at f32 and under autodiff.

use multicalc::error::SpatialError;
use multicalc::linear_algebra::{Matrix, Matrix3D, Vector, Vector3D};
use multicalc::scalar::Dual;
use multicalc::spatial::{SE3, SpatialInertia, Twist};

fn unit_inertia() -> Matrix3D {
    Matrix::from_diagonal([1.0, 1.0, 1.0])
}

fn origin() -> Vector3D {
    Vector::new([0.0, 0.0, 0.0])
}

/// A body that does not balance on its own origin and resists spinning differently about each axis.
fn offset_body() -> SpatialInertia<f64> {
    SpatialInertia::new(
        2.5,
        Vector::new([0.1, -0.2, 0.3]),
        Matrix::new([
            [0.05, 0.01, -0.02],
            [0.01, 0.07, 0.003],
            [-0.02, 0.003, 0.09],
        ]),
    )
    .unwrap()
}

fn second_body() -> SpatialInertia<f64> {
    SpatialInertia::from_diagonal_inertia(
        1.25,
        Vector::new([-0.3, 0.15, 0.05]),
        Vector::new([0.02, 0.03, 0.04]),
    )
    .unwrap()
}

fn sample_velocity() -> Twist<f64> {
    Twist::from_array([0.4, -1.1, 0.7, 1.3, 0.6, -0.8])
}

fn sample_pose() -> SE3<f64> {
    SE3::exp(Vector::new([0.3, -0.4, 0.5, 0.2, 0.7, -0.1]))
}

#[test]
fn rejects_non_finite_values() {
    let mass = SpatialInertia::new(f64::INFINITY, origin(), unit_inertia());
    assert_eq!(mass, Err(SpatialError::NonFinite));

    let center = SpatialInertia::new(1.0, Vector::new([f64::NAN, 0.0, 0.0]), unit_inertia());
    assert_eq!(center, Err(SpatialError::NonFinite));

    let inertia = SpatialInertia::new(1.0, origin(), Matrix::from_diagonal([1.0, f64::NAN, 1.0]));
    assert_eq!(inertia, Err(SpatialError::NonFinite));
}

#[test]
fn rejects_non_positive_mass() {
    assert_eq!(
        SpatialInertia::new(0.0, origin(), unit_inertia()),
        Err(SpatialError::NonPositiveMass)
    );
    assert_eq!(
        SpatialInertia::new(-1.0, origin(), unit_inertia()),
        Err(SpatialError::NonPositiveMass)
    );
}

#[test]
fn rejects_inertia_that_differs_across_the_diagonal() {
    let lopsided = Matrix::new([[1.0, 0.5, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]);
    assert_eq!(
        SpatialInertia::new(1.0, origin(), lopsided),
        Err(SpatialError::NotSymmetric)
    );
}

#[test]
fn accepts_inertia_off_by_one_rounding_step() {
    // The symmetry check is scaled by the entries, not an exact comparison, so a pair that differs
    // by a single rounding step is still the same number as far as this type is concerned.
    let nudged = Matrix::new([
        [1.0, 0.25, 0.0],
        [0.25 + f64::EPSILON, 2.0, 0.0],
        [0.0, 0.0, 3.0],
    ]);
    assert!(SpatialInertia::new(1.0, origin(), nudged).is_ok());
}

#[test]
fn rejects_non_positive_inertia_diagonal() {
    assert_eq!(
        SpatialInertia::new(1.0, origin(), Matrix::from_diagonal([1.0, 0.0, 1.0])),
        Err(SpatialError::NonPositiveInertia)
    );
    assert_eq!(
        SpatialInertia::new(1.0, origin(), Matrix::from_diagonal([1.0, 1.0, -2.0])),
        Err(SpatialError::NonPositiveInertia)
    );
}

#[test]
fn diagonal_constructor_matches_the_full_one() {
    let from_diagonal =
        SpatialInertia::from_diagonal_inertia(2.0, origin(), Vector::new([1.0, 2.0, 3.0])).unwrap();
    let from_full =
        SpatialInertia::new(2.0, origin(), Matrix::from_diagonal([1.0, 2.0, 3.0])).unwrap();
    assert_eq!(from_diagonal, from_full);
}

#[test]
fn inertia_about_the_balance_point_is_unchanged() {
    let center = Vector::new([0.3, -0.2, 0.5]);
    let body = SpatialInertia::new(2.0, center, unit_inertia()).unwrap();
    assert_eq!(body.inertia_about(center), unit_inertia());
}

#[test]
fn inertia_about_grows_with_distance() {
    let body = SpatialInertia::new(1.0, origin(), unit_inertia()).unwrap();
    // A metre along x leaves spinning about x alone and doubles the other two.
    let shifted = body.inertia_about(Vector::new([1.0, 0.0, 0.0]));
    assert_eq!(shifted.diagonal(), [1.0, 2.0, 2.0]);
    assert_eq!(shifted[(0, 1)], 0.0);
    assert_eq!(shifted[(0, 2)], 0.0);
    assert_eq!(shifted[(1, 2)], 0.0);
}

#[test]
fn inertia_about_couples_axes_for_a_diagonal_offset() {
    let body = SpatialInertia::new(2.0, origin(), unit_inertia()).unwrap();
    let offset = Vector::new([1.0, 2.0, 0.0]);
    let shifted = body.inertia_about(offset);
    // Each entry picks up the mass times the offset spread: the diagonal by how far the point is
    // overall, the off-diagonal by the product of the two directions.
    assert!((shifted[(0, 0)] - (1.0 + 2.0 * 4.0)).abs() < 1e-12);
    assert!((shifted[(1, 1)] - (1.0 + 2.0 * 1.0)).abs() < 1e-12);
    assert!((shifted[(2, 2)] - (1.0 + 2.0 * 5.0)).abs() < 1e-12);
    assert!((shifted[(0, 1)] - (-2.0 * 2.0)).abs() < 1e-12);
    assert_eq!(shifted[(0, 1)], shifted[(1, 0)]);
}

#[test]
fn reports_whether_every_number_is_finite() {
    let body = SpatialInertia::new(1.0, origin(), unit_inertia()).unwrap();
    assert!(body.is_finite());
}

#[test]
fn works_at_single_precision() {
    let body = SpatialInertia::new(
        2.0_f32,
        Vector::new([0.0_f32, 0.0, 0.0]),
        Matrix::from_diagonal([1.0_f32, 1.0, 1.0]),
    )
    .unwrap();

    // Mass 2 at an offset of (1, 2, 0), which is 5 away squared: each diagonal entry picks up the
    // mass times what is left after taking out its own direction, and the xy pair picks up the
    // product of the two.
    let shifted = body.inertia_about(Vector::new([1.0_f32, 2.0, 0.0]));
    assert!((shifted[(0, 0)] - 9.0).abs() < 1e-5);
    assert!((shifted[(1, 1)] - 3.0).abs() < 1e-5);
    assert!((shifted[(2, 2)] - 11.0).abs() < 1e-5);
    assert!((shifted[(0, 1)] + 4.0).abs() < 1e-5);
    assert_eq!(shifted[(0, 1)], shifted[(1, 0)]);
}

#[test]
fn differentiates_through_the_mass() {
    // Seed the mass with a unit derivative and read it back out of the shifted inertia. Moving the
    // reference point a metre along x adds the mass itself to the y entry, so the derivative there
    // is exactly one.
    let mass = Dual::variable(2.0_f64);
    let center = Vector::new([Dual::constant(0.0); 3]);
    let inertia = Matrix::from_diagonal([Dual::constant(1.0); 3]);
    let body = SpatialInertia::new(mass, center, inertia).unwrap();

    let offset = Vector::new([
        Dual::constant(1.0),
        Dual::constant(0.0),
        Dual::constant(0.0),
    ]);
    let shifted = body.inertia_about(offset);

    assert!((shifted[(1, 1)].value - 3.0).abs() < 1e-12);
    assert!((shifted[(1, 1)].deriv - 1.0).abs() < 1e-12);
    // Spinning about the direction the point moved in does not change, so nothing depends on the
    // mass there.
    assert!((shifted[(0, 0)].deriv - 0.0).abs() < 1e-12);
}

#[test]
fn momentum_matches_the_matrix_form() {
    let body = offset_body();
    let velocity = sample_velocity();
    let got = body.momentum(velocity).to_vector();
    let want = body.to_matrix() * velocity.to_vector();
    for component in 0..6 {
        assert!((got[component] - want[component]).abs() < 1e-12);
    }
}

#[test]
fn inertia_matrix_reads_the_same_across_the_diagonal() {
    let block = offset_body().to_matrix();
    for row in 0..6 {
        for col in 0..6 {
            assert!((block[(row, col)] - block[(col, row)]).abs() < 1e-12);
        }
    }
}

#[test]
fn kinetic_energy_is_half_the_power_product() {
    let body = offset_body();
    let velocity = sample_velocity();
    let paired = 0.5 * velocity.dot_wrench(body.momentum(velocity));
    assert!((body.kinetic_energy(velocity) - paired).abs() < 1e-12);
    assert!(body.kinetic_energy(velocity) > 0.0);
    assert_eq!(body.kinetic_energy(Twist::zeros()), 0.0);
}

#[test]
fn bias_wrench_is_the_velocity_carried_through_the_momentum() {
    let body = offset_body();
    let velocity = sample_velocity();
    assert_eq!(
        body.bias_wrench(velocity).as_array(),
        velocity.cross_wrench(body.momentum(velocity)).as_array()
    );
}

#[test]
fn momentum_moves_with_the_frame() {
    let body = offset_body();
    let velocity = sample_velocity();
    let pose = sample_pose();

    let carried = pose.act_wrench(body.momentum(velocity));
    let restated = pose.act_inertia(body).momentum(pose.act_twist(velocity));
    assert!((carried - restated).to_vector().norm() < 1e-11);
}

#[test]
fn kinetic_energy_does_not_care_which_frame_it_is_read_in() {
    let body = offset_body();
    let velocity = sample_velocity();
    let pose = sample_pose();

    let here = body.kinetic_energy(velocity);
    let there = pose
        .act_inertia(body)
        .kinetic_energy(pose.act_twist(velocity));
    assert!((here - there).abs() < 1e-11);
}

#[test]
fn inertia_action_round_trips() {
    let body = offset_body();
    let pose = sample_pose();
    let round_trip = pose.inverse_act_inertia(pose.act_inertia(body));

    assert!((round_trip.mass() - body.mass()).abs() < 1e-12);
    assert!((round_trip.center_of_mass() - body.center_of_mass()).norm() < 1e-12);
    for row in 0..3 {
        for col in 0..3 {
            let got = round_trip.rotational_inertia()[(row, col)];
            assert!((got - body.rotational_inertia()[(row, col)]).abs() < 1e-12);
        }
    }
}

#[test]
fn combining_adds_the_masses_and_balances_between_them() {
    let left = SpatialInertia::new(1.0, Vector::new([-1.0, 0.0, 0.0]), unit_inertia()).unwrap();
    let right = SpatialInertia::new(1.0, Vector::new([1.0, 0.0, 0.0]), unit_inertia()).unwrap();
    let whole = left.combined(right);

    assert_eq!(whole.mass(), 2.0);
    assert!(whole.center_of_mass().norm() < 1e-12);
    // Each unit inertia picks up 1·1² about y and z from the metre offset, and nothing about x.
    let diagonal = whole.rotational_inertia().diagonal();
    assert!((diagonal[0] - 2.0).abs() < 1e-12);
    assert!((diagonal[1] - 4.0).abs() < 1e-12);
    assert!((diagonal[2] - 4.0).abs() < 1e-12);
}

#[test]
fn combining_adds_the_matrix_forms() {
    let got = offset_body().combined(second_body()).to_matrix();
    let want = offset_body().to_matrix() + second_body().to_matrix();
    for row in 0..6 {
        for col in 0..6 {
            assert!((got[(row, col)] - want[(row, col)]).abs() < 1e-12);
        }
    }
}

#[test]
fn combining_does_not_depend_on_the_order() {
    let forwards = offset_body().combined(second_body());
    let backwards = second_body().combined(offset_body());

    assert!((forwards.mass() - backwards.mass()).abs() < 1e-12);
    assert!((forwards.center_of_mass() - backwards.center_of_mass()).norm() < 1e-12);
    for row in 0..3 {
        for col in 0..3 {
            let got = forwards.rotational_inertia()[(row, col)];
            assert!((got - backwards.rotational_inertia()[(row, col)]).abs() < 1e-12);
        }
    }
}

#[test]
fn inertia_algebra_is_transparent_to_dual_scalars() {
    // Kinetic energy is ½·m·|v + ω×c|² + ½·ωᵀ·I_c·ω, so seeding the mass with a unit derivative
    // leaves the first half's coefficient behind.
    let center = Vector::new([0.1, -0.2, 0.3]);
    let rotational_inertia = offset_body().rotational_inertia();
    let body = SpatialInertia::new(
        Dual::variable(2.5_f64),
        center.map(Dual::constant),
        Matrix::from_fn(|row, col| Dual::constant(rotational_inertia[(row, col)])),
    )
    .unwrap();
    let velocity = sample_velocity();
    let seeded = Twist::new(
        velocity.linear().map(Dual::constant),
        velocity.angular().map(Dual::constant),
    );

    let balance_point_velocity = velocity.linear() + velocity.angular().cross(center);
    let analytic = 0.5 * balance_point_velocity.dot(balance_point_velocity);

    let energy = body.kinetic_energy(seeded);
    assert!((energy.value - offset_body().kinetic_energy(velocity)).abs() < 1e-12);
    assert!((energy.deriv - analytic).abs() < 1e-12);
}

#[test]
fn inertia_algebra_works_in_single_precision() {
    let body = SpatialInertia::new(
        2.5_f32,
        Vector::new([0.1_f32, -0.2, 0.3]),
        Matrix::new([
            [0.05_f32, 0.01, -0.02],
            [0.01, 0.07, 0.003],
            [-0.02, 0.003, 0.09],
        ]),
    )
    .unwrap();
    let velocity = Twist::from_array([0.4_f32, -1.1, 0.7, 1.3, 0.6, -0.8]);

    let got = body.momentum(velocity).to_vector();
    let want = body.to_matrix() * velocity.to_vector();
    for component in 0..6 {
        assert!((got[component] - want[component]).abs() < 1e-5);
    }
}
