use multicalc::Dual;
use multicalc::error::IntegrateError;
use multicalc::linear_algebra::{Vector, Vector3D};
use multicalc::ode::ExponentialMap;
use multicalc::spatial::SO3;

// ω(t) = [0.8·cos(1.3 t), 0.5·sin(0.7 t), 1.1] rad/s, about the body's own axes.
fn prescribed_rate(time: f64) -> Vector3D<f64> {
    Vector::new([0.8 * (1.3 * time).cos(), 0.5 * (0.7 * time).sin(), 1.1])
}

// The same rate differentiated: ω̇(t) = [-1.04·sin(1.3 t), 0.35·cos(0.7 t), 0].
fn prescribed_rate_change(time: f64) -> Vector3D<f64> {
    Vector::new([-1.04 * (1.3 * time).sin(), 0.35 * (0.7 * time).cos(), 0.0])
}

// Where the prescribed rate leaves an orientation after `steps` steps of the second-order loop.
fn reference_orientation(steps: usize, final_time: f64) -> SO3<f64> {
    let rate = |time: f64, _orientation: SO3<f64>| prescribed_rate(time);
    ExponentialMap::integrate_attitude(
        &rate,
        0.0,
        SO3::identity(),
        final_time / steps as f64,
        steps,
        |_, _| {},
    )
    .unwrap()
}

// How far apart two orientations are, in radians.
fn angle_between(a: SO3<f64>, b: SO3<f64>) -> f64 {
    (a.inverse() * b).log().norm()
}

#[test]
fn steady_rate_matches_the_closed_form_turn() {
    // A steady 2 rad/s for 0.35 s is a 0.7 rad turn — half that would mean a stray half somewhere.
    let turning_about_z = Vector::new([0.0, 0.0, 2.0]);
    let timestep = 0.35;
    let stepped = ExponentialMap::attitude_step(SO3::identity(), turning_about_z, timestep);
    let expected = SO3::exp(Vector::new([0.0, 0.0, 0.7]));
    assert!(angle_between(stepped, expected) < 1e-15);
}

#[test]
fn zero_rate_leaves_the_orientation_alone() {
    // A zero turn rate runs through the series the exponential uses near no rotation at all.
    let start = SO3::exp(Vector::new([0.3, -0.2, 0.5]));
    let not_turning = Vector::new([0.0, 0.0, 0.0]);
    let stepped = ExponentialMap::attitude_step(start, not_turning, 0.01);
    assert!(angle_between(start, stepped) < 1e-15);
}

#[test]
fn half_turn_step_lands_on_the_half_turn() {
    // π rad/s for one second is a half turn, the far end of what a single step can name.
    let turning_about_z = Vector::new([0.0, 0.0, core::f64::consts::PI]);
    let stepped = ExponentialMap::attitude_step(SO3::identity(), turning_about_z, 1.0);
    assert!((stepped.log().norm() - core::f64::consts::PI).abs() < 1e-12);
}

#[test]
fn unit_length_holds_over_a_long_run() {
    // Twenty seconds of tumbling at a tenth of a millisecond a step: the length has to stay put.
    let rate = |time: f64, _orientation: SO3<f64>| prescribed_rate(time);
    let facing =
        ExponentialMap::integrate_attitude(&rate, 0.0, SO3::identity(), 1e-4, 200_000, |_, _| {})
            .unwrap();
    assert!((facing.quaternion().norm() - 1.0).abs() < 1e-12);
}

#[test]
fn first_order_step_converges_first_order() {
    // Halving the step should roughly halve the endpoint error.
    let reference = reference_orientation(200_000, 1.0);
    let endpoint_error = |steps: usize| {
        let timestep = 1.0 / steps as f64;
        let mut orientation = SO3::<f64>::identity();
        for step in 0..steps {
            let time = step as f64 * timestep;
            orientation =
                ExponentialMap::attitude_step(orientation, prescribed_rate(time), timestep);
        }
        angle_between(orientation, reference)
    };
    let ratio = endpoint_error(200) / endpoint_error(400);
    assert!((1.6..=2.4).contains(&ratio), "convergence ratio {ratio}");
}

#[test]
fn midpoint_step_converges_second_order() {
    // Halving the step should cut the endpoint error by about four.
    let reference = reference_orientation(200_000, 1.0);
    let endpoint_error = |steps: usize| {
        let timestep = 1.0 / steps as f64;
        let mut orientation = SO3::<f64>::identity();
        for step in 0..steps {
            let time = step as f64 * timestep;
            orientation = ExponentialMap::attitude_step_with_angular_acceleration(
                orientation,
                prescribed_rate(time),
                prescribed_rate_change(time),
                timestep,
            );
        }
        angle_between(orientation, reference)
    };
    let ratio = endpoint_error(200) / endpoint_error(400);
    assert!((3.2..=4.8).contains(&ratio), "convergence ratio {ratio}");
}

#[test]
fn integrate_attitude_converges_second_order() {
    // The loop works the half-way rate out for itself, and lands at the same order.
    let reference = reference_orientation(200_000, 1.0);
    let endpoint_error = |steps: usize| {
        let rate = |time: f64, _orientation: SO3<f64>| prescribed_rate(time);
        let facing = ExponentialMap::integrate_attitude(
            &rate,
            0.0,
            SO3::identity(),
            1.0 / steps as f64,
            steps,
            |_, _| {},
        )
        .unwrap();
        angle_between(facing, reference)
    };
    let ratio = endpoint_error(200) / endpoint_error(400);
    assert!((3.2..=4.8).contains(&ratio), "convergence ratio {ratio}");
}

#[test]
fn observer_sees_every_node_starting_with_the_first() {
    let steady = |_time: f64, _orientation: SO3<f64>| Vector::new([0.0, 0.0, 1.0]);
    let steps = 10;
    let timestep = 0.1;

    let mut nodes: Vec<(f64, f64)> = Vec::new();
    let _facing = ExponentialMap::integrate_attitude(
        &steady,
        0.0,
        SO3::identity(),
        timestep,
        steps,
        |time, orientation| nodes.push((time, orientation.log()[2])),
    )
    .unwrap();

    assert_eq!(nodes.len(), steps + 1);
    assert_eq!(nodes[0].0, 0.0);
    assert!(nodes[0].1.abs() < 1e-15);
    assert!((nodes[steps].0 - 1.0).abs() < 1e-12);
    assert!((nodes[steps].1 - 1.0).abs() < 1e-12);
}

#[test]
fn differentiates_through_the_small_angle_branch() {
    // Turning about z by ω_z·dt, so how the turn responds to ω_z is just dt — including at the
    // zero rate, where the exponential switches to its series.
    let timestep = 1e-3;
    let turn_from_rate = |rate_z: f64| {
        let angular_rate = Vector::new([
            Dual::constant(0.0),
            Dual::constant(0.0),
            Dual::variable(rate_z),
        ]);
        let stepped = ExponentialMap::attitude_step(
            SO3::<Dual<f64>>::identity(),
            angular_rate,
            Dual::constant(timestep),
        );
        stepped.log()[2].deriv
    };

    // ‖ω‖·dt below, at, and above the series threshold.
    for rate_z in [0.0, 1e-5, 10.0] {
        let derivative = turn_from_rate(rate_z);
        assert!(derivative.is_finite(), "derivative at ω_z = {rate_z}");
        assert!(
            (derivative - timestep).abs() < 1e-6,
            "derivative {derivative} at ω_z = {rate_z}"
        );
    }
}

#[test]
fn f32_holds_unit_length_and_round_trips() {
    // The same long run at f32, where the length has much further to drift: the smaller scalar
    // leaves about a hundredth of its own rounding step behind on each composition, so ten seconds
    // of stepping lands a couple of ten-thousandths off unit length rather than at rounding.
    let rate = |time: f32, _orientation: SO3<f32>| {
        Vector::new([0.8 * (1.3 * time).cos(), 0.5 * (0.7 * time).sin(), 1.1])
    };
    let facing = ExponentialMap::integrate_attitude(
        &rate,
        0.0_f32,
        SO3::identity(),
        1e-4_f32,
        100_000,
        |_, _| {},
    )
    .unwrap();
    assert!((facing.quaternion().norm() - 1.0).abs() < 1e-3);

    let turn = Vector::new([0.2_f32, -0.1, 0.4]);
    let round_tripped = SO3::<f32>::exp(turn).log();
    assert!((round_tripped - turn).norm() < 1e-5);
}

#[test]
fn integrate_attitude_rejects_non_positive_timestep() {
    let rate = |_time: f64, _orientation: SO3<f64>| Vector::new([0.0, 0.0, 1.0]);
    assert_eq!(
        ExponentialMap::integrate_attitude(&rate, 0.0, SO3::identity(), 0.0, 10, |_, _| {}),
        Err(IntegrateError::NonPositiveTimestep)
    );
    assert_eq!(
        ExponentialMap::integrate_attitude(&rate, 0.0, SO3::identity(), -0.1, 10, |_, _| {}),
        Err(IntegrateError::NonPositiveTimestep)
    );
}

#[test]
fn integrate_attitude_rejects_non_finite_rate() {
    let rate = |_time: f64, _orientation: SO3<f64>| Vector::new([f64::NAN, 0.0, 0.0]);
    assert_eq!(
        ExponentialMap::integrate_attitude(&rate, 0.0, SO3::identity(), 0.01, 10, |_, _| {}),
        Err(IntegrateError::NonFinite)
    );
}
