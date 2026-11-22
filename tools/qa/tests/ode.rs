#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks RK45 dense-output samples and exp-map attitude steps against scipy solve_ivp goldens.

use multicalc::dynamics::{RigidBody, free_joint_from_state_vector};
use multicalc::linear_algebra::{Vector, Vector3D};
use multicalc::ode::{ExponentialMap, Rk45};
use multicalc::spatial::{FreeJointState, Quaternion, SO3, SpatialInertia, Wrench};
use multicalc_qa::load::*;
use multicalc_qa::problems::{ode_exp_decay, ode_harmonic, ode_two_body, ode_van_der_pol_mild};
use multicalc_qa::schema::Fixture;

fn run_case<const N: usize>(fixture: &Fixture, f: &dyn Fn(f64, &Vector<N>) -> Vector<N>) {
    let problem = fixture.inputs["problem"].as_str();
    let state_start = fixture.inputs["y0"].as_vector();
    let time_start = fixture.inputs["t0"].as_scalar();
    let times = fixture.inputs["times"].as_vector();
    let (rows, cols, states) = fixture.expected["states"].as_matrix();
    assert_eq!(cols, N, "{problem}: column count");
    assert_eq!(rows, times.len(), "{problem}: row count");
    let tol = fixture.tolerances.f64;

    let state_start_vec = Vector::<N>::from_fn(|i| state_start[i]);
    let mut out = vec![Vector::<N>::zeros(); times.len()];
    Rk45::<f64>::default()
        .with_rtol(1e-10)
        .with_atol(1e-12)
        .solve_on_grid(&f, time_start, &state_start_vec, &times, &mut out)
        .unwrap();
    for (i, y) in out.iter().enumerate() {
        for j in 0..N {
            let want = states[i * N + j];
            assert!(
                close(y[j], want, tol),
                "{problem}[t{i}][{j}]: got {}, want {want}, tol {tol:?}",
                y[j]
            );
        }
    }
}

// ω(t) = [0.8·cos(1.3 t), 0.5·sin(0.7 t), 1.1], the same rate the generator prescribes.
fn prescribed_rate(time: f64) -> Vector3D<f64> {
    Vector::new([0.8 * (1.3 * time).cos(), 0.5 * (0.7 * time).sin(), 1.1])
}

fn prescribed_rate_change(time: f64) -> Vector3D<f64> {
    Vector::new([-1.04 * (1.3 * time).sin(), 0.35 * (0.7 * time).cos(), 0.0])
}

// The orientation a golden row names, built the same way on both sides so the four numbers never
// have to be compared directly.
fn orientation_from(row: &[f64]) -> SO3<f64> {
    SO3::from_quaternion(
        Quaternion::new(row[0], row[1], row[2], row[3])
            .try_normalized()
            .unwrap(),
    )
}

// How many whole steps of `timestep` it takes to reach `sample_time` from `time_start`, refusing a
// grid that does not divide — otherwise a mismatched grid would quietly stop short of each sample.
// Counted from `time_start` every time rather than from the last sample, so the count never picks
// up the drift that adding the step over and over would leave behind.
fn steps_to(case: &str, time_start: f64, sample_time: f64, timestep: f64) -> usize {
    let count = (sample_time - time_start) / timestep;
    assert!(
        (count - count.round()).abs() < 1e-6,
        "{case}: the sample grid does not divide into whole steps of {timestep}"
    );
    count.round() as usize
}

fn check_prescribed_rate_attitude(fixture: &Fixture, second_order: bool) {
    let case = fixture.case.as_str();
    let state_start = fixture.inputs["y0"].as_vector();
    let time_start = fixture.inputs["t0"].as_scalar();
    let times = fixture.inputs["times"].as_vector();
    let timestep = fixture.inputs["timestep"].as_scalar();
    let (rows, cols, states) = fixture.expected["states"].as_matrix();
    assert_eq!(cols, 4, "{case}: column count");
    assert_eq!(rows, times.len(), "{case}: row count");
    let tol = fixture.tolerances.f64;

    let mut orientation = orientation_from(&state_start);
    let mut taken = 0;
    let mut worst = 0.0_f64;
    for (index, &sample_time) in times.iter().enumerate() {
        let wanted = steps_to(case, time_start, sample_time, timestep);
        while taken < wanted {
            let time = time_start + taken as f64 * timestep;
            orientation = if second_order {
                ExponentialMap::attitude_step_with_angular_acceleration(
                    orientation,
                    prescribed_rate(time),
                    prescribed_rate_change(time),
                    timestep,
                )
            } else {
                ExponentialMap::attitude_step(orientation, prescribed_rate(time), timestep)
            };
            taken += 1;
        }
        // Compare the turn between the two orientations rather than the four numbers: a
        // quaternion and its negation name the same rotation.
        let golden = orientation_from(&states[index * 4..index * 4 + 4]);
        let apart = (orientation.inverse() * golden).log().norm();
        worst = worst.max(apart);
        assert!(
            close(apart, 0.0, tol),
            "{case}[t{index}]: {apart} rad from the golden, tol {tol:?}"
        );
    }
    println!("{case}: worst angle {worst:e} rad");
}

fn check_tumbling_free_body(fixture: &Fixture) {
    let case = fixture.case.as_str();
    let state_start = fixture.inputs["y0"].as_vector();
    let time_start = fixture.inputs["t0"].as_scalar();
    let times = fixture.inputs["times"].as_vector();
    let timestep = fixture.inputs["timestep"].as_scalar();
    let (rows, cols, states) = fixture.expected["states"].as_matrix();
    assert_eq!(cols, 13, "{case}: column count");
    assert_eq!(rows, times.len(), "{case}: row count");
    let tol = fixture.tolerances.f64;

    let inertia = SpatialInertia::from_diagonal_inertia(
        0.8,
        Vector::new([0.0, 0.0, 0.0]),
        Vector::new([0.005, 0.007, 0.009]),
    )
    .unwrap();
    let body = RigidBody::new(inertia, Vector::new([0.0, 0.0, -9.81])).unwrap();
    let wrench = Wrench::new(
        Vector::new([0.0, 0.0, 8.0]),
        Vector::new([0.02, -0.01, 0.005]),
    );

    let start = Vector::<13>::from_fn(|index| state_start[index]);
    let mut state: FreeJointState<f64> = free_joint_from_state_vector(&start).unwrap();
    let mut taken = 0;
    let (mut worst_position, mut worst_angle, mut worst_velocity) = (0.0_f64, 0.0_f64, 0.0_f64);

    for (index, &sample_time) in times.iter().enumerate() {
        let wanted = steps_to(case, time_start, sample_time, timestep);
        while taken < wanted {
            state = body.stepped(state, wrench, timestep);
            taken += 1;
        }
        let row = &states[index * 13..index * 13 + 13];

        let position = state.pose().translation();
        for axis in 0..3 {
            let want = row[axis];
            worst_position = worst_position.max((position[axis] - want).abs());
            assert!(
                close(position[axis], want, tol),
                "{case}[t{index}] position[{axis}]: got {}, want {want}, tol {tol:?}",
                position[axis]
            );
        }

        let golden_orientation = orientation_from(&row[3..7]);
        let apart = (state.pose().rotation().inverse() * golden_orientation)
            .log()
            .norm();
        worst_angle = worst_angle.max(apart);
        assert!(
            close(apart, 0.0, tol),
            "{case}[t{index}]: {apart} rad from the golden, tol {tol:?}"
        );

        let linear = state.velocity().linear();
        let angular = state.velocity().angular();
        for axis in 0..3 {
            let want_linear = row[7 + axis];
            let want_angular = row[10 + axis];
            worst_velocity = worst_velocity
                .max((linear[axis] - want_linear).abs())
                .max((angular[axis] - want_angular).abs());
            assert!(
                close(linear[axis], want_linear, tol),
                "{case}[t{index}] velocity[{axis}]: got {}, want {want_linear}, tol {tol:?}",
                linear[axis]
            );
            assert!(
                close(angular[axis], want_angular, tol),
                "{case}[t{index}] turn rate[{axis}]: got {}, want {want_angular}, tol {tol:?}",
                angular[axis]
            );
        }
    }
    println!(
        "{case}: worst position {worst_position:e}, worst angle {worst_angle:e} rad, \
         worst velocity {worst_velocity:e}"
    );
}

#[test]
fn ode() {
    for fixture in load_dir("ode") {
        // Keyed on the case first: the two attitude cases share one problem key.
        match fixture.case.as_str() {
            "prescribed_rate_attitude_first_order" => {
                check_prescribed_rate_attitude(&fixture, false)
            }
            "prescribed_rate_attitude_second_order" => {
                check_prescribed_rate_attitude(&fixture, true)
            }
            "tumbling_free_body" => check_tumbling_free_body(&fixture),
            _ => match fixture.inputs["problem"].as_str() {
                "exp_decay" => run_case(&fixture, &ode_exp_decay),
                "harmonic" => run_case(&fixture, &ode_harmonic),
                "two_body" => run_case(&fixture, &ode_two_body),
                "van_der_pol_mild" => run_case(&fixture, &ode_van_der_pol_mild),
                other => panic!("unknown ode problem {other}"),
            },
        }
    }
}
