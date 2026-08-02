//! Four feedback loops: a PID on a motor, an optimal law on a cart carrying a pole, attitude
//! control on a tumbling body, and all of it together flying a set of waypoints and coming home.
//!
//! Run with: `cargo run -p multicalc-demos --example control_loops`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::SO3;
use multicalc::control::{GeometricAttitudeController, Lqr, Pid, thrust_command_from_acceleration};
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::motion::{MinimumSnapPlanner, durations_from_average_speed};
use multicalc::zoh;

const GRAVITY: f64 = 9.81;

fn main() {
    motor_speed_loop();
    cart_and_pole();
    tumbling_body();
    waypoint_mission();
}

/// A motor whose speed chases its command with a lag, driven by a PID.
fn motor_speed_loop() {
    println!("PID on a motor");

    const DT: f64 = 0.001;
    const TIME_CONSTANT: f64 = 0.05;
    let step = |speed: f64, command: f64| speed + DT * (command - speed) / TIME_CONSTANT;

    let setpoint = 0.6;
    let mut controller = Pid::new(4.0, 20.0, 0.02, DT)
        .unwrap()
        .with_output_limits(-1.0, 1.0)
        .unwrap()
        .with_derivative_filter(0.2)
        .unwrap();

    let mut speed = 0.0;
    for _ in 0..2000 {
        let command = controller.update(setpoint, speed);
        speed = step(speed, command);
    }
    println!("  settled at {speed:.4} against a setpoint of {setpoint:.2}");
    assert!((speed - setpoint).abs() < 1e-3, "did not settle: {speed}");

    // The derivative watches the measurement, not the error, so moving the setpoint does not send
    // a spike through it. Differentiating the error would have thrown 0.3 / 0.001 through the
    // derivative gain instead.
    let before_the_step = controller.update(setpoint, speed);
    let on_the_step = controller.update(0.9, speed);
    println!(
        "  command before a 0.3 setpoint jump {before_the_step:.4}, on the jump {on_the_step:.4}"
    );
    let jump = (on_the_step - before_the_step).abs();
    assert!(
        jump <= 4.0 * 0.3,
        "the setpoint jump spiked the command: {jump}"
    );

    // Handover: something else has been driving the motor, and the controller picks it up without
    // the command stepping.
    let manual_command = 0.45;
    let mut speed = 0.0;
    for _ in 0..200 {
        speed = step(speed, manual_command);
    }
    let mut controller = Pid::new(4.0, 20.0, 0.02, DT).unwrap();
    controller
        .resume_from(manual_command, setpoint, speed)
        .unwrap();
    let first_automatic = controller.update(setpoint, speed);
    println!(
        "  took over a manual command of {manual_command:.2}, first automatic {first_automatic:.4}"
    );
    assert!(
        (first_automatic - manual_command).abs() < 1e-12,
        "handover stepped the command: {first_automatic}"
    );
    println!();
}

/// The cart-and-pole system, as continuous matrices: cart 1.0 kg carrying a pole of 0.1 kg whose
/// balance point is 0.5 m up. The state is cart position, cart speed, pole tilt, and how fast the
/// tilt changes.
fn cart_and_pole_system() -> (Matrix<4, 4>, Matrix<4, 1>) {
    let cart_mass = 1.0;
    let pole_mass = 0.1;
    let pole_half_length = 0.5;
    let continuous_state = Matrix::<4, 4>::new([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, pole_mass * GRAVITY / cart_mass, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [
            0.0,
            0.0,
            (cart_mass + pole_mass) * GRAVITY / (cart_mass * pole_half_length),
            0.0,
        ],
    ]);
    let continuous_input = Matrix::<4, 1>::new([
        [0.0],
        [1.0 / cart_mass],
        [0.0],
        [-1.0 / (cart_mass * pole_half_length)],
    ]);
    zoh::<4, 1, 5, f64>(continuous_state, continuous_input, 0.02).unwrap()
}

/// An optimal feedback law catching a pole before it falls over.
fn cart_and_pole() {
    println!("Optimal feedback on a cart carrying a pole");

    let (state_transition, input_model) = cart_and_pole_system();
    let state_cost = Matrix::<4, 4>::from_diagonal([10.0, 1.0, 10.0, 1.0]);
    let input_cost = Matrix::<1, 1>::new([[0.1]]);
    let controller = Lqr::new(state_transition, input_model, state_cost, input_cost).unwrap();

    let gain = controller.gain();
    println!(
        "  gain [{:.3}, {:.3}, {:.3}, {:.3}]",
        gain[(0, 0)],
        gain[(0, 1)],
        gain[(0, 2)],
        gain[(0, 3)]
    );
    let cost_to_go = controller.cost_to_go();
    println!(
        "  cost-to-go diagonal [{:.2}, {:.2}, {:.2}, {:.2}]",
        cost_to_go[(0, 0)],
        cost_to_go[(1, 1)],
        cost_to_go[(2, 2)],
        cost_to_go[(3, 3)]
    );

    let _ = controller.certify_stability().unwrap();
    println!("  closed loop certified: P is positive definite (design-time check)");

    // A pole leaning about 8.6 degrees.
    let mut state = Vector::new([0.0, 0.0, 0.15, 0.0]);
    for step in 0..1000 {
        if step % 200 == 0 {
            println!("  step {step:4}: tilt {:+.4} rad", state[2]);
        }
        state = state_transition * state + input_model * controller.control(state);
    }
    println!("  step 1000: tilt {:+.4} rad", state[2]);
    assert!(state.norm() < 1e-3, "the pole was not caught: {state:?}");
    println!();
}

/// A rigid body thrown into a tumble, brought back to pointing where it should.
fn tumbling_body() {
    println!("Attitude control on a tumbling body");

    let inertia = Matrix::<3, 3>::from_diagonal([0.02, 0.02, 0.04]);
    let inverse_inertia = Matrix::<3, 3>::from_diagonal([50.0, 50.0, 25.0]);
    let controller = GeometricAttitudeController::new(6.0, 1.2, inertia).unwrap();
    let target = SO3::<f64>::identity();
    let still = Vector::new([0.0, 0.0, 0.0]);

    let mut attitude = SO3::exp(Vector::new([0.6, -0.4, 0.9]));
    let mut body_rate = Vector::new([1.5, -1.0, 0.8]);
    const DT: f64 = 0.002;
    for step in 0..4000 {
        if step % 800 == 0 {
            let error = GeometricAttitudeController::<f64>::attitude_error(attitude, target);
            println!(
                "  step {step:4}: off by {:.4} rad, turning at {:.4} rad/s",
                error.norm(),
                body_rate.norm()
            );
        }
        let torque = controller.torque(attitude, body_rate, target, still, still);
        // What is left over after the body's own spin drives the change in turn rate.
        let rate_change = inverse_inertia * (torque - body_rate.cross(inertia * body_rate));
        body_rate += rate_change.scale(DT);
        // Stepping the rotation along its own turn rate keeps it a rotation, so nothing has to be
        // corrected afterwards.
        attitude = attitude.compose(SO3::exp(body_rate.scale(DT))).normalized();
    }
    let error = GeometricAttitudeController::<f64>::attitude_error(attitude, target);
    println!(
        "  step 4000: off by {:.6} rad, turning at {:.6} rad/s",
        error.norm(),
        body_rate.norm()
    );
    assert!(error.norm() < 1e-3, "still off target: {}", error.norm());
    assert!(
        body_rate.norm() < 1e-2,
        "still turning: {}",
        body_rate.norm()
    );
    println!();
}

/// The whole chain: waypoints in, a smooth path through them, a position loop, the tilt that loop
/// asks for, an attitude loop, and a body that flies it.
fn waypoint_mission() {
    println!("A waypoint mission, out and back home");

    const HOME: [f64; 3] = [0.0, 0.0, 1.0];
    let waypoints = [
        Vector::new(HOME),
        Vector::new([4.0, 0.0, 2.0]),
        Vector::new([4.0, 4.0, 2.0]),
        Vector::new([0.0, 4.0, 3.0]),
        Vector::new(HOME),
    ];

    // Plan the path once, before anything flies.
    const AVERAGE_SPEED: f64 = 1.5;
    let mut durations = [0.0; 4];
    durations_from_average_speed(&waypoints, AVERAGE_SPEED, &mut durations).unwrap();
    let planner = MinimumSnapPlanner::<4, 9, 3, f64>::new();
    let trajectory = planner.plan(&waypoints, &durations).unwrap();

    // Two loops at two rates, which is how a flight stack is built and why the attitude controller
    // is its own block.
    const INNER_STEP: f64 = 0.002; // 500 Hz attitude loop
    const TICKS_PER_OUTER: usize = 10; // so the position loop runs at 50 Hz
    const OUTER_STEP: f64 = INNER_STEP * TICKS_PER_OUTER as f64;

    // The position loop, over three positions then three speeds, against a plant that carries
    // speed into position.
    let mut state_transition = Matrix::<6, 6>::identity();
    let mut input_model = Matrix::<6, 3>::zeros();
    for axis in 0..3 {
        state_transition[(axis, axis + 3)] = OUTER_STEP;
        input_model[(axis, axis)] = 0.5 * OUTER_STEP * OUTER_STEP;
        input_model[(axis + 3, axis)] = OUTER_STEP;
    }
    let state_cost = Matrix::<6, 6>::from_diagonal([8.0, 8.0, 8.0, 2.0, 2.0, 2.0]);
    let input_cost = Matrix::<3, 3>::from_diagonal([0.4, 0.4, 0.4]);
    let position_loop = Lqr::new(state_transition, input_model, state_cost, input_cost).unwrap();
    let _ = position_loop.certify_stability().unwrap();
    println!("  closed loop certified: P is positive definite (design-time check)");

    let inertia = Matrix::<3, 3>::from_diagonal([0.02, 0.02, 0.04]);
    let inverse_inertia = Matrix::<3, 3>::from_diagonal([50.0, 50.0, 25.0]);
    let attitude_loop = GeometricAttitudeController::new(60.0, 8.0, inertia).unwrap();
    let facing_along_x = 0.0;
    let still = Vector::new([0.0, 0.0, 0.0]);

    let mut position = Vector::new(HOME);
    let mut velocity = Vector::new([0.0, 0.0, 0.0]);
    let mut attitude = SO3::<f64>::identity();
    let mut body_rate = Vector::new([0.0, 0.0, 0.0]);

    let mission_time: f64 = durations.iter().sum();
    let ticks = (mission_time / INNER_STEP) as usize;
    let mut command = thrust_command_from_acceleration(still, facing_along_x, GRAVITY).unwrap();
    let mut closest = [f64::INFINITY; 5];

    for tick in 0..ticks {
        let time = tick as f64 * INNER_STEP;

        if tick % TICKS_PER_OUTER == 0 {
            // Where the path says to be, how fast, and how hard it is turning right now.
            let [wanted_position, wanted_velocity, wanted_acceleration] =
                trajectory.evaluate_with_derivatives::<3>(time).unwrap();

            let mut state = [0.0; 6];
            let mut reference = [0.0; 6];
            for axis in 0..3 {
                state[axis] = position[axis];
                state[axis + 3] = velocity[axis];
                reference[axis] = wanted_position[axis];
                reference[axis + 3] = wanted_velocity[axis];
            }
            // The path's own turn is handed straight through, so the loop only has to answer the
            // gap.
            let acceleration_command = position_loop.control_tracking(
                Vector::new(state),
                Vector::new(reference),
                wanted_acceleration,
            );
            command =
                thrust_command_from_acceleration(acceleration_command, facing_along_x, GRAVITY)
                    .unwrap();
        }

        // Attitude loop: close the gap to the tilt the position loop asked for.
        let torque = attitude_loop.torque(attitude, body_rate, command.attitude(), still, still);

        // The body. Thrust pushes out along its own up axis; gravity pulls straight down.
        let body_up = attitude.act(Vector::new([0.0, 0.0, 1.0]));
        let acceleration =
            body_up.scale(command.thrust_acceleration()) - Vector::new([0.0, 0.0, GRAVITY]);
        velocity += acceleration.scale(INNER_STEP);
        position += velocity.scale(INNER_STEP);

        let rate_change = inverse_inertia * (torque - body_rate.cross(inertia * body_rate));
        body_rate += rate_change.scale(INNER_STEP);
        attitude = attitude
            .compose(SO3::exp(body_rate.scale(INNER_STEP)))
            .normalized();

        for (slot, waypoint) in closest.iter_mut().zip(waypoints.iter()) {
            let distance = (position - *waypoint).norm();
            if distance < *slot {
                *slot = distance;
            }
        }
    }

    for (index, distance) in closest.iter().enumerate() {
        println!("  waypoint {index}: closest approach {distance:.3} m");
        assert!(
            *distance < 0.40,
            "waypoint {index} missed by {distance:.3} m"
        );
    }
    let home_distance = (position - Vector::new(HOME)).norm();
    println!("  back home, {home_distance:.3} m from where it started");
    assert!(
        home_distance < 0.40,
        "did not get home: {home_distance:.3} m"
    );
    assert!(
        velocity.norm() < 0.5,
        "still moving at the end: {}",
        velocity.norm()
    );
    println!("  turn rate reference held at zero; the path's own turn is not fed forward");
}
