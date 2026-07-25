use multicalc::kinematics::BodyTwist;
use multicalc::linear_algebra::Vector;
use multicalc_demos::sim::wheeled_vehicle::WheeledVehicle;
use rand::SeedableRng;
use rand_pcg::Pcg32;

const WHEELBASE: f64 = 0.235;

fn noiseless() -> WheeledVehicle {
    WheeledVehicle::new(WHEELBASE, 0.0, 0.0, 1.2).unwrap()
}

#[test]
fn straight_motion_advances_x_only() {
    let mut rng = Pcg32::seed_from_u64(1);
    let step = noiseless().step(
        Vector::new([0.0, 0.0, 0.0]),
        BodyTwist::new(1.0, 0.0),
        0.1,
        false,
        &mut rng,
    );
    assert!(step.pose[0] > 0.0, "x should advance: {}", step.pose[0]);
    assert!(
        step.pose[1].abs() < 1e-12,
        "y should not move: {}",
        step.pose[1]
    );
    assert!(
        step.pose[2].abs() < 1e-12,
        "heading should not turn: {}",
        step.pose[2]
    );
}

#[test]
fn a_positive_turn_rate_steers_left() {
    let mut rng = Pcg32::seed_from_u64(2);
    let dt = 0.2;
    let yaw_rate = 1.0;
    let step = noiseless().step(
        Vector::new([0.0, 0.0, 0.0]),
        BodyTwist::new(0.5, yaw_rate),
        dt,
        false,
        &mut rng,
    );
    // Heading integrates the turn rate exactly, and turning left moves the vehicle up.
    assert!(
        (step.pose[2] - yaw_rate * dt).abs() < 1e-12,
        "heading: {}",
        step.pose[2]
    );
    assert!(step.pose[1] > 0.0, "should steer left: {}", step.pose[1]);
}

#[test]
fn zero_noise_reports_the_command() {
    let mut rng = Pcg32::seed_from_u64(3);
    let step = noiseless().step(
        Vector::new([0.0, 0.0, 0.0]),
        BodyTwist::new(0.8, -0.3),
        0.1,
        false,
        &mut rng,
    );
    assert_eq!(step.measured_speed, 0.8);
    assert_eq!(step.measured_yaw_rate, -0.3);
}

#[test]
fn slipping_scales_only_the_measured_speed() {
    let pose = Vector::new([0.2, 0.1, 0.3]);
    let command = BodyTwist::new(1.0, 0.4);

    let mut rng = Pcg32::seed_from_u64(4);
    let slipping = noiseless().step(pose, command, 0.1, true, &mut rng);
    let mut rng = Pcg32::seed_from_u64(4);
    let rolling = noiseless().step(pose, command, 0.1, false, &mut rng);

    // The reported speed is scaled up, but the truth and the reported turn rate are untouched.
    assert_eq!(slipping.measured_speed, 1.2);
    assert_eq!(slipping.pose.into_array(), rolling.pose.into_array());
    assert_eq!(slipping.measured_yaw_rate, rolling.measured_yaw_rate);
}

#[test]
fn a_fixed_seed_reproduces_the_step() {
    let vehicle = WheeledVehicle::new(WHEELBASE, 0.05, 0.05, 1.2).unwrap();
    let pose = Vector::new([0.0, 0.0, 0.0]);
    let command = BodyTwist::new(1.0, 0.2);

    let mut first = Pcg32::seed_from_u64(9);
    let mut second = Pcg32::seed_from_u64(9);
    assert_eq!(
        vehicle.step(pose, command, 0.1, false, &mut first),
        vehicle.step(pose, command, 0.1, false, &mut second)
    );
}
