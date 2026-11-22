//! Position-servo tests: that the stored pair is what `zoh` returns, the critically damped and
//! underdamped step responses against their closed forms, that a tick many natural periods long is
//! still safe, that two half-steps compose into one, and the refusals.

use multicalc::discretization::zoh;
use multicalc::error::PlantError;
use multicalc::linear_algebra::{Matrix, Matrix2D, Vector};
use multicalc::plant::PositionServo;

/// `(F, G)` for one joint's servo, worked out here rather than taken from the model.
fn reference_pair(
    natural_frequency: f64,
    damping_ratio: f64,
    timestep: f64,
) -> (Matrix2D<f64>, Matrix<2, 1, f64>) {
    let stiffness = natural_frequency * natural_frequency;
    let a = Matrix2D::new([
        [0.0, 1.0],
        [-stiffness, -2.0 * damping_ratio * natural_frequency],
    ]);
    let b = Matrix::<2, 1, f64>::new([[0.0], [stiffness]]);
    zoh::<2, 1, 3, f64>(a, b, timestep).unwrap()
}

#[test]
fn discretization_matches_zoh() {
    let natural_frequency = 50.0;
    let damping_ratio = 0.7;
    let timestep = 0.001;
    let (transition, input) = reference_pair(natural_frequency, damping_ratio, timestep);

    // The model's own pair, read out by stepping from a known state: one tick from `[1, 0]` is F's
    // first column, from `[0, 1]` is F's second, and from rest under a unit command is G.
    let mut servo =
        PositionServo::<1, f64>::uniform(natural_frequency, damping_ratio, timestep).unwrap();
    for (column, seed) in [(0, [1.0, 0.0]), (1, [0.0, 1.0])] {
        servo = servo.with_state(Vector::new([seed[0]]), Vector::new([seed[1]]));
        let _ = servo.stepped(Vector::zeros());
        assert!((servo.positions()[0] - transition[(0, column)]).abs() < 1e-14);
        assert!((servo.velocities()[0] - transition[(1, column)]).abs() < 1e-14);
    }
    servo.reset();
    let _ = servo.stepped(Vector::new([1.0]));
    assert!((servo.positions()[0] - input[(0, 0)]).abs() < 1e-14);
    assert!((servo.velocities()[0] - input[(1, 0)]).abs() < 1e-14);
}

#[test]
fn critically_damped_step_matches_the_closed_form() {
    let natural_frequency = 50.0;
    let timestep = 0.001;
    let commanded = 0.4;
    let mut servo = PositionServo::<1, f64>::uniform(natural_frequency, 1.0, timestep).unwrap();

    let mut ticks = 0;
    for target in [1, 20, 200] {
        while ticks < target {
            let _ = servo.stepped(Vector::new([commanded]));
            ticks += 1;
        }
        let elapsed = timestep * f64::from(ticks);
        let decay = (-natural_frequency * elapsed).exp();
        let position = commanded * (1.0 - (1.0 + natural_frequency * elapsed) * decay);
        let velocity = commanded * natural_frequency * natural_frequency * elapsed * decay;
        assert!(
            (servo.positions()[0] - position).abs() < 1e-12,
            "position after {ticks}: {} vs {position}",
            servo.positions()[0]
        );
        assert!(
            (servo.velocities()[0] - velocity).abs() < 1e-12,
            "rate after {ticks}: {} vs {velocity}",
            servo.velocities()[0]
        );
    }
}

#[test]
fn underdamped_step_matches_the_closed_form() {
    let natural_frequency = 50.0;
    let damping_ratio = 0.5;
    let timestep = 0.001;
    let commanded = 0.4;
    let damped_frequency = natural_frequency * (1.0_f64 - damping_ratio * damping_ratio).sqrt();
    let mut servo =
        PositionServo::<1, f64>::uniform(natural_frequency, damping_ratio, timestep).unwrap();

    let mut ticks = 0;
    for target in [1, 20, 200] {
        while ticks < target {
            let _ = servo.stepped(Vector::new([commanded]));
            ticks += 1;
        }
        let elapsed = timestep * f64::from(ticks);
        let decay = (-damping_ratio * natural_frequency * elapsed).exp();
        let position = commanded
            * (1.0
                - decay
                    * ((damped_frequency * elapsed).cos()
                        + (damping_ratio * natural_frequency / damped_frequency)
                            * (damped_frequency * elapsed).sin()));
        assert!(
            (servo.positions()[0] - position).abs() < 1e-12,
            "after {ticks}: {} vs {position}",
            servo.positions()[0]
        );
    }
}

#[test]
fn a_long_tick_is_as_safe_as_a_short_one() {
    // 0.1 s is about fifty natural periods, where an explicit integrator would diverge.
    let mut servo = PositionServo::<1, f64>::uniform(500.0, 1.0, 0.1).unwrap();
    let commanded = Vector::new([0.4]);
    for _ in 0..100 {
        let _ = servo.stepped(commanded);
    }
    assert!(servo.positions()[0].is_finite());
    assert!(
        (servo.positions()[0] - 0.4).abs() < 1e-9,
        "{}",
        servo.positions()[0]
    );
}

#[test]
fn stepped_over_agrees_with_stepped() {
    let timestep = 0.001;
    let commanded = Vector::new([0.4, -0.2]);
    let mut whole = PositionServo::<2, f64>::uniform(50.0, 0.7, timestep).unwrap();
    let mut halves = whole;

    let _ = whole.stepped(commanded);
    let _ = halves.stepped_over(commanded, timestep / 2.0).unwrap();
    let _ = halves.stepped_over(commanded, timestep / 2.0).unwrap();

    for joint in 0..2 {
        assert!((whole.positions()[joint] - halves.positions()[joint]).abs() < 1e-12);
        assert!((whole.velocities()[joint] - halves.velocities()[joint]).abs() < 1e-12);
    }
}

#[test]
fn construction_is_validated() {
    for natural_frequency in [0.0, -1.0] {
        assert_eq!(
            PositionServo::<2, f64>::uniform(natural_frequency, 1.0, 0.001),
            Err(PlantError::NonPositiveNaturalFrequency)
        );
    }
    assert_eq!(
        PositionServo::<2, f64>::uniform(50.0, -0.1, 0.001),
        Err(PlantError::NegativeDampingRatio)
    );
    assert_eq!(
        PositionServo::<2, f64>::uniform(50.0, 1.0, 0.0),
        Err(PlantError::NonPositiveTimestep)
    );
    for (natural_frequency, damping_ratio, timestep) in [
        (f64::NAN, 1.0, 0.001),
        (50.0, f64::NAN, 0.001),
        (50.0, 1.0, f64::NAN),
    ] {
        assert_eq!(
            PositionServo::<2, f64>::uniform(natural_frequency, damping_ratio, timestep),
            Err(PlantError::NonFinite)
        );
    }
}

#[test]
fn reset_zeroes_the_state() {
    let mut servo = PositionServo::<2, f64>::uniform(50.0, 1.0, 0.001).unwrap();
    for _ in 0..50 {
        let _ = servo.stepped(Vector::new([0.4, -0.2]));
    }
    assert!(servo.positions().norm() > 0.0);

    servo.reset();
    assert_eq!(servo.positions(), Vector::zeros());
    assert_eq!(servo.velocities(), Vector::zeros());
}
