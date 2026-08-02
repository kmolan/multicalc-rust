//! The two light attitude filters: what they share, what each does on its own, and how they hold
//! up in single precision.

use multicalc::error::EstimationError;
use multicalc::estimation::{MadgwickFilter, MahonyFilter};
use multicalc::linear_algebra::{Vector, Vector3D};
use multicalc::ode::ExponentialMap;
use multicalc::random::{Pcg32, RandomSource};
use multicalc::spatial::SO3;

const GRAVITY_STRENGTH: f64 = 9.81;
const MAGNETIC_DIP: f64 = 60.0;

// ---- helpers ----------------------------------------------------------------

/// What a push sensor reads on a still body held at the given facing.
fn still_accelerometer(orientation: SO3<f64>) -> Vector3D<f64> {
    orientation
        .inverse()
        .act(Vector::new([0.0, 0.0, GRAVITY_STRENGTH]))
}

/// What a magnetometer reads at the given facing, in a field pointing north and 60 degrees down.
fn dipping_magnetometer(orientation: SO3<f64>) -> Vector3D<f64> {
    let dip = MAGNETIC_DIP.to_radians();
    orientation
        .inverse()
        .act(Vector::new([dip.cos(), 0.0, -dip.sin()]))
}

/// How far one facing is from another, in radians.
fn angle_between(a: SO3<f64>, b: SO3<f64>) -> f64 {
    (a.inverse() * b).log().norm()
}

/// The same, in single precision.
fn angle_between_f32(a: SO3<f32>, b: SO3<f32>) -> f32 {
    (a.inverse() * b).log().norm()
}

/// What a push sensor reads on a still body, in single precision.
fn still_accelerometer_f32(orientation: SO3<f32>) -> Vector3D<f32> {
    orientation
        .inverse()
        .act(Vector::new([0.0, 0.0, GRAVITY_STRENGTH as f32]))
}

/// What a magnetometer reads at the given facing, in single precision.
fn dipping_magnetometer_f32(orientation: SO3<f32>) -> Vector3D<f32> {
    let dip = (MAGNETIC_DIP as f32).to_radians();
    orientation
        .inverse()
        .act(Vector::new([dip.cos(), 0.0, -dip.sin()]))
}

// ---- what both filters must do ----------------------------------------------

#[test]
fn mahony_still_and_level_body_stays_put() {
    let level = SO3::<f64>::identity();
    let mut filter = MahonyFilter::new(level);

    let not_turning = Vector::zeros();
    let timestep = 0.002;
    for _ in 0..1000 {
        filter
            .step(
                not_turning,
                still_accelerometer(level),
                Some(dipping_magnetometer(level)),
                timestep,
            )
            .unwrap();
    }

    assert!(angle_between(filter.orientation(), level) < 1e-12);
    assert!(filter.gyroscope_bias().norm() < 1e-12);
}

#[test]
fn madgwick_still_and_level_body_stays_put() {
    let level = SO3::<f64>::identity();
    let mut filter = MadgwickFilter::new(level);

    let not_turning = Vector::zeros();
    let timestep = 0.002;
    for _ in 0..1000 {
        filter
            .step(
                not_turning,
                still_accelerometer(level),
                Some(dipping_magnetometer(level)),
                timestep,
            )
            .unwrap();
    }

    assert!(angle_between(filter.orientation(), level) < 1e-12);
    assert!(filter.gyroscope_bias().norm() < 1e-12);
}

#[test]
fn mahony_gyroscope_only_matches_the_exponential_map() {
    let start = SO3::exp(Vector::new([0.2, 0.1, -0.3]));
    let mut filter = MahonyFilter::new(start);

    // A dead push sensor and no magnetometer: nothing corrects, so this is pure integration.
    let dead_accelerometer = Vector::zeros();
    let turn_rate = Vector::new([0.3, -0.2, 0.5]);
    let timestep = 0.01;

    let mut expected = start;
    for _ in 0..500 {
        filter
            .step(turn_rate, dead_accelerometer, None, timestep)
            .unwrap();
        expected = ExponentialMap::attitude_step(expected, turn_rate, timestep).normalized();
    }

    assert!(angle_between(filter.orientation(), expected) < 1e-13);
    assert_eq!(filter.gyroscope_bias(), Vector3D::zeros());
}

#[test]
fn madgwick_gyroscope_only_matches_the_exponential_map() {
    let start = SO3::exp(Vector::new([0.2, 0.1, -0.3]));
    let mut filter = MadgwickFilter::new(start);

    let dead_accelerometer = Vector::zeros();
    let turn_rate = Vector::new([0.3, -0.2, 0.5]);
    let timestep = 0.01;

    let mut expected = start;
    for _ in 0..500 {
        filter
            .step(turn_rate, dead_accelerometer, None, timestep)
            .unwrap();
        expected = ExponentialMap::attitude_step(expected, turn_rate, timestep).normalized();
    }

    assert!(angle_between(filter.orientation(), expected) < 1e-13);
    assert_eq!(filter.gyroscope_bias(), Vector3D::zeros());
}

#[test]
fn mahony_converges_from_a_wrong_start() {
    let level = SO3::<f64>::identity();
    let mut filter = MahonyFilter::new(SO3::exp(Vector::new([0.6, -0.4, 0.9])));

    // A start this far out winds the running total up hard, so the facing rings for a minute or
    // so on the way in. Two minutes is what it takes to settle at the default gains.
    let not_turning = Vector::zeros();
    let timestep = 0.005;
    for _ in 0..24_000 {
        filter
            .step(
                not_turning,
                still_accelerometer(level),
                Some(dipping_magnetometer(level)),
                timestep,
            )
            .unwrap();
    }

    assert!(angle_between(filter.orientation(), level) < 0.02);
}

#[test]
fn madgwick_converges_from_a_wrong_start() {
    let level = SO3::<f64>::identity();
    let mut filter =
        MadgwickFilter::new(SO3::exp(Vector::new([0.6, -0.4, 0.9]))).with_correction_gain(0.3);

    let not_turning = Vector::zeros();
    let timestep = 0.005;
    for _ in 0..4000 {
        filter
            .step(
                not_turning,
                still_accelerometer(level),
                Some(dipping_magnetometer(level)),
                timestep,
            )
            .unwrap();
    }

    assert!(angle_between(filter.orientation(), level) < 0.02);
}

#[test]
fn mahony_keeps_the_facing_a_true_rotation() {
    let mut filter = MahonyFilter::new(SO3::<f64>::identity());
    let mut truth = SO3::<f64>::identity();
    let mut generator = Pcg32::new(20260802);

    let turn_rate = Vector::new([0.7, -0.5, 1.1]);
    let jitter = 0.05;
    let timestep = 0.001;
    for _ in 0..20_000 {
        let reading = turn_rate + Vector::from_fn(|_| jitter * generator.standard_normal());
        filter
            .step(
                reading,
                still_accelerometer(truth),
                Some(dipping_magnetometer(truth)),
                timestep,
            )
            .unwrap();
        truth = ExponentialMap::attitude_step(truth, turn_rate, timestep);

        assert!((filter.orientation().quaternion().norm() - 1.0).abs() < 1e-12);
    }
}

#[test]
fn madgwick_keeps_the_facing_a_true_rotation() {
    let mut filter = MadgwickFilter::new(SO3::<f64>::identity());
    let mut truth = SO3::<f64>::identity();
    let mut generator = Pcg32::new(20260802);

    let turn_rate = Vector::new([0.7, -0.5, 1.1]);
    let jitter = 0.05;
    let timestep = 0.001;
    for _ in 0..20_000 {
        let reading = turn_rate + Vector::from_fn(|_| jitter * generator.standard_normal());
        filter
            .step(
                reading,
                still_accelerometer(truth),
                Some(dipping_magnetometer(truth)),
                timestep,
            )
            .unwrap();
        truth = ExponentialMap::attitude_step(truth, turn_rate, timestep);

        assert!((filter.orientation().quaternion().norm() - 1.0).abs() < 1e-12);
    }
}

#[test]
fn mahony_rejects_non_finite_inputs() {
    let level = SO3::<f64>::identity();
    let good_push = still_accelerometer(level);
    let good_field = dipping_magnetometer(level);
    let not_turning: Vector3D<f64> = Vector::zeros();
    let timestep = 0.01;

    let filter = MahonyFilter::new(level);

    let cases: [(Vector3D<f64>, Vector3D<f64>, Option<Vector3D<f64>>, f64); 4] = [
        (
            Vector::new([f64::NAN, 0.0, 0.0]),
            good_push,
            Some(good_field),
            timestep,
        ),
        (
            not_turning,
            Vector::new([f64::INFINITY, 0.0, 0.0]),
            Some(good_field),
            timestep,
        ),
        (
            not_turning,
            good_push,
            Some(Vector::new([0.0, f64::NAN, 0.0])),
            timestep,
        ),
        (not_turning, good_push, Some(good_field), f64::NAN),
    ];

    for (gyroscope, accelerometer, magnetometer, step_size) in cases {
        let mut attempted = filter;
        assert_eq!(
            attempted.step(gyroscope, accelerometer, magnetometer, step_size),
            Err(EstimationError::NonFinite)
        );
        assert_eq!(attempted, filter);
    }
}

#[test]
fn madgwick_rejects_non_finite_inputs() {
    let level = SO3::<f64>::identity();
    let good_push = still_accelerometer(level);
    let good_field = dipping_magnetometer(level);
    let not_turning: Vector3D<f64> = Vector::zeros();
    let timestep = 0.01;

    let filter = MadgwickFilter::new(level);

    let cases: [(Vector3D<f64>, Vector3D<f64>, Option<Vector3D<f64>>, f64); 4] = [
        (
            Vector::new([f64::NAN, 0.0, 0.0]),
            good_push,
            Some(good_field),
            timestep,
        ),
        (
            not_turning,
            Vector::new([f64::INFINITY, 0.0, 0.0]),
            Some(good_field),
            timestep,
        ),
        (
            not_turning,
            good_push,
            Some(Vector::new([0.0, f64::NAN, 0.0])),
            timestep,
        ),
        (not_turning, good_push, Some(good_field), f64::NAN),
    ];

    for (gyroscope, accelerometer, magnetometer, step_size) in cases {
        let mut attempted = filter;
        assert_eq!(
            attempted.step(gyroscope, accelerometer, magnetometer, step_size),
            Err(EstimationError::NonFinite)
        );
        assert_eq!(attempted, filter);
    }
}

#[test]
fn mahony_magnetometer_never_leans_the_body() {
    // The field dips 60 degrees, nothing like the level north the filter references by default.
    let level = SO3::<f64>::identity();
    let mut filter = MahonyFilter::new(SO3::exp(Vector::new([0.0, 0.0, 0.5])));

    let not_turning = Vector::zeros();
    let timestep = 0.005;
    for _ in 0..24_000 {
        filter
            .step(
                not_turning,
                still_accelerometer(level),
                Some(dipping_magnetometer(level)),
                timestep,
            )
            .unwrap();
    }

    let (roll, pitch, yaw) = filter.orientation().quaternion().to_euler_zyx();
    assert!(roll.abs() < 1e-3, "roll leaned to {roll}");
    assert!(pitch.abs() < 1e-3, "pitch leaned to {pitch}");
    assert!(yaw.abs() < 1e-2, "heading left at {yaw}");
}

#[test]
fn madgwick_magnetometer_never_leans_the_body() {
    let level = SO3::<f64>::identity();
    // At the default gain one step of walking is 5e-4 rad, so the chatter stays under the bound
    // being checked and what is left to see is a lean or nothing.
    let mut filter = MadgwickFilter::new(SO3::exp(Vector::new([0.0, 0.0, 0.5])));

    let not_turning = Vector::zeros();
    let timestep = 0.005;
    for _ in 0..4000 {
        filter
            .step(
                not_turning,
                still_accelerometer(level),
                Some(dipping_magnetometer(level)),
                timestep,
            )
            .unwrap();
    }

    let (roll, pitch, yaw) = filter.orientation().quaternion().to_euler_zyx();
    assert!(roll.abs() < 1e-3, "roll leaned to {roll}");
    assert!(pitch.abs() < 1e-3, "pitch leaned to {pitch}");
    assert!(yaw.abs() < 1e-2, "heading left at {yaw}");
}

#[test]
fn mahony_leaves_a_correct_estimate_alone() {
    let truth = SO3::exp(Vector::new([0.4, -0.25, 0.9]));
    let mut filter = MahonyFilter::new(truth);

    filter
        .step(
            Vector::zeros(),
            still_accelerometer(truth),
            Some(dipping_magnetometer(truth)),
            0.01,
        )
        .unwrap();

    assert!(angle_between(filter.orientation(), truth) < 1e-13);
    assert!(filter.gyroscope_bias().norm() < 1e-13);
}

#[test]
fn madgwick_moves_one_walking_step_when_the_estimate_is_already_right() {
    // A fixed-rate walk cannot stand still: with nothing left to correct it still takes one step,
    // in whatever direction the leftover rounding points. That step is the whole of its error
    // floor, and it is what a caller trades for a pull that does not grow with the mistake.
    let truth = SO3::exp(Vector::new([0.4, -0.25, 0.9]));
    let correction_gain = 0.1;
    let bias_gain = 0.01;
    let timestep = 0.01;
    let mut filter = MadgwickFilter::new(truth)
        .with_correction_gain(correction_gain)
        .with_bias_gain(bias_gain);

    filter
        .step(
            Vector::zeros(),
            still_accelerometer(truth),
            Some(dipping_magnetometer(truth)),
            timestep,
        )
        .unwrap();

    let one_step = correction_gain * timestep;
    assert!(angle_between(filter.orientation(), truth) < 1.1 * one_step);
    assert!(filter.gyroscope_bias().norm() < 1.1 * bias_gain * timestep);
}

// ---- Mahony's own behaviour --------------------------------------------------

#[test]
fn mahony_learns_a_known_turn_rate_offset() {
    let level = SO3::<f64>::identity();
    let true_offset = Vector::new([0.02, -0.015, 0.01]);
    let mut filter = MahonyFilter::new(level)
        .with_proportional_gain(1.0)
        .with_integral_gain(0.3);

    let timestep = 0.005;
    for _ in 0..12_000 {
        filter
            .step(
                true_offset,
                still_accelerometer(level),
                Some(dipping_magnetometer(level)),
                timestep,
            )
            .unwrap();
    }

    let learned = filter.gyroscope_bias();
    for axis in 0..3 {
        assert!(
            (learned[axis] - true_offset[axis]).abs() < 5e-3,
            "axis {axis}: learned {} against {}",
            learned[axis],
            true_offset[axis]
        );
    }
    assert!(angle_between(filter.orientation(), level) < 0.01);
}

#[test]
fn mahony_zero_integral_gain_leaves_the_offset_alone() {
    let level = SO3::<f64>::identity();
    let true_offset = Vector::new([0.02, -0.015, 0.01]);
    let mut filter = MahonyFilter::new(level)
        .with_proportional_gain(1.0)
        .with_integral_gain(0.0);

    let timestep = 0.005;
    for _ in 0..12_000 {
        filter
            .step(
                true_offset,
                still_accelerometer(level),
                Some(dipping_magnetometer(level)),
                timestep,
            )
            .unwrap();
    }

    assert_eq!(filter.gyroscope_bias(), Vector3D::zeros());
    // With no offset learned the facing settles at a small steady lean, not at level.
    assert!(angle_between(filter.orientation(), level) > 1e-3);
}

#[test]
fn mahony_higher_proportional_gain_pulls_harder() {
    let level = SO3::<f64>::identity();
    let wrong_start = SO3::exp(Vector::new([0.3, 0.0, 0.0]));
    let mut gentle = MahonyFilter::new(wrong_start)
        .with_proportional_gain(0.5)
        .with_integral_gain(0.0);
    let mut firm = MahonyFilter::new(wrong_start)
        .with_proportional_gain(4.0)
        .with_integral_gain(0.0);

    let not_turning = Vector::zeros();
    let timestep = 0.005;
    for _ in 0..200 {
        for filter in [&mut gentle, &mut firm] {
            filter
                .step(
                    not_turning,
                    still_accelerometer(level),
                    Some(dipping_magnetometer(level)),
                    timestep,
                )
                .unwrap();
        }
    }

    assert!(angle_between(firm.orientation(), level) < angle_between(gentle.orientation(), level));
}

#[test]
fn mahony_matches_a_hand_stepped_loop() {
    let level = SO3::<f64>::identity();
    let start = SO3::exp(Vector::new([0.1, -0.2, 0.05]));
    let start_offset = Vector::new([0.01, 0.02, -0.03]);
    let proportional_gain = 2.0;
    let integral_gain = 0.5;
    let turn_rate = Vector::new([0.1, 0.2, 0.3]);
    let push = still_accelerometer(level);
    let field = dipping_magnetometer(level);
    let timestep = 0.01;

    let mut filter = MahonyFilter::new(start)
        .with_proportional_gain(proportional_gain)
        .with_integral_gain(integral_gain);
    filter.set_gyroscope_bias(start_offset);
    filter.step(turn_rate, push, Some(field), timestep).unwrap();

    let correction = hand_correction(start, push, field);
    let offset = start_offset - correction * integral_gain * timestep;
    let corrected_rate = turn_rate - offset + correction * proportional_gain;
    let orientation = ExponentialMap::attitude_step(start, corrected_rate, timestep).normalized();

    assert!(angle_between(filter.orientation(), orientation) < 1e-15);
    for axis in 0..3 {
        assert!((filter.gyroscope_bias()[axis] - offset[axis]).abs() < 1e-15);
    }
}

/// The correction term, written out here so a change to the filter's own ordering shows up.
fn hand_correction(
    orientation: SO3<f64>,
    accelerometer_reading: Vector3D<f64>,
    magnetometer_reading: Vector3D<f64>,
) -> Vector3D<f64> {
    let upward = Vector::new([0.0, 0.0, 1.0]);
    let north = Vector::new([1.0, 0.0, 0.0]);

    let push = accelerometer_reading.normalized();
    let from_push = push.cross(orientation.inverse().act(upward));

    let field = magnetometer_reading.normalized();
    let in_world = orientation.act(field);
    let vertical = in_world.dot(upward);
    let horizontal = (1.0 - vertical * vertical).max(0.0).sqrt();
    let reference = north * horizontal + upward * vertical;
    let from_field = field.cross(orientation.inverse().act(reference));

    from_push + from_field
}

// ---- Madgwick's own behaviour ------------------------------------------------

#[test]
fn madgwick_learns_a_known_turn_rate_offset() {
    let level = SO3::<f64>::identity();
    let true_offset = Vector::new([0.02, -0.015, 0.01]);
    let mut filter = MadgwickFilter::new(level)
        .with_correction_gain(0.3)
        .with_bias_gain(0.05);

    let timestep = 0.005;
    for _ in 0..12_000 {
        filter
            .step(
                true_offset,
                still_accelerometer(level),
                Some(dipping_magnetometer(level)),
                timestep,
            )
            .unwrap();
    }

    let learned = filter.gyroscope_bias();
    for axis in 0..3 {
        assert!(
            (learned[axis] - true_offset[axis]).abs() < 5e-3,
            "axis {axis}: learned {} against {}",
            learned[axis],
            true_offset[axis]
        );
    }
    assert!(angle_between(filter.orientation(), level) < 0.01);
}

#[test]
fn madgwick_zero_bias_gain_leaves_the_offset_alone() {
    let level = SO3::<f64>::identity();
    let true_offset = Vector::new([0.02, -0.015, 0.01]);
    let mut filter = MadgwickFilter::new(level)
        .with_correction_gain(0.3)
        .with_bias_gain(0.0);

    let timestep = 0.005;
    for _ in 0..12_000 {
        filter
            .step(
                true_offset,
                still_accelerometer(level),
                Some(dipping_magnetometer(level)),
                timestep,
            )
            .unwrap();
    }

    assert_eq!(filter.gyroscope_bias(), Vector3D::zeros());
}

#[test]
fn madgwick_walks_at_the_same_rate_however_wrong_it_is() {
    let level = SO3::<f64>::identity();
    let correction_gain = 0.2;
    let timestep = 0.005;
    let steps = 100;

    let closed = |start: SO3<f64>| {
        let mut filter = MadgwickFilter::new(start)
            .with_correction_gain(correction_gain)
            .with_bias_gain(0.0);
        for _ in 0..steps {
            filter
                .step(
                    Vector::zeros(),
                    still_accelerometer(level),
                    Some(dipping_magnetometer(level)),
                    timestep,
                )
                .unwrap();
        }
        angle_between(start, level) - angle_between(filter.orientation(), level)
    };

    let near = closed(SO3::exp(Vector::new([0.3, 0.0, 0.0])));
    let far = closed(SO3::exp(Vector::new([0.9, 0.0, 0.0])));
    let walked = correction_gain * timestep * f64::from(steps);

    // Neither quite reaches the nominal walk, because the correction is the sum of two sensors'
    // pulls and its direction is not exactly the shortest way home. What matters is that the two
    // starts, three times apart, close the gap by nearly the same amount.
    assert!((near - walked).abs() < 0.15 * walked, "near closed {near}");
    assert!((far - walked).abs() < 0.15 * walked, "far closed {far}");
    assert!((near - far).abs() < 0.1 * walked);
}

#[test]
fn madgwick_higher_correction_gain_pulls_harder() {
    let level = SO3::<f64>::identity();
    let wrong_start = SO3::exp(Vector::new([0.3, 0.0, 0.0]));
    let mut gentle = MadgwickFilter::new(wrong_start)
        .with_correction_gain(0.05)
        .with_bias_gain(0.0);
    let mut firm = MadgwickFilter::new(wrong_start)
        .with_correction_gain(0.5)
        .with_bias_gain(0.0);

    let not_turning = Vector::zeros();
    let timestep = 0.005;
    for _ in 0..200 {
        for filter in [&mut gentle, &mut firm] {
            filter
                .step(
                    not_turning,
                    still_accelerometer(level),
                    Some(dipping_magnetometer(level)),
                    timestep,
                )
                .unwrap();
        }
    }

    assert!(angle_between(firm.orientation(), level) < angle_between(gentle.orientation(), level));
}

#[test]
fn madgwick_matches_a_hand_stepped_loop() {
    let level = SO3::<f64>::identity();
    let start = SO3::exp(Vector::new([0.1, -0.2, 0.05]));
    let start_offset = Vector::new([0.01, 0.02, -0.03]);
    let correction_gain = 0.4;
    let bias_gain = 0.05;
    let turn_rate = Vector::new([0.1, 0.2, 0.3]);
    let push = still_accelerometer(level);
    let field = dipping_magnetometer(level);
    let timestep = 0.01;

    let mut filter = MadgwickFilter::new(start)
        .with_correction_gain(correction_gain)
        .with_bias_gain(bias_gain);
    filter.set_gyroscope_bias(start_offset);
    filter.step(turn_rate, push, Some(field), timestep).unwrap();

    let direction = hand_correction(start, push, field).normalized();
    let offset = start_offset - direction * bias_gain * timestep;
    let corrected_rate = turn_rate - offset + direction * correction_gain;
    let orientation = ExponentialMap::attitude_step(start, corrected_rate, timestep).normalized();

    assert!(angle_between(filter.orientation(), orientation) < 1e-15);
    for axis in 0..3 {
        assert!((filter.gyroscope_bias()[axis] - offset[axis]).abs() < 1e-15);
    }
}

// ---- single precision --------------------------------------------------------

#[test]
fn mahony_single_precision_runs_and_stays_finite() {
    let mut filter = MahonyFilter::new(SO3::<f32>::identity());
    let mut truth = SO3::<f32>::identity();

    let turn_rate = Vector::new([0.7_f32, -0.5, 1.1]);
    let timestep = 0.001_f32;
    for step in 0..100_000 {
        filter
            .step(
                turn_rate,
                still_accelerometer_f32(truth),
                Some(dipping_magnetometer_f32(truth)),
                timestep,
            )
            .unwrap();
        truth = ExponentialMap::attitude_step(truth, turn_rate, timestep).normalized();

        if step % 100 == 0 {
            assert!(filter.orientation().quaternion().as_array()[0].is_finite());
            assert!((filter.orientation().quaternion().norm() - 1.0).abs() < 1e-5);
        }
    }
}

#[test]
fn madgwick_single_precision_runs_and_stays_finite() {
    let mut filter = MadgwickFilter::new(SO3::<f32>::identity());
    let mut truth = SO3::<f32>::identity();

    let turn_rate = Vector::new([0.7_f32, -0.5, 1.1]);
    let timestep = 0.001_f32;
    for step in 0..100_000 {
        filter
            .step(
                turn_rate,
                still_accelerometer_f32(truth),
                Some(dipping_magnetometer_f32(truth)),
                timestep,
            )
            .unwrap();
        truth = ExponentialMap::attitude_step(truth, turn_rate, timestep).normalized();

        if step % 100 == 0 {
            assert!(filter.orientation().quaternion().as_array()[0].is_finite());
            assert!((filter.orientation().quaternion().norm() - 1.0).abs() < 1e-5);
        }
    }
}

#[test]
fn mahony_single_precision_converges_from_a_wrong_start() {
    let level = SO3::<f32>::identity();
    let mut filter = MahonyFilter::new(SO3::exp(Vector::new([0.6_f32, -0.4, 0.9])));

    let not_turning = Vector::zeros();
    let timestep = 0.005_f32;
    for _ in 0..24_000 {
        filter
            .step(
                not_turning,
                still_accelerometer_f32(level),
                Some(dipping_magnetometer_f32(level)),
                timestep,
            )
            .unwrap();
    }

    assert!(angle_between_f32(filter.orientation(), level) < 2e-2);
}

#[test]
fn madgwick_single_precision_converges_from_a_wrong_start() {
    let level = SO3::<f32>::identity();
    let mut filter =
        MadgwickFilter::new(SO3::exp(Vector::new([0.6_f32, -0.4, 0.9]))).with_correction_gain(0.3);

    let not_turning = Vector::zeros();
    let timestep = 0.005_f32;
    for _ in 0..4000 {
        filter
            .step(
                not_turning,
                still_accelerometer_f32(level),
                Some(dipping_magnetometer_f32(level)),
                timestep,
            )
            .unwrap();
    }

    assert!(angle_between_f32(filter.orientation(), level) < 2e-2);
}

#[test]
fn mahony_single_precision_round_trips_exp_and_log() {
    let mut filter = MahonyFilter::new(SO3::<f32>::identity());
    let mut truth = SO3::<f32>::identity();

    let turn_rate = Vector::new([0.7_f32, -0.5, 1.1]);
    let timestep = 0.001_f32;
    for _ in 0..100_000 {
        filter
            .step(
                turn_rate,
                still_accelerometer_f32(truth),
                Some(dipping_magnetometer_f32(truth)),
                timestep,
            )
            .unwrap();
        truth = ExponentialMap::attitude_step(truth, turn_rate, timestep).normalized();
    }

    let turned = filter.orientation();
    assert!(angle_between_f32(SO3::exp(turned.log()), turned) < 1e-5);
}

#[test]
fn madgwick_single_precision_round_trips_exp_and_log() {
    let mut filter = MadgwickFilter::new(SO3::<f32>::identity());
    let mut truth = SO3::<f32>::identity();

    let turn_rate = Vector::new([0.7_f32, -0.5, 1.1]);
    let timestep = 0.001_f32;
    for _ in 0..100_000 {
        filter
            .step(
                turn_rate,
                still_accelerometer_f32(truth),
                Some(dipping_magnetometer_f32(truth)),
                timestep,
            )
            .unwrap();
        truth = ExponentialMap::attitude_step(truth, turn_rate, timestep).normalized();
    }

    let turned = filter.orientation();
    assert!(angle_between_f32(SO3::exp(turned.log()), turned) < 1e-5);
}
