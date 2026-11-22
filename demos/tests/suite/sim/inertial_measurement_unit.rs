use multicalc::scalar::Numeric;
use multicalc_demos::sim::inertial_measurement_unit::InertialMeasurementUnit;
use rand::SeedableRng;
use rand_pcg::Pcg32;
use std::f64::consts::PI;

#[test]
fn zero_noise_and_offset_returns_the_truth() {
    let mut rng = Pcg32::seed_from_u64(1);
    let unit = InertialMeasurementUnit::new(0.0, 0.0, 0.0);
    // A heading past π is folded back into range; the turn rate passes through unchanged.
    let reading = unit.read(4.0, 0.2, &mut rng);
    assert!(
        (reading.heading - (4.0).wrap_to_pi()).abs() < 1e-12,
        "heading: {}",
        reading.heading
    );
    assert_eq!(reading.yaw_rate, 0.2);
}

#[test]
fn a_heading_near_pi_plus_offset_wraps_past_pi() {
    let mut rng = Pcg32::seed_from_u64(2);
    let unit = InertialMeasurementUnit::new(0.0, 0.0, 0.1);
    // Just under π plus the offset lands past π, so it should fold to the negative side rather
    // than report a value above π.
    let reading = unit.read(PI - 0.05, 0.0, &mut rng);
    assert!(
        reading.heading < 0.0,
        "should wrap to negative: {}",
        reading.heading
    );
    assert!(
        reading.heading > -PI && reading.heading <= PI,
        "out of range: {}",
        reading.heading
    );
    assert!(
        (reading.heading - (PI - 0.05 + 0.1).wrap_to_pi()).abs() < 1e-12,
        "heading: {}",
        reading.heading
    );
}

#[test]
fn a_fixed_seed_reproduces_the_reading() {
    let unit = InertialMeasurementUnit::new(0.02, 0.01, 0.01);
    let mut first = Pcg32::seed_from_u64(7);
    let mut second = Pcg32::seed_from_u64(7);
    assert_eq!(
        unit.read(0.3, -0.4, &mut first),
        unit.read(0.3, -0.4, &mut second)
    );
}
