//! Where the beams of a scan point, and the readings a scan will believe.

use multicalc::control::FollowTheGap;
use multicalc::error::MappingError;
use multicalc::mapping::ScanGeometry;

const QUARTER_TURN: f64 = core::f64::consts::FRAC_PI_2;
const WIDE_ARC: f64 = 2.0 * core::f64::consts::PI / 3.0;

fn scan<const NUM_BEAMS: usize>(field_of_view: f64) -> ScanGeometry<NUM_BEAMS, f64> {
    ScanGeometry::try_new(field_of_view, 4.0).unwrap()
}

#[test]
fn a_scan_needs_at_least_two_beams() {
    assert_eq!(
        ScanGeometry::<0, f64>::try_new(QUARTER_TURN, 4.0).unwrap_err(),
        MappingError::TooFewBeams
    );
    assert_eq!(
        ScanGeometry::<1, f64>::try_new(QUARTER_TURN, 4.0).unwrap_err(),
        MappingError::TooFewBeams
    );
    assert!(ScanGeometry::<2, f64>::try_new(QUARTER_TURN, 4.0).is_ok());
}

#[test]
fn an_unusable_arc_or_range_is_rejected() {
    assert_eq!(
        ScanGeometry::<8, f64>::try_new(0.0, 4.0).unwrap_err(),
        MappingError::InvalidFieldOfView
    );
    assert_eq!(
        ScanGeometry::<8, f64>::try_new(-1.0, 4.0).unwrap_err(),
        MappingError::InvalidFieldOfView
    );
    assert_eq!(
        ScanGeometry::<8, f64>::try_new(7.0, 4.0).unwrap_err(),
        MappingError::InvalidFieldOfView,
        "more than a whole turn is not an arc"
    );
    assert!(
        ScanGeometry::<8, f64>::try_new(core::f64::consts::TAU, 4.0).is_ok(),
        "a whole turn is allowed"
    );
    assert_eq!(
        ScanGeometry::<8, f64>::try_new(QUARTER_TURN, 0.0).unwrap_err(),
        MappingError::NonPositiveRange
    );
    assert_eq!(
        ScanGeometry::<8, f64>::try_new(f64::NAN, 4.0).unwrap_err(),
        MappingError::NonFinite
    );
    assert_eq!(
        ScanGeometry::<8, f64>::try_new(QUARTER_TURN, f64::INFINITY).unwrap_err(),
        MappingError::NonFinite
    );
}

#[test]
fn beams_span_the_arc_from_right_to_left() {
    let geometry = scan::<61>(WIDE_ARC);
    let half_the_arc = WIDE_ARC / 2.0;
    assert!((geometry.beam_angle(0).unwrap() + half_the_arc).abs() < 1e-12);
    assert!(geometry.beam_angle(30).unwrap().abs() < 1e-12);
    assert!((geometry.beam_angle(60).unwrap() - half_the_arc).abs() < 1e-12);
    assert_eq!(geometry.beam_angle(61), None);
    assert_eq!(geometry.field_of_view(), WIDE_ARC);
    assert_eq!(geometry.maximum_range(), 4.0);
    assert_eq!(geometry.num_beams(), 61);
}

#[test]
fn the_increment_is_the_step_between_beams() {
    let geometry = scan::<5>(QUARTER_TURN);
    assert!((geometry.angle_increment() - QUARTER_TURN / 4.0).abs() < 1e-12);
    for beam in 0..4 {
        let here = geometry.beam_angle(beam).unwrap();
        let next = geometry.beam_angle(beam + 1).unwrap();
        assert!(
            (next - here - geometry.angle_increment()).abs() < 1e-12,
            "beam {beam}"
        );
    }
    // Two beams sit one whole arc apart.
    assert!((scan::<2>(QUARTER_TURN).angle_increment() - QUARTER_TURN).abs() < 1e-12);
}

/// The property the shared beam formula exists to guarantee: a scan and the steering worked out
/// from it must number their beams identically, or a robot turns the wrong way.
#[test]
fn a_scan_and_the_gap_follower_agree_beam_for_beam() {
    fn check<const NUM_BEAMS: usize>(field_of_view: f64) {
        let maximum_range = 4.0;
        let geometry =
            ScanGeometry::<NUM_BEAMS, f64>::try_new(field_of_view, maximum_range).unwrap();
        let chassis_width = 0.5;
        let free_range_threshold = 0.6;
        let cruise_speed = 0.4;
        let follower = FollowTheGap::<NUM_BEAMS, f64>::try_new(
            field_of_view,
            maximum_range,
            chassis_width,
            free_range_threshold,
            cruise_speed,
        )
        .unwrap();
        for beam in 0..NUM_BEAMS {
            assert_eq!(
                geometry.beam_angle(beam),
                follower.beam_angle(beam),
                "beam {beam} of {NUM_BEAMS} across {field_of_view}"
            );
        }
        assert_eq!(geometry.beam_angle(NUM_BEAMS), None);
        assert_eq!(follower.beam_angle(NUM_BEAMS), None);
    }
    check::<2>(1.0);
    check::<5>(QUARTER_TURN);
    check::<31>(WIDE_ARC);
    check::<61>(WIDE_ARC);
    check::<180>(core::f64::consts::TAU);
}

#[test]
fn every_beam_direction_leads_back_to_its_own_beam() {
    let geometry = scan::<61>(WIDE_ARC);
    for beam in 0..61 {
        let angle = geometry.beam_angle(beam).unwrap();
        assert_eq!(
            geometry.nearest_beam_index(angle),
            Some(beam),
            "beam {beam}"
        );
    }
}

#[test]
fn a_direction_between_two_beams_goes_to_the_nearer_one() {
    let geometry = scan::<3>(QUARTER_TURN);
    let halfway = geometry.beam_angle(0).unwrap() + geometry.angle_increment() / 2.0;
    assert_eq!(geometry.nearest_beam_index(halfway - 1e-6), Some(0));
    assert_eq!(geometry.nearest_beam_index(halfway + 1e-6), Some(1));
}

#[test]
fn a_direction_outside_the_arc_has_no_beam() {
    let geometry = scan::<3>(QUARTER_TURN);
    let half_the_arc = QUARTER_TURN / 2.0;
    assert_eq!(geometry.nearest_beam_index(-half_the_arc), Some(0));
    assert_eq!(geometry.nearest_beam_index(half_the_arc), Some(2));
    assert_eq!(geometry.nearest_beam_index(-half_the_arc - 1e-9), None);
    assert_eq!(geometry.nearest_beam_index(half_the_arc + 1e-9), None);
    assert_eq!(geometry.nearest_beam_index(f64::NAN), None);
    assert_eq!(geometry.nearest_beam_index(f64::INFINITY), None);
}

#[test]
fn a_scan_sees_from_zero_until_it_is_told_otherwise() {
    let geometry = scan::<5>(QUARTER_TURN);
    assert_eq!(geometry.minimum_range(), 0.0);
    // A beam that started on a wall reads zero, which a sensor with no blind spot can produce.
    assert!(geometry.range_is_valid(0.0));
}

#[test]
fn an_unusable_blind_spot_is_rejected() {
    let geometry = scan::<5>(QUARTER_TURN);
    assert_eq!(
        geometry.with_minimum_range(-0.1).unwrap_err(),
        MappingError::InvalidRangeLimits
    );
    assert_eq!(
        geometry.with_minimum_range(4.0).unwrap_err(),
        MappingError::InvalidRangeLimits,
        "a blind spot reaching the range leaves nothing the sensor can see"
    );
    assert_eq!(
        geometry.with_minimum_range(f64::NAN).unwrap_err(),
        MappingError::NonFinite
    );
    assert_eq!(
        geometry.with_minimum_range(0.0).unwrap().minimum_range(),
        0.0
    );
    assert!(geometry.with_minimum_range(3.999).is_ok());
}

#[test]
fn a_reading_counts_only_between_the_two_limits() {
    let geometry = scan::<5>(QUARTER_TURN).with_minimum_range(0.12).unwrap();
    assert!(
        geometry.range_is_valid(0.12),
        "the closest it can see still counts"
    );
    assert!(geometry.range_is_valid(2.0));
    assert!(
        geometry.range_is_valid(4.0),
        "the furthest it can see still counts"
    );
    assert!(!geometry.range_is_valid(0.119));
    assert!(!geometry.range_is_valid(4.001));
    assert!(!geometry.range_is_valid(-1.0));
    assert!(
        !geometry.range_is_valid(f64::INFINITY),
        "a beam that met nothing is not a distance"
    );
    assert!(!geometry.range_is_valid(f64::NAN));
}

#[test]
fn a_scan_answers_the_same_way_at_f32() {
    let field_of_view = core::f32::consts::FRAC_PI_2;
    let geometry: ScanGeometry<5, f32> = ScanGeometry::try_new(field_of_view, 4.0).unwrap();
    assert_eq!(geometry.num_beams(), 5);
    assert!((geometry.angle_increment() - field_of_view / 4.0).abs() < 1e-6);
    for beam in 0..5 {
        let angle = geometry.beam_angle(beam).unwrap();
        assert_eq!(
            geometry.nearest_beam_index(angle),
            Some(beam),
            "beam {beam}"
        );
    }
}
