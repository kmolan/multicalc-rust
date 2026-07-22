//! Follow-the-Gap tests: the clear-scan case, gap selection and goal bias, the chassis-width gate,
//! dropped beams, speed scaling, the blocked stop, configuration validation, and an f32 run.

use core::f64::consts::PI;

use multicalc::control::FollowTheGap;
use multicalc::error::ControlError;

// 31 beams over 120°, 4 m range, a 0.5 m robot, a 0.5 m open-space threshold, 0.4 m/s cruise. Beam
// 15 is the exact centre, and neighbouring beams are (2π/3)/30 = 0.06981… rad apart.
fn follower() -> FollowTheGap<31, f64> {
    FollowTheGap::try_new(2.0 * PI / 3.0, 4.0, 0.5, 0.5, 0.4).unwrap()
}

#[test]
fn clear_scan_drives_straight() {
    let output = follower().compute(&[4.0; 31], 0.0).unwrap();
    assert!(!output.is_blocked());
    assert!(output.heading().abs() < 1e-12);
    assert!((output.body_twist().linear() - 0.4).abs() < 1e-12);
    assert!(output.body_twist().angular().abs() < 1e-12);
    assert_eq!(output.gap_start_index(), 0);
    assert_eq!(output.gap_end_index(), 30);
    assert!((output.minimum_clearance() - 4.0).abs() < 1e-12);
}

#[test]
fn obstacle_on_the_right_steers_left() {
    let mut ranges = [4.0; 31];
    for range in ranges.iter_mut().take(16) {
        *range = 0.4;
    }
    let output = follower().compute(&ranges, 0.0).unwrap();
    assert!(!output.is_blocked());
    assert!(output.heading() > 0.0);
    assert!(output.body_twist().angular() > 0.0);
    assert_eq!(output.gap_start_index(), 16);
    // The gap runs off the edge of the scan, so it has an obstacle only on the near side. The aim
    // is pulled in from that one edge (beam 16, at 0.0698 rad) by atan(0.25 / 0.4) = 0.5586 rad.
    assert!(
        (output.heading() - 0.62841).abs() < 1e-5,
        "heading {}",
        output.heading()
    );
}

#[test]
fn obstacle_on_the_left_steers_right() {
    let mut ranges = [4.0; 31];
    for range in ranges.iter_mut().skip(15) {
        *range = 0.4;
    }
    let output = follower().compute(&ranges, 0.0).unwrap();
    assert!(!output.is_blocked());
    assert!(output.heading() < 0.0);
    assert_eq!(output.gap_end_index(), 14);
}

#[test]
fn fully_blocked_scan_stops() {
    let output = follower().compute(&[0.2; 31], 0.0).unwrap();
    assert!(output.is_blocked());
    assert_eq!(output.body_twist().linear(), 0.0);
    assert_eq!(output.body_twist().angular(), 0.0);
    assert_eq!(output.heading(), 0.0);
    assert!((output.minimum_clearance() - 0.2).abs() < 1e-12);
}

#[test]
fn gap_narrower_than_the_chassis_is_rejected() {
    // Both bands below span a wide angle — wider than the robot needs in open space. But the
    // obstacles on their edges are only 0.4 m away, so the straight-line distance between them is
    // short. A width in metres catches that; an angle-only test would not.
    let follower = follower();

    // Open beams 6..=23, with obstacles at beams 5 and 24: √(2 · 0.4² · (1 − cos 1.3265)) ≈ 0.4925 m.
    let mut narrow = [0.4; 31];
    for range in narrow.iter_mut().take(24).skip(6) {
        *range = 4.0;
    }
    let output = follower.compute(&narrow, 0.0).unwrap();
    assert!(
        output.is_blocked(),
        "a 0.4925 m gap must not admit a 0.5 m chassis"
    );

    // Open beams 6..=24, with obstacles at beams 5 and 25: one beam wider, ≈ 0.5142 m, and passable.
    let mut wide = [0.4; 31];
    for range in wide.iter_mut().take(25).skip(6) {
        *range = 4.0;
    }
    let output = follower.compute(&wide, 0.0).unwrap();
    assert!(!output.is_blocked());
    assert_eq!(output.gap_start_index(), 6);
    assert_eq!(output.gap_end_index(), 24);
}

#[test]
fn dropped_beams_read_as_free_space() {
    let mut ranges = [4.0; 31];
    ranges[3] = f64::NAN;
    ranges[7] = f64::INFINITY;
    ranges[11] = -1.0;
    ranges[12] = 0.0;
    let output = follower().compute(&ranges, 0.0).unwrap();
    assert!(!output.is_blocked());
    assert!(output.heading().abs() < 1e-12);
    assert_eq!(output.gap_start_index(), 0);
    assert_eq!(output.gap_end_index(), 30);
}

#[test]
fn goal_bias_selects_the_gap_toward_the_goal() {
    // Blocking the middle leaves two gaps. Each runs off its scan edge and is pulled in by the same
    // atan(0.25 / 0.3) at its inner edge, so both end up the same width. The goal bias is then the
    // only thing separating them.
    let mut ranges = [4.0; 31];
    for range in ranges.iter_mut().take(18).skip(13) {
        *range = 0.3;
    }
    let follower = follower();

    let toward_left = follower.compute(&ranges, 0.6).unwrap();
    assert!(toward_left.heading() > 0.0);
    assert_eq!(toward_left.gap_start_index(), 18);

    let toward_right = follower.compute(&ranges, -0.6).unwrap();
    assert!(toward_right.heading() < 0.0);
    assert_eq!(toward_right.gap_end_index(), 12);

    // With the goal straight ahead, both gaps score the same and aim the same distance off-centre,
    // so nothing separates them and the earlier beam wins. The point is that a symmetric scan
    // resolves the same way every time, not that it picks a meaningful side.
    let symmetric = follower.compute(&ranges, 0.0).unwrap();
    assert_eq!(symmetric.gap_end_index(), 12);
}

#[test]
fn speed_scales_with_frontal_clearance() {
    // The "ahead" half-angle defaults to field_of_view / 4 = π/6 ≈ 0.5236 rad, covering beams
    // 8..=22. Every case stays above the 0.5 m open-space threshold, so this checks speed scaling,
    // not the blocked path.
    let follower = follower().with_speed_scaling(1.0, 3.0).unwrap();
    for (frontal, expected) in [(1.0, 0.0), (2.0, 0.2), (3.0, 0.4)] {
        let mut ranges = [4.0; 31];
        for range in ranges.iter_mut().take(23).skip(8) {
            *range = frontal;
        }
        let output = follower.compute(&ranges, 0.0).unwrap();
        assert!(!output.is_blocked(), "frontal {frontal} m must not block");
        assert!(
            (output.body_twist().linear() - expected).abs() < 1e-12,
            "frontal {frontal} m gave {}",
            output.body_twist().linear()
        );
    }
}

#[test]
fn invalid_configuration_is_rejected() {
    assert_eq!(
        FollowTheGap::<1, f64>::try_new(2.0, 4.0, 0.5, 0.5, 0.4).err(),
        Some(ControlError::InvalidBeamCount)
    );
    assert_eq!(
        FollowTheGap::<31, f64>::try_new(f64::NAN, 4.0, 0.5, 0.5, 0.4).err(),
        Some(ControlError::NonFinite)
    );
    assert_eq!(
        FollowTheGap::<31, f64>::try_new(0.0, 4.0, 0.5, 0.5, 0.4).err(),
        Some(ControlError::InvalidFieldOfView)
    );
    // 7.0 > 2π ≈ 6.2832, so this hits the upper bound of the field-of-view check.
    assert_eq!(
        FollowTheGap::<31, f64>::try_new(7.0, 4.0, 0.5, 0.5, 0.4).err(),
        Some(ControlError::InvalidFieldOfView)
    );
    assert_eq!(
        FollowTheGap::<31, f64>::try_new(2.0, 0.0, 0.5, 0.5, 0.4).err(),
        Some(ControlError::NonPositiveRange)
    );
    assert_eq!(
        FollowTheGap::<31, f64>::try_new(2.0, 4.0, 0.5, 5.0, 0.4).err(),
        Some(ControlError::NonPositiveRange)
    );
    assert_eq!(
        FollowTheGap::<31, f64>::try_new(2.0, 4.0, 0.0, 0.5, 0.4).err(),
        Some(ControlError::NonPositiveChassisWidth)
    );
    assert_eq!(
        FollowTheGap::<31, f64>::try_new(2.0, 4.0, -0.1, 0.5, 0.4).err(),
        Some(ControlError::NonPositiveChassisWidth)
    );
    // Half of 8.0 reaches the 4.0 m maximum range, leaving no room to scale speed.
    assert_eq!(
        FollowTheGap::<31, f64>::try_new(2.0, 4.0, 8.0, 0.5, 0.4).err(),
        Some(ControlError::NonPositiveChassisWidth)
    );
    assert_eq!(
        FollowTheGap::<31, f64>::try_new(2.0, 4.0, 0.5, 0.5, 0.0).err(),
        Some(ControlError::NonPositiveSpeed)
    );
    assert_eq!(
        follower().with_steering_gain(0.0).err(),
        Some(ControlError::NonPositiveSpeed)
    );
    assert_eq!(
        follower().with_goal_bias(-0.1).err(),
        Some(ControlError::NegativeGoalBias)
    );
    assert_eq!(
        follower().with_speed_scaling(2.0, 1.0).err(),
        Some(ControlError::InvalidSpeedScaling)
    );
    assert_eq!(
        follower().with_speed_scaling(-1.0, 3.0).err(),
        Some(ControlError::InvalidSpeedScaling)
    );
    assert_eq!(
        follower().with_frontal_half_angle(2.0).err(),
        Some(ControlError::InvalidFieldOfView)
    );
    assert_eq!(
        follower().with_frontal_half_angle(0.0).err(),
        Some(ControlError::InvalidFieldOfView)
    );
}

#[test]
fn non_finite_goal_angle_is_an_error() {
    assert_eq!(
        follower().compute(&[4.0; 31], f64::NAN).err(),
        Some(ControlError::NonFinite)
    );
    assert_eq!(
        follower().compute(&[4.0; 31], f64::INFINITY).err(),
        Some(ControlError::NonFinite)
    );
}

#[test]
fn beam_angle_spans_the_field_of_view() {
    let follower = follower();
    assert!((follower.beam_angle(0).unwrap() + PI / 3.0).abs() < 1e-12);
    assert!(follower.beam_angle(15).unwrap().abs() < 1e-12);
    assert!((follower.beam_angle(30).unwrap() - PI / 3.0).abs() < 1e-12);
    assert!(follower.beam_angle(31).is_none());
}

#[test]
fn clear_scan_runs_at_f32() {
    let follower: FollowTheGap<31, f32> =
        FollowTheGap::try_new(2.0 * core::f32::consts::PI / 3.0, 4.0, 0.5, 0.5, 0.4).unwrap();
    let output = follower.compute(&[4.0_f32; 31], 0.0).unwrap();
    assert!(output.heading().abs() < 1e-6);
    assert!((output.body_twist().linear() - 0.4).abs() < 1e-6);
}
