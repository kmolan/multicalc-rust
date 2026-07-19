//! Follow-the-Gap tests: the clear-scan case, gap selection and goal bias, the chassis-width gate,
//! dropped beams, speed scaling, the blocked stop, configuration validation, and an f32 run.

use core::f64::consts::PI;

use multicalc::control::FollowTheGap;
use multicalc::error::ControlError;

// 31 beams over 120°, 4 m range, a 0.5 m chassis, a 0.5 m gap threshold, 0.4 m/s cruise. Beam 15
// is the exact centre, and the beams are (2π/3)/30 = 0.06981… rad apart.
fn follower() -> FollowTheGap<31, f64> {
    FollowTheGap::try_new(2.0 * PI / 3.0, 4.0, 0.5, 0.5, 0.4).unwrap()
}

#[test]
fn clear_scan_drives_straight() {
    let plan = follower().plan(&[4.0; 31], 0.0).unwrap();
    assert!(!plan.is_blocked());
    assert!(plan.heading().abs() < 1e-12);
    assert!((plan.body_twist().linear() - 0.4).abs() < 1e-12);
    assert!(plan.body_twist().angular().abs() < 1e-12);
    assert_eq!(plan.gap_start(), 0);
    assert_eq!(plan.gap_end(), 30);
    assert!((plan.minimum_clearance() - 4.0).abs() < 1e-12);
}

#[test]
fn obstacle_on_the_right_steers_left() {
    let mut ranges = [4.0; 31];
    for range in ranges.iter_mut().take(16) {
        *range = 0.4;
    }
    let plan = follower().plan(&ranges, 0.0).unwrap();
    assert!(!plan.is_blocked());
    assert!(plan.heading() > 0.0);
    assert!(plan.body_twist().angular() > 0.0);
    assert_eq!(plan.gap_start(), 16);
    // The run is open-ended at the far end, so the aim is inset only at beam 16, by
    // atan(0.25 / 0.4) = 0.5586 rad from that beam's bearing of 0.0698 rad.
    assert!(
        (plan.heading() - 0.62841).abs() < 1e-5,
        "heading {}",
        plan.heading()
    );
}

#[test]
fn obstacle_on_the_left_steers_right() {
    let mut ranges = [4.0; 31];
    for range in ranges.iter_mut().skip(15) {
        *range = 0.4;
    }
    let plan = follower().plan(&ranges, 0.0).unwrap();
    assert!(!plan.is_blocked());
    assert!(plan.heading() < 0.0);
    assert_eq!(plan.gap_end(), 14);
}

#[test]
fn fully_blocked_scan_stops() {
    let plan = follower().plan(&[0.2; 31], 0.0).unwrap();
    assert!(plan.is_blocked());
    assert_eq!(plan.body_twist().linear(), 0.0);
    assert_eq!(plan.body_twist().angular(), 0.0);
    assert_eq!(plan.heading(), 0.0);
    assert!((plan.minimum_clearance() - 0.2).abs() < 1e-12);
}

#[test]
fn gap_narrower_than_the_chassis_is_rejected() {
    // Both bands below are far wider in angle than the robot needs in open space. It is the 0.4 m
    // range of the returns bounding them that makes the chord short, which is what a width in
    // metres captures and an angular threshold would not.
    let follower = follower();

    // Free 6..=23, bounded by beams 5 and 24: √(2 · 0.4² · (1 − cos 1.3265)) ≈ 0.4925 m.
    let mut narrow = [0.4; 31];
    for range in narrow.iter_mut().take(24).skip(6) {
        *range = 4.0;
    }
    let plan = follower.plan(&narrow, 0.0).unwrap();
    assert!(
        plan.is_blocked(),
        "a 0.4925 m gap must not admit a 0.5 m chassis"
    );

    // Free 6..=24, bounded by beams 5 and 25: one beam wider, ≈ 0.5142 m, and passable.
    let mut wide = [0.4; 31];
    for range in wide.iter_mut().take(25).skip(6) {
        *range = 4.0;
    }
    let plan = follower.plan(&wide, 0.0).unwrap();
    assert!(!plan.is_blocked());
    assert_eq!(plan.gap_start(), 6);
    assert_eq!(plan.gap_end(), 24);
}

#[test]
fn dropped_beams_read_as_free_space() {
    let mut ranges = [4.0; 31];
    ranges[3] = f64::NAN;
    ranges[7] = f64::INFINITY;
    ranges[11] = -1.0;
    ranges[12] = 0.0;
    let plan = follower().plan(&ranges, 0.0).unwrap();
    assert!(!plan.is_blocked());
    assert!(plan.heading().abs() < 1e-12);
    assert_eq!(plan.gap_start(), 0);
    assert_eq!(plan.gap_end(), 30);
}

#[test]
fn goal_bias_selects_the_gap_toward_the_goal() {
    // Blocking the middle leaves two gaps. Both are open-ended at the scan edge and inset by the
    // same atan(0.25 / 0.3) at their inner edge, so their usable spans are identical and the goal
    // bias is the only thing separating them.
    let mut ranges = [4.0; 31];
    for range in ranges.iter_mut().take(18).skip(13) {
        *range = 0.3;
    }
    let follower = follower();

    let toward_left = follower.plan(&ranges, 0.6).unwrap();
    assert!(toward_left.heading() > 0.0);
    assert_eq!(toward_left.gap_start(), 18);

    let toward_right = follower.plan(&ranges, -0.6).unwrap();
    assert!(toward_right.heading() < 0.0);
    assert_eq!(toward_right.gap_end(), 12);

    // With the goal dead ahead the two gaps score identically and have equal |aim|, so neither the
    // score nor the tie-break separates them and the lower index wins. The point is that a
    // symmetric scan resolves deterministically, not that it resolves in a meaningful direction.
    let symmetric = follower.plan(&ranges, 0.0).unwrap();
    assert_eq!(symmetric.gap_end(), 12);
}

#[test]
fn speed_scales_with_frontal_clearance() {
    // The frontal half-angle defaults to field_of_view / 4 = π/6 ≈ 0.5236 rad, covering beams
    // 8..=22. Every case stays above the 0.5 m gap threshold, so this measures speed scaling and
    // not the blocked path.
    let follower = follower().with_speed_scaling(1.0, 3.0).unwrap();
    for (frontal, expected) in [(1.0, 0.0), (2.0, 0.2), (3.0, 0.4)] {
        let mut ranges = [4.0; 31];
        for range in ranges.iter_mut().take(23).skip(8) {
            *range = frontal;
        }
        let plan = follower.plan(&ranges, 0.0).unwrap();
        assert!(!plan.is_blocked(), "frontal {frontal} m must not block");
        assert!(
            (plan.body_twist().linear() - expected).abs() < 1e-12,
            "frontal {frontal} m gave {}",
            plan.body_twist().linear()
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
        follower().with_turn_gain(0.0).err(),
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
fn non_finite_goal_bearing_is_an_error() {
    assert_eq!(
        follower().plan(&[4.0; 31], f64::NAN).err(),
        Some(ControlError::NonFinite)
    );
    assert_eq!(
        follower().plan(&[4.0; 31], f64::INFINITY).err(),
        Some(ControlError::NonFinite)
    );
}

#[test]
fn beam_bearing_spans_the_field_of_view() {
    let follower = follower();
    assert!((follower.beam_bearing(0).unwrap() + PI / 3.0).abs() < 1e-12);
    assert!(follower.beam_bearing(15).unwrap().abs() < 1e-12);
    assert!((follower.beam_bearing(30).unwrap() - PI / 3.0).abs() < 1e-12);
    assert!(follower.beam_bearing(31).is_none());
}

#[test]
fn clear_scan_runs_at_f32() {
    let follower: FollowTheGap<31, f32> =
        FollowTheGap::try_new(2.0 * core::f32::consts::PI / 3.0, 4.0, 0.5, 0.5, 0.4).unwrap();
    let plan = follower.plan(&[4.0_f32; 31], 0.0).unwrap();
    assert!(plan.heading().abs() < 1e-6);
    assert!((plan.body_twist().linear() - 0.4).abs() < 1e-6);
}
