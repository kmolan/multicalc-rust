use multicalc_demos::sim::localization_obstacle_avoidance_2d::lap_driver_2d::LOCALIZE_CAP_TICKS;
use multicalc_demos::sim::localization_obstacle_avoidance_2d::{LapWorld, Phase};

#[test]
fn the_first_tick_is_localizing() {
    let mut world = LapWorld::new(1).unwrap();
    assert_eq!(world.step().phase, Phase::Localizing);
}

#[test]
fn it_converges_to_driving_within_the_cap() {
    let mut world = LapWorld::new(20260722).unwrap();
    let mut driving = false;
    for _ in 0..LOCALIZE_CAP_TICKS + 10 {
        if world.step().phase == Phase::Driving {
            driving = true;
            break;
        }
    }
    assert!(driving, "never reached the driving phase");
    assert!(
        world.metrics().localization_ticks <= LOCALIZE_CAP_TICKS,
        "localization took {} ticks",
        world.metrics().localization_ticks
    );
}

#[test]
fn a_fixed_seed_reproduces_the_run() {
    let run = |seed| {
        let mut world = LapWorld::new(seed).unwrap();
        let mut last = world.step();
        for _ in 0..4000 {
            last = world.step();
        }
        (last.pose.into_array(), world.metrics())
    };
    let (first_pose, first_metrics) = run(20260722);
    let (second_pose, second_metrics) = run(20260722);
    assert_eq!(first_pose, second_pose);
    assert_eq!(first_metrics.laps, second_metrics.laps);
    assert_eq!(first_metrics.contacts, second_metrics.contacts);
    assert_eq!(first_metrics.driving_ticks, second_metrics.driving_ticks);
}

#[test]
fn different_seeds_diverge() {
    let run = |seed| {
        let mut world = LapWorld::new(seed).unwrap();
        let mut last = world.step();
        for _ in 0..4000 {
            last = world.step();
        }
        last.pose.into_array()
    };
    assert_ne!(run(1), run(2));
}

#[test]
fn fusion_beats_dead_reckoning() {
    let mut world = LapWorld::new(20260722).unwrap();
    for _ in 0..80_000 {
        let _ = world.step();
    }
    let metrics = world.metrics();
    assert!(metrics.driving_ticks > 0, "never drove");
    assert!(metrics.gps_updates > 0, "no GPS updates");
    assert!(
        metrics.fused_position_rms_error() < metrics.dead_reckoned_position_rms_error(),
        "fused {} should beat dead-reckoned {}",
        metrics.fused_position_rms_error(),
        metrics.dead_reckoned_position_rms_error()
    );
}
