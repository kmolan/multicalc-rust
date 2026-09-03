//! The endpoint measurement model: its guards, that it ranks the true pose first, and that its
//! score is smoother in the pose than the beam model's.

use multicalc::error::EstimationError;
use multicalc::estimation::{
    BeamModel, InitialParticleCloud, LikelihoodFieldModel, MonteCarloLocalizer,
};
use multicalc::mapping::{
    DistanceField, DistanceTransformWorkspace, MutableOccupancyMap, OccupancyGrid, OccupancyMap,
    ScanGeometry,
};
use multicalc::{SE2, SO2, Vector2D};

const NUM_BEAMS: usize = 16;
const SEED: u64 = 20260830;

/// A 6 m square walled room at 20 cm cells, and the distance field over it.
fn walled_room() -> (OccupancyGrid<30, 30, 1>, DistanceField<30, 30>) {
    let mut room: OccupancyGrid<30, 30, 1> = OccupancyGrid::try_new(0.2, [0.0, 0.0]).unwrap();
    let walls = [[0.5, 0.5], [5.5, 0.5], [5.5, 5.5], [0.5, 5.5]];
    room.occupy_polyline(&walls, true);

    let mut workspace: DistanceTransformWorkspace<31> = DistanceTransformWorkspace::new();
    let field = DistanceField::try_build(&room, &mut workspace).unwrap();
    (room, field)
}

fn scan_geometry() -> ScanGeometry<NUM_BEAMS> {
    ScanGeometry::try_new(core::f64::consts::TAU * 0.9, 8.0).unwrap()
}

/// The ranges a perfect sensor at `pose` would read in `room`.
fn reading_at(room: &OccupancyGrid<30, 30, 1>, pose: [f64; 3]) -> [f64; NUM_BEAMS] {
    let scan = scan_geometry();
    let placed = SE2::from_parts(SO2::from_angle(pose[2]), Vector2D::new([pose[0], pose[1]]));
    room.cast_scan(placed, &scan)
}

fn localizer_at(hint: [f64; 3]) -> MonteCarloLocalizer<NUM_BEAMS> {
    let cloud = InitialParticleCloud {
        particle_count: 200,
        position_variance: 1e-8,
        heading_variance: 1e-8,
    };
    MonteCarloLocalizer::new(hint, cloud, BeamModel::default(), SEED).unwrap()
}

#[test]
fn rejects_a_non_positive_deviation() {
    let (room, field) = walled_room();
    let scan = scan_geometry();
    let ranges = reading_at(&room, [3.0, 3.0, 0.0]);
    let mut localizer = localizer_at([3.0, 3.0, 0.0]);

    for deviation in [0.0, -0.2, f64::NAN, f64::INFINITY] {
        let model = LikelihoodFieldModel {
            measurement_deviation: deviation,
            ..Default::default()
        };
        assert_eq!(
            localizer
                .update_against_field(&field, &scan, &ranges, model)
                .err(),
            Some(EstimationError::InvalidTuning),
            "deviation {deviation}"
        );
    }
}

#[test]
fn rejects_a_weight_outside_zero_to_one() {
    let (room, field) = walled_room();
    let scan = scan_geometry();
    let ranges = reading_at(&room, [3.0, 3.0, 0.0]);
    let mut localizer = localizer_at([3.0, 3.0, 0.0]);

    for weight in [-0.1, 1.1, f64::NAN] {
        let model = LikelihoodFieldModel {
            random_measurement_weight: weight,
            ..Default::default()
        };
        assert_eq!(
            localizer
                .update_against_field(&field, &scan, &ranges, model)
                .err(),
            Some(EstimationError::InvalidTuning),
            "weight {weight}"
        );
    }

    // The ends of the range are allowed.
    for weight in [0.0, 1.0] {
        let model = LikelihoodFieldModel {
            random_measurement_weight: weight,
            ..Default::default()
        };
        assert!(
            localizer
                .update_against_field(&field, &scan, &ranges, model)
                .is_ok()
        );
    }
}

/// The model's own per-pose log-score, so a test can compare two poses without a whole filter.
fn log_score(field: &DistanceField<30, 30>, ranges: &[f64; NUM_BEAMS], pose: [f64; 3]) -> f64 {
    let scan = scan_geometry();
    let model = LikelihoodFieldModel::<f64>::default();
    let deviation = model.measurement_deviation;
    let weight = model.random_measurement_weight;

    let mut score = 0.0;
    for (beam, &range) in ranges.iter().enumerate() {
        let Some(offset) = scan.beam_angle(beam) else {
            continue;
        };
        if !scan.range_is_valid(range) {
            continue;
        }
        let bearing = pose[2] + offset;
        let endpoint = [
            pose[0] + range * bearing.cos(),
            pose[1] + range * bearing.sin(),
        ];
        let distance = field.distance_at(endpoint).unwrap_or(f64::INFINITY);
        let hit = (-(distance * distance) / (2.0 * deviation * deviation)).exp();
        score += ((1.0 - weight) * hit + weight / scan.maximum_range()).ln();
    }
    score
}

#[test]
fn the_true_pose_outscores_a_displaced_one_f64() {
    let (room, field) = walled_room();
    let truth = [3.0, 3.0, 0.3];
    let ranges = reading_at(&room, truth);

    let at_the_truth = log_score(&field, &ranges, truth);
    for displacement in [0.5, -0.5] {
        let displaced = [truth[0] + displacement, truth[1], truth[2]];
        assert!(
            at_the_truth > log_score(&field, &ranges, displaced),
            "displacement {displacement}"
        );
        let displaced = [truth[0], truth[1] + displacement, truth[2]];
        assert!(
            at_the_truth > log_score(&field, &ranges, displaced),
            "displacement {displacement}"
        );
    }
}

#[test]
fn an_endpoint_off_the_field_scores_the_noise_floor_f64() {
    let (_, field) = walled_room();
    let scan = scan_geometry();

    // A pose well outside the map throws every endpoint off the field, so every beam contributes
    // the pure-noise term and nothing else.
    let far_away = [1000.0, 1000.0, 0.0];
    let ranges = [4.0; NUM_BEAMS];
    let model = LikelihoodFieldModel::<f64>::default();
    let per_beam = (model.random_measurement_weight / scan.maximum_range()).ln();

    let score = log_score(&field, &ranges, far_away);
    assert!((score - per_beam * NUM_BEAMS as f64).abs() < 1e-12);
}

#[test]
fn the_cloud_settles_onto_the_robot_f64() {
    let (room, field) = walled_room();
    let scan = scan_geometry();
    let truth = [2.0, 3.0, 0.4];
    let ranges = reading_at(&room, truth);

    let cloud = InitialParticleCloud {
        particle_count: 400,
        position_variance: 0.05,
        heading_variance: 0.01,
    };
    let mut localizer: MonteCarloLocalizer<NUM_BEAMS> =
        MonteCarloLocalizer::new([2.3, 2.7, 0.4], cloud, BeamModel::default(), SEED).unwrap();

    for _ in 0..8 {
        localizer
            .update_against_field(&field, &scan, &ranges, LikelihoodFieldModel::default())
            .unwrap();
    }

    let (pose, _spread) = localizer.estimate();
    assert!((pose[0] - truth[0]).abs() < 0.5, "{pose:?}");
    assert!((pose[1] - truth[1]).abs() < 0.5, "{pose:?}");
}

#[test]
fn likelihood_is_smoother_than_the_beam_model_f64() {
    let (room, field) = walled_room();
    let scan = scan_geometry();
    let truth = [3.0, 3.0, 0.0];
    let ranges = reading_at(&room, truth);
    let beam_model = BeamModel::<f64>::default();

    // Sweep the pose in 1 mm steps across one cell width, scoring both ways.
    let steps = 200;
    let mut field_scores = Vec::with_capacity(steps);
    let mut beam_scores = Vec::with_capacity(steps);
    for step in 0..steps {
        let pose = [truth[0] + (step as f64) * 0.001, truth[1], truth[2]];
        field_scores.push(log_score(&field, &ranges, pose));

        let mut score = 0.0;
        for (beam, &measured) in ranges.iter().enumerate() {
            let Some(offset) = scan.beam_angle(beam) else {
                continue;
            };
            let modelled =
                room.cast_ray([pose[0], pose[1]], pose[2] + offset, scan.maximum_range());
            score += match modelled {
                Some(distance) => {
                    let residual = distance - measured;
                    -(residual * residual)
                        / (2.0 * beam_model.range_deviation * beam_model.range_deviation)
                }
                None => beam_model.mismatch_penalty,
            };
        }
        beam_scores.push(score);
    }

    // The beam model reads a range quantized to where the cast crosses a cell edge, so its score
    // steps as the pose moves; the field's bilinear lookup varies continuously. The measure that
    // separates them is the largest single step, not the number of turning points.
    let field_step = largest_step(&field_scores);
    let beam_step = largest_step(&beam_scores);
    assert!(
        beam_step > 3.0 * field_step,
        "beam {beam_step} against field {field_step}"
    );
}

/// The largest change between consecutive samples — how far the score can jump for a millimetre
/// of pose.
fn largest_step(values: &[f64]) -> f64 {
    values
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).abs())
        .fold(0.0_f64, f64::max)
}
