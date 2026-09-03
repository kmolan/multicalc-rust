#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks the mapping goldens: the exact Euclidean distance transform against
//! `scipy.ndimage.distance_transform_edt`, and the costmap inflation and likelihood-field score
//! against the closed forms they are defined by.
//!
//! Both sides of the distance comparison compute an exact transform by different methods, so the
//! tolerance is floating-point noise rather than algorithmic slack.

use multicalc::mapping::{
    DistanceField, DistanceTransformWorkspace, MutableOccupancyMap, OccupancyGrid,
};
use multicalc_qa::load::*;
use multicalc_qa::schema::*;

/// The largest map any fixture carries, and the workspace and grid sizes that follow from it.
const MAX_SPAN: usize = 65;
const MAX_ROWS: usize = 64;
const MAX_COLUMNS: usize = 64;
const WORDS_PER_ROW: usize = 2;

/// A map built to the largest fixture shape, with the cells outside the fixture's own extent left
/// free. Only the fixture's own cells are ever read back.
fn map_from(
    fixture: &Fixture,
) -> (
    OccupancyGrid<MAX_ROWS, MAX_COLUMNS, WORDS_PER_ROW>,
    usize,
    usize,
) {
    let rows = fixture.inputs["rows"].as_int() as usize;
    let columns = fixture.inputs["columns"].as_int() as usize;
    let resolution = fixture.inputs["resolution"].as_scalar();
    assert!(
        rows <= MAX_ROWS && columns <= MAX_COLUMNS,
        "fixture too large"
    );

    let (stored_rows, stored_columns, occupancy) = fixture.inputs["occupancy"].as_matrix();
    assert_eq!(
        (stored_rows, stored_columns),
        (rows, columns),
        "occupancy shape"
    );

    let mut map: OccupancyGrid<MAX_ROWS, MAX_COLUMNS, WORDS_PER_ROW> =
        OccupancyGrid::try_new(resolution, [0.0, 0.0]).unwrap();
    for row in 0..rows {
        for column in 0..columns {
            if occupancy[row * columns + column] != 0.0 {
                map.set_cell(row, column, true);
            }
        }
    }
    (map, rows, columns)
}

/// The distance field of a fixture's map, built at the full grid size.
fn field_from(
    map: &OccupancyGrid<MAX_ROWS, MAX_COLUMNS, WORDS_PER_ROW>,
) -> DistanceField<MAX_ROWS, MAX_COLUMNS> {
    let mut workspace: DistanceTransformWorkspace<MAX_SPAN> = DistanceTransformWorkspace::new();
    DistanceField::try_build(map, &mut workspace).unwrap()
}

#[test]
fn distance_transform_matches_scipy() {
    let mut checked = 0;
    for fixture in load_dir("mapping") {
        if fixture.inputs["kind"].as_str() != "distance_transform" {
            continue;
        }
        checked += 1;
        let (map, rows, columns) = map_from(&fixture);
        let field = field_from(&map);

        let (_, _, expected) = fixture.expected["distance"].as_matrix();
        let tolerance = fixture.tolerances.f64;
        let case = &fixture.case;

        // The map is padded out to the full grid, so a cell near the fixture's own edge sees no
        // obstacle the fixture had; only the fixture's own extent is compared, and its own
        // obstacles are the nearest ones inside it.
        for row in 0..rows {
            for column in 0..columns {
                let got = field.distance_of(row, column).unwrap();
                let want = expected[row * columns + column];
                assert!(
                    close(got, want, tolerance),
                    "{case}: cell ({row}, {column}): got {got}, want {want}, tol {tolerance:?}"
                );
            }
        }
    }
    assert!(
        checked >= 5,
        "expected five distance-transform fixtures, saw {checked}"
    );
}

#[test]
fn costmap_matches_the_closed_form() {
    use multicalc::mapping::CostGrid;

    let mut checked = 0;
    for fixture in load_dir("mapping") {
        if fixture.inputs["kind"].as_str() != "costmap" {
            continue;
        }
        checked += 1;
        let (map, rows, columns) = map_from(&fixture);
        let field = field_from(&map);

        let inscribed = fixture.inputs["inscribed_radius"].as_vector();
        let inflation = fixture.inputs["inflation_radius"].as_vector();
        let scaling = fixture.inputs["cost_scaling_factor"].as_vector();
        let case = &fixture.case;

        for (index, ((&inscribed, &inflation), &scaling)) in inscribed
            .iter()
            .zip(inflation.iter())
            .zip(scaling.iter())
            .enumerate()
        {
            let costmap: CostGrid<MAX_ROWS, MAX_COLUMNS> =
                CostGrid::try_build(&field, inscribed, inflation, scaling).unwrap();
            let (_, _, expected) = fixture.expected[&format!("cost_{index}")].as_matrix();

            // Both sides are u8 counts, so this is an exact comparison.
            for row in 0..rows {
                for column in 0..columns {
                    let got = costmap.cost_of(row, column).unwrap();
                    let want = expected[row * columns + column] as u8;
                    assert_eq!(
                        got, want,
                        "{case}: set {index}, cell ({row}, {column}): got {got}, want {want}"
                    );
                }
            }
        }
    }
    assert!(checked >= 2, "expected two costmap fixtures, saw {checked}");
}

#[test]
fn likelihood_field_matches_the_closed_form() {
    let mut checked = 0;
    for fixture in load_dir("mapping") {
        if fixture.inputs["kind"].as_str() != "likelihood_field" {
            continue;
        }
        checked += 1;
        let (map, _, _) = map_from(&fixture);
        let field = field_from(&map);

        let field_of_view = fixture.inputs["field_of_view"].as_scalar();
        let maximum_range = fixture.inputs["maximum_range"].as_scalar();
        let deviation = fixture.inputs["measurement_deviation"].as_scalar();
        let random_weight = fixture.inputs["random_measurement_weight"].as_scalar();
        let ranges = fixture.inputs["ranges"].as_vector();
        let pose_x = fixture.inputs["pose_x"].as_vector();
        let pose_y = fixture.inputs["pose_y"].as_vector();
        let pose_heading = fixture.inputs["pose_heading"].as_vector();

        let want_distances = fixture.expected["endpoint_distance"].as_vector();
        let want_weights = fixture.expected["log_weight"].as_vector();
        let tolerance = fixture.tolerances.f64;
        let case = &fixture.case;

        let num_beams = ranges.len();
        let span = (num_beams - 1) as f64;
        let mut entry = 0;
        for (pose, ((&x, &y), &heading)) in pose_x
            .iter()
            .zip(pose_y.iter())
            .zip(pose_heading.iter())
            .enumerate()
        {
            let mut score = 0.0;
            for (beam, &measured) in ranges.iter().enumerate() {
                let offset = -field_of_view * 0.5 + field_of_view * beam as f64 / span;
                let bearing = heading + offset;
                let endpoint = [x + measured * bearing.cos(), y + measured * bearing.sin()];
                let looked_up = field.distance_at(endpoint);

                // The generator marks an endpoint that fell off the field with -1.
                let want_distance = want_distances[entry];
                match looked_up {
                    Some(distance) => assert!(
                        want_distance >= 0.0 && close(distance, want_distance, tolerance),
                        "{case}: pose {pose}, beam {beam}: got {distance}, want {want_distance}"
                    ),
                    None => assert!(
                        want_distance < 0.0,
                        "{case}: pose {pose}, beam {beam}: off the field, want {want_distance}"
                    ),
                }
                entry += 1;

                let hit = match looked_up {
                    Some(distance) => {
                        (-(distance * distance) / (2.0 * deviation * deviation)).exp()
                    }
                    None => 0.0,
                };
                score += ((1.0 - random_weight) * hit + random_weight / maximum_range).ln();
            }

            let want = want_weights[pose];
            assert!(
                close(score, want, tolerance),
                "{case}: pose {pose} log-weight: got {score}, want {want}, tol {tolerance:?}"
            );
        }
    }
    assert!(
        checked >= 1,
        "expected a likelihood-field fixture, saw {checked}"
    );
}
