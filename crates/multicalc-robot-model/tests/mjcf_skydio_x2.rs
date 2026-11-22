#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Loads the vendored Skydio X2 and checks it against numbers worked out by hand from the file.
//!
//! The model states no mass properties of its own, so every number here comes out of the four
//! rotor discs and the hull, with their form and size supplied by nested default blocks. Each
//! number in the file is an exact decimal and each shape states its mass outright, so the whole
//! result is a ratio of whole numbers and the expected values below are exact.

use std::path::Path;

use multicalc::linear_algebra::Vector;
use multicalc_robot_model::{GeometryShape, RobotModel};

/// Mass, in kilograms: four 0.25 kg rotors and a 0.325 kg hull.
const MASS: f64 = 1.325;

/// Where the body balances, in metres. The rotors sit at two different heights, so this is not
/// the middle of anything obvious.
const CENTER_OF_MASS: [f64; 3] = [0.0, 0.0, 0.053962264150943406];

/// Units are in kg·m².
/// The corner terms are nonzero because the front rotors sit 3 cm above the rear pair, so the
/// body does not spin cleanly about its own axes.
const ROTATIONAL_INERTIA: [[f64; 3]; 3] = [
    [0.036651698113207544, 0.0, -0.0021],
    [0.0, 0.025411698113207547, 0.0],
    [-0.0021, 0.0, 0.060528],
];

const TOLERANCE: f64 = 1e-12;

#[must_use]
fn skydio_x2() -> RobotModel {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../third_party/menagerie/skydio_x2/x2.xml");
    multicalc_robot_model::mjcf::load_path(&path).unwrap()
}

fn assert_close(actual: f64, expected: f64, label: &str) {
    assert!(
        (actual - expected).abs() < TOLERANCE,
        "{label}: {actual} is not {expected}"
    );
}

#[test]
fn reads_the_body_and_its_free_joint() {
    let model = skydio_x2();

    assert_eq!(model.name(), "Skydio X2");
    assert_eq!(model.body(0).unwrap().name(), "x2");
    assert!(model.has_floating_base());
    assert_eq!(model.body_count(), 1);
}

#[test]
fn reads_where_the_body_sits() {
    let model = skydio_x2();
    let body = model.body_named("x2").unwrap();

    let translation = body.pose().translation().into_array();
    for (index, expected) in [0.0, 0.0, 0.1].into_iter().enumerate() {
        assert_close(translation[index], expected, "translation");
    }

    // The file gives the body no turn, so its rotation is the one that does nothing.
    let quaternion = body.pose().rotation().quaternion().as_array();
    for (index, expected) in [1.0, 0.0, 0.0, 0.0].into_iter().enumerate() {
        assert_close(quaternion[index], expected, "quaternion");
    }
}

#[test]
fn works_the_mass_out_from_the_shapes() {
    let model = skydio_x2();
    let inertia = model.body_named("x2").unwrap().inertia().unwrap();

    assert_close(inertia.mass(), MASS, "mass");

    let center_of_mass = inertia.center_of_mass().into_array();
    for (index, expected) in CENTER_OF_MASS.into_iter().enumerate() {
        assert_close(center_of_mass[index], expected, "centre of mass");
    }
}

#[test]
fn works_out_how_hard_the_body_is_to_spin() {
    let model = skydio_x2();
    let inertia = model
        .body_named("x2")
        .unwrap()
        .inertia()
        .unwrap()
        .rotational_inertia();

    for row in 0..3 {
        for column in 0..3 {
            assert_close(
                inertia[(row, column)],
                ROTATIONAL_INERTIA[row][column],
                "rotational inertia",
            );
        }
    }
}

#[test]
fn the_airframe_draws_a_scaled_mesh_and_nine_primitives() {
    let model = skydio_x2();
    let shapes = model.body(0).unwrap().visual_geometry();
    assert_eq!(shapes.len(), 10);

    // `<compiler assetdir="assets">` stands in for meshdir, and the scale comes from
    // `<default class="x2"><mesh scale="0.01 0.01 0.01"/>`, not from the geom.
    assert_eq!(
        shapes[0].shape(),
        &GeometryShape::Mesh {
            file: "assets/X2_lowpoly.obj".to_owned(),
            scale: Vector::new([0.01, 0.01, 0.01]),
        }
    );
    assert_eq!(shapes[0].group(), 2);
    // `<material name="phong3SG">` states a texture and no rgba, so it is left white.
    assert_eq!(shapes[0].color(), [1.0, 1.0, 1.0, 1.0]);

    // Four collision boxes, then four rotor ellipsoids from `<default class="rotor">`.
    assert_eq!(
        shapes[1].shape(),
        &GeometryShape::Box {
            half_extents: Vector::new([0.06, 0.027, 0.02])
        }
    );
    assert_eq!(
        shapes[5].shape(),
        &GeometryShape::Ellipsoid {
            semi_axes: Vector::new([0.13, 0.13, 0.01])
        }
    );
    assert_eq!(shapes[5].group(), 3);

    // The last is `material="invisible"`, rgba="0 0 0 0".
    assert_eq!(shapes[9].color(), [0.0, 0.0, 0.0, 0.0]);
}
