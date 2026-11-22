#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks the spatial algebra against Pinocchio's own, on the same random poses, motions, forces
//! and mass distributions.
//!
//! Pinocchio stores these six numbers linear part first, the same way this crate does, so a golden
//! that disagrees means the ordering was transcribed wrong — which is the whole reason these
//! fixtures exist. The generator works every case out a second time in numpy before writing it.

use multicalc::linear_algebra::{Matrix, Matrix6D, Vector, Vector6D};
use multicalc::spatial::{Quaternion, SE3, SO3, SpatialInertia, Twist, Wrench};
use multicalc_qa::load::*;
use multicalc_qa::schema::*;

const SAMPLE_COUNT: usize = 8;

#[test]
fn spatial_goldens() {
    let fixtures = load_dir("spatial");
    let mut checked = 0;
    for fixture in &fixtures {
        match fixture.case.as_str() {
            "plucker_transforms" => check_plucker_transforms(fixture),
            "spatial_cross_products" => check_cross_products(fixture),
            "spatial_inertia_algebra" => check_inertia_algebra(fixture),
            "spatial_inertia_transform_and_composite" => {
                check_inertia_transform_and_composite(fixture);
            }
            other => panic!("no check registered for spatial fixture {other}"),
        }
        checked += 1;
    }
    assert_eq!(
        checked, 4,
        "expected four spatial fixtures, found {checked}"
    );
}

fn twist_row(rows: &Matrix<SAMPLE_COUNT, 6>, index: usize) -> Twist<f64> {
    Twist::from_array([
        rows[(index, 0)],
        rows[(index, 1)],
        rows[(index, 2)],
        rows[(index, 3)],
        rows[(index, 4)],
        rows[(index, 5)],
    ])
}

fn wrench_row(rows: &Matrix<SAMPLE_COUNT, 6>, index: usize) -> Wrench<f64> {
    Wrench::from_array([
        rows[(index, 0)],
        rows[(index, 1)],
        rows[(index, 2)],
        rows[(index, 3)],
        rows[(index, 4)],
        rows[(index, 5)],
    ])
}

fn pose_row(rows: &Matrix<SAMPLE_COUNT, 7>, index: usize) -> SE3<f64> {
    let translation = Vector::new([rows[(index, 0)], rows[(index, 1)], rows[(index, 2)]]);
    let quaternion = Quaternion::new(
        rows[(index, 3)],
        rows[(index, 4)],
        rows[(index, 5)],
        rows[(index, 6)],
    );
    SE3::from_parts(SO3::try_from_quaternion(quaternion).unwrap(), translation)
}

fn inertia_row(rows: &Matrix<SAMPLE_COUNT, 10>, index: usize) -> SpatialInertia<f64> {
    let mass = rows[(index, 0)];
    let center = Vector::new([rows[(index, 1)], rows[(index, 2)], rows[(index, 3)]]);
    let (xx, yy, zz) = (rows[(index, 4)], rows[(index, 5)], rows[(index, 6)]);
    let (xy, xz, yz) = (rows[(index, 7)], rows[(index, 8)], rows[(index, 9)]);
    SpatialInertia::new(
        mass,
        center,
        Matrix::new([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]),
    )
    .unwrap()
}

/// One six-vector out of a stack of samples, compared entry by entry.
fn assert_row(got: &Vector6D<f64>, want: &Value, index: usize, tolerance: Tol, ctx: &str) {
    let (_rows, cols, data) = want.as_matrix();
    assert_eq!(cols, 6, "{ctx}: width");
    for component in 0..6 {
        let expected = data[index * 6 + component];
        assert!(
            close(got[component], expected, tolerance),
            "{ctx}[{index}][{component}]: got {}, want {expected}, tol {tolerance:?}",
            got[component]
        );
    }
}

/// One 6×6 block out of a stack of blocks, compared entry by entry.
fn assert_block(got: &Matrix6D<f64>, want: &Value, index: usize, tolerance: Tol, ctx: &str) {
    let (_rows, cols, data) = want.as_matrix();
    assert_eq!(cols, 6, "{ctx}: width");
    for row in 0..6 {
        for col in 0..6 {
            let expected = data[(index * 6 + row) * 6 + col];
            assert!(
                close(got[(row, col)], expected, tolerance),
                "{ctx}[{index}]({row},{col}): got {}, want {expected}, tol {tolerance:?}",
                got[(row, col)]
            );
        }
    }
}

/// The ten numbers a mass distribution stores, compared one by one.
fn assert_inertia_row(
    got: &SpatialInertia<f64>,
    want: &Value,
    index: usize,
    tolerance: Tol,
    ctx: &str,
) {
    let (_rows, cols, data) = want.as_matrix();
    assert_eq!(cols, 10, "{ctx}: width");
    let inertia = got.rotational_inertia();
    let center = got.center_of_mass();
    let entries = [
        got.mass(),
        center[0],
        center[1],
        center[2],
        inertia[(0, 0)],
        inertia[(1, 1)],
        inertia[(2, 2)],
        inertia[(0, 1)],
        inertia[(0, 2)],
        inertia[(1, 2)],
    ];
    for (component, entry) in entries.iter().enumerate() {
        let expected = data[index * 10 + component];
        assert!(
            close(*entry, expected, tolerance),
            "{ctx}[{index}][{component}]: got {entry}, want {expected}, tol {tolerance:?}"
        );
    }
}

fn check_plucker_transforms(fixture: &Fixture) {
    let poses = to_matrix::<SAMPLE_COUNT, 7>(&fixture.inputs["poses"]);
    let twists = to_matrix::<SAMPLE_COUNT, 6>(&fixture.inputs["twists"]);
    let wrenches = to_matrix::<SAMPLE_COUNT, 6>(&fixture.inputs["wrenches"]);
    let tolerance = fixture.tolerances.f64;

    for index in 0..SAMPLE_COUNT {
        let pose = pose_row(&poses, index);
        let twist = twist_row(&twists, index);
        let wrench = wrench_row(&wrenches, index);

        assert_row(
            &pose.act_twist(twist).to_vector(),
            &fixture.expected["transformed_twists"],
            index,
            tolerance,
            "transformed twist",
        );
        assert_row(
            &pose.act_wrench(wrench).to_vector(),
            &fixture.expected["transformed_wrenches"],
            index,
            tolerance,
            "transformed wrench",
        );
        assert_row(
            &pose.inverse_act_twist(twist).to_vector(),
            &fixture.expected["inverse_transformed_twists"],
            index,
            tolerance,
            "inverse transformed twist",
        );
        assert_row(
            &pose.inverse_act_wrench(wrench).to_vector(),
            &fixture.expected["inverse_transformed_wrenches"],
            index,
            tolerance,
            "inverse transformed wrench",
        );
        assert_block(
            &pose.adjoint(),
            &fixture.expected["motion_adjoints"],
            index,
            tolerance,
            "motion adjoint",
        );
        assert_block(
            &pose.force_adjoint(),
            &fixture.expected["force_adjoints"],
            index,
            tolerance,
            "force adjoint",
        );
    }
}

fn check_cross_products(fixture: &Fixture) {
    let first_twists = to_matrix::<SAMPLE_COUNT, 6>(&fixture.inputs["first_twists"]);
    let second_twists = to_matrix::<SAMPLE_COUNT, 6>(&fixture.inputs["second_twists"]);
    let wrenches = to_matrix::<SAMPLE_COUNT, 6>(&fixture.inputs["wrenches"]);
    let powers = fixture.expected["powers"].as_vector();
    let tolerance = fixture.tolerances.f64;

    for index in 0..SAMPLE_COUNT {
        let first = twist_row(&first_twists, index);
        let second = twist_row(&second_twists, index);
        let wrench = wrench_row(&wrenches, index);

        assert_row(
            &first.cross(second).to_vector(),
            &fixture.expected["motion_crosses"],
            index,
            tolerance,
            "motion cross",
        );
        assert_row(
            &first.cross_wrench(wrench).to_vector(),
            &fixture.expected["force_crosses"],
            index,
            tolerance,
            "force cross",
        );

        let power = first.dot_wrench(wrench);
        assert!(
            close(power, powers[index], tolerance),
            "power[{index}]: got {power}, want {}, tol {tolerance:?}",
            powers[index]
        );
    }
}

fn check_inertia_algebra(fixture: &Fixture) {
    let parameters = to_matrix::<SAMPLE_COUNT, 10>(&fixture.inputs["inertia_parameters"]);
    let twists = to_matrix::<SAMPLE_COUNT, 6>(&fixture.inputs["twists"]);
    let energies = fixture.expected["kinetic_energies"].as_vector();
    let tolerance = fixture.tolerances.f64;

    for index in 0..SAMPLE_COUNT {
        let inertia = inertia_row(&parameters, index);
        let velocity = twist_row(&twists, index);

        assert_block(
            &inertia.to_matrix(),
            &fixture.expected["inertia_matrices"],
            index,
            tolerance,
            "inertia matrix",
        );
        assert_row(
            &inertia.momentum(velocity).to_vector(),
            &fixture.expected["momenta"],
            index,
            tolerance,
            "momentum",
        );
        assert_row(
            &inertia.bias_wrench(velocity).to_vector(),
            &fixture.expected["bias_wrenches"],
            index,
            tolerance,
            "bias wrench",
        );

        let energy = inertia.kinetic_energy(velocity);
        assert!(
            close(energy, energies[index], tolerance),
            "kinetic energy[{index}]: got {energy}, want {}, tol {tolerance:?}",
            energies[index]
        );
    }
}

fn check_inertia_transform_and_composite(fixture: &Fixture) {
    let first_parameters =
        to_matrix::<SAMPLE_COUNT, 10>(&fixture.inputs["first_inertia_parameters"]);
    let second_parameters =
        to_matrix::<SAMPLE_COUNT, 10>(&fixture.inputs["second_inertia_parameters"]);
    let poses = to_matrix::<SAMPLE_COUNT, 7>(&fixture.inputs["poses"]);
    let tolerance = fixture.tolerances.f64;

    for index in 0..SAMPLE_COUNT {
        let first = inertia_row(&first_parameters, index);
        let second = inertia_row(&second_parameters, index);
        let pose = pose_row(&poses, index);

        assert_inertia_row(
            &pose.act_inertia(first),
            &fixture.expected["transformed_inertia_parameters"],
            index,
            tolerance,
            "transformed inertia",
        );
        assert_inertia_row(
            &first.combined(second),
            &fixture.expected["combined_inertia_parameters"],
            index,
            tolerance,
            "combined inertia",
        );
    }
}
