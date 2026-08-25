#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Fixture reconstruction: a malformed committed fixture is a bug in the generator, not a runtime
//! condition.

use multicalc::dynamics::ArticulatedBody;
use multicalc::kinematics::{Joint, JointParent, KinematicTree};
use multicalc::linear_algebra::{Matrix, Vector, Vector3D};
use multicalc::spatial::{Quaternion, SE3, SO3, SpatialInertia};

use crate::schema::Fixture;

/// Covers every fixture, the Franka's eleven slots included.
pub const MAX_JOINTS: usize = 16;

/// Row `index` of a matrix value, as a three-vector.
pub fn row3(data: &[f64], columns: usize, index: usize) -> Vector3D<f64> {
    let start = index * columns;
    Vector::new([data[start], data[start + 1], data[start + 2]])
}

/// One row of a K x N matrix value, padded out to the model's width.
pub fn row_readings(data: &[f64], joint_count: usize, row: usize) -> Vector<MAX_JOINTS, f64> {
    let start = row * joint_count;
    Vector::from_fn(|index| {
        if index < joint_count {
            data[start + index]
        } else {
            0.0
        }
    })
}

/// The model the fixture carries, with the joint count it was written for.
///
/// `friction` chooses whether the joint-friction fields come across: with them zeroed the torque
/// is the oracle-pinned rigid-body term alone, which the fixture stores separately.
pub fn body_from_fixture(
    fixture: &Fixture,
    friction: bool,
) -> (ArticulatedBody<MAX_JOINTS, MAX_JOINTS, f64>, usize) {
    let case = fixture.case.as_str();

    let kinds: Vec<char> = fixture.inputs["joint_kinds"].as_str().chars().collect();
    let parents = fixture.inputs["parents"].as_vector();
    let zero_offsets = fixture.inputs["zero_offsets"].as_vector();
    let armatures = fixture.inputs["armatures"].as_vector();
    let dampings = fixture.inputs["dampings"].as_vector();
    let friction_losses = fixture.inputs["friction_losses"].as_vector();
    let masses = fixture.inputs["masses"].as_vector();
    let (_, origin_position_columns, origin_positions) =
        fixture.inputs["origin_positions"].as_matrix();
    let (_, origin_quaternion_columns, origin_quaternions) =
        fixture.inputs["origin_quaternions"].as_matrix();
    let (_, axis_columns, axes) = fixture.inputs["axes"].as_matrix();
    let (_, anchor_columns, anchors) = fixture.inputs["anchors"].as_matrix();
    let (_, center_columns, centers) = fixture.inputs["centers_of_mass"].as_matrix();
    let (_, inertia_columns, rotational_inertias) =
        fixture.inputs["rotational_inertias"].as_matrix();
    let (_, joint_count, _) = fixture.inputs["joint_positions"].as_matrix();

    assert_eq!(joint_count, kinds.len(), "{case}: joint count");
    assert!(
        joint_count <= MAX_JOINTS,
        "{case}: {joint_count} joints exceeds the {MAX_JOINTS} this test builds for"
    );

    let mut joints = Vec::with_capacity(joint_count);
    let mut joint_parents = Vec::with_capacity(joint_count);
    let mut inertias = Vec::with_capacity(joint_count);
    for index in 0..joint_count {
        let start = index * origin_quaternion_columns;
        let orientation = Quaternion::new(
            origin_quaternions[start],
            origin_quaternions[start + 1],
            origin_quaternions[start + 2],
            origin_quaternions[start + 3],
        )
        .try_normalized()
        .unwrap_or_else(|| unreachable!("{case}: joint {index} has a zero origin quaternion"));
        let origin = SE3::from_parts(
            SO3::from_quaternion(orientation),
            row3(&origin_positions, origin_position_columns, index),
        );
        let axis = row3(&axes, axis_columns, index);
        let anchor = row3(&anchors, anchor_columns, index);
        let (damping, friction_loss) = if friction {
            (dampings[index], friction_losses[index])
        } else {
            (0.0, 0.0)
        };

        let joint = match kinds[index] {
            'R' => Joint::revolute(axis, origin).with_anchor(anchor),
            'P' => Joint::prismatic(axis, origin),
            'F' => Joint::fixed(origin),
            other => unreachable!("{case}: joint {index} has kind {other:?}"),
        };
        joints.push(
            joint
                .with_zero_offset(zero_offsets[index])
                .with_armature(armatures[index])
                .with_damping(damping)
                .with_friction_loss(friction_loss),
        );
        joint_parents.push(match parents[index] {
            parent if parent < 0.0 => JointParent::World,
            parent => JointParent::Joint(parent as usize),
        });

        // A slot the model gives no mass carries none, which is what a weld tool frame looks like.
        inertias.push(if masses[index] == 0.0 {
            None
        } else {
            let start = index * inertia_columns;
            let rotational =
                Matrix::from_fn(|row, column| rotational_inertias[start + row * 3 + column]);
            Some(
                SpatialInertia::new(
                    masses[index],
                    row3(&centers, center_columns, index),
                    rotational,
                )
                .unwrap_or_else(|err| unreachable!("{case}: slot {index} inertia: {err:?}")),
            )
        });
    }

    let tree =
        KinematicTree::<MAX_JOINTS, MAX_JOINTS, f64>::try_from_joints(&joints, &joint_parents)
            .unwrap_or_else(|err| unreachable!("{case}: building the tree: {err}"));
    let body = ArticulatedBody::new(tree, &inertias, Vector::new([0.0, 0.0, -9.81]))
        .unwrap_or_else(|err| unreachable!("{case}: building the body: {err:?}"));
    (body, joint_count)
}
