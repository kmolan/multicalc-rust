#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks model ingestion against MuJoCo's own compile of the same file.
//!
//! MuJoCo defines what an MJCF file means, so it is the reference here rather than a second reader
//! of the same format. It reports a body's inertia as three numbers along the axes the body lines
//! up best with, plus the turn to get there; the generator rebuilds the full tensor in the body's
//! own axes before writing the golden, so what is compared below is nine numbers against nine.

use std::path::Path;

use multicalc::kinematics::JointKind;
use multicalc::spatial::FreeJointState;
use multicalc_qa::load::*;
use multicalc_qa::schema::{Fixture, Tol};

/// MuJoCo does not keep a body's inertia as the nine numbers the file states (or the three plus a
/// turn `diaginertia` gives); it diagonalizes whatever it is given and keeps that, so the tensor
/// rebuilt from `body_inertia`/`body_iquat` has been through an eigendecomposition MuJoCo did
/// internally rather than being read back unchanged. Mass, position and orientation carry no such
/// step and are still checked at the fixture's own tight bound; only this one field needs the
/// looser one, measured against the file's own numbers up to a small safety margin.
const ROTATIONAL_INERTIA_TOLERANCE: Tol = Tol {
    abs: 1e-6,
    rel: 1e-6,
};

/// The vendored models the fixtures name, which sit outside this crate.
#[must_use]
fn menagerie() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../third_party/menagerie")
}

#[test]
fn ingested_models_match_mujoco() {
    for fx in load_dir("mjcf") {
        match fx.case.as_str() {
            "skydio_x2_free_joint" => check_free_body(&fx),
            "franka_panda_tree" => check_body_tree(&fx),
            other => unreachable!("no comparison for fixture {other}"),
        }
    }
}

/// The Skydio X2: a single body on a free joint, its mass worked out from its shapes.
fn check_free_body(fx: &Fixture) {
    let case = fx.case.as_str();
    let path = menagerie().join(fx.inputs["model_file"].as_str());
    let model = multicalc_mjcf::load_path(&path)
        .unwrap_or_else(|e| unreachable!("{case}: loading {path:?}: {e}"));
    let body = model
        .body_named("x2")
        .unwrap_or_else(|| unreachable!("{case}: model has no body called x2"));
    let t = fx.tolerances.f64;

    assert_scalar(
        body.inertia().mass(),
        &fx.expected["mass"],
        t,
        &format!("{case} mass"),
    );
    assert_vector::<3>(
        &body.inertia().center_of_mass(),
        &fx.expected["center_of_mass"],
        t,
        &format!("{case} center_of_mass"),
    );
    assert_matrix::<3, 3>(
        &body.inertia().rotational_inertia(),
        &fx.expected["rotational_inertia"],
        t,
        &format!("{case} rotational_inertia"),
    );
    assert_vector::<3>(
        &body.pose().translation(),
        &fx.expected["body_position"],
        t,
        &format!("{case} body_position"),
    );

    // A quaternion and its negative name the same turn, so both sides are compared in their
    // scalar-positive form rather than component by component as written.
    let quaternion = body.pose().rotation().quaternion().as_array();
    let want = fx.expected["body_quaternion"].as_vector();
    let flip = if quaternion[0] < 0.0 { -1.0 } else { 1.0 };
    let want_flip = if want[0] < 0.0 { -1.0 } else { 1.0 };
    for index in 0..4 {
        assert!(
            close(flip * quaternion[index], want_flip * want[index], t),
            "{case} body_quaternion[{index}]: got {}, want {}",
            flip * quaternion[index],
            want_flip * want[index]
        );
    }

    assert_eq!(
        i64::from(model.has_floating_base()),
        fx.expected["free_joint"].as_int(),
        "{case} free_joint"
    );

    // Ties the parsed model to how the core type packs a free joint's numbers. The fixture
    // keys keep MuJoCo's own `nq` / `nv` wording, since they record what MuJoCo reported.
    assert_eq!(
        FreeJointState::<f64>::GENERALIZED_POSITION_DIMENSION as i64,
        fx.expected["configuration_dimension"].as_int(),
        "{case} configuration_dimension"
    );
    assert_eq!(
        FreeJointState::<f64>::GENERALIZED_VELOCITY_DIMENSION as i64,
        fx.expected["velocity_dimension"].as_int(),
        "{case} velocity_dimension"
    );
}

/// The Franka Panda: the body tree, every joint's geometry and travel, and the dynamics figures
/// forward kinematics never reads — field for field against MuJoCo's compiled model, and again
/// once the model has gone through `kinematic_tree`.
fn check_body_tree(fx: &Fixture) {
    let case = fx.case.as_str();
    let path = menagerie().join(fx.inputs["model_file"].as_str());
    let model = multicalc_mjcf::load_path(&path)
        .unwrap_or_else(|e| unreachable!("{case}: loading {path:?}: {e}"));
    let t = fx.tolerances.f64;

    assert_eq!(
        model.body_count() as i64,
        fx.expected["body_count"].as_int(),
        "{case} body_count"
    );
    assert_eq!(
        model.movable_joint_count() as i64,
        fx.expected["movable_joint_count"].as_int(),
        "{case} movable_joint_count"
    );
    assert_eq!(
        i64::from(model.has_floating_base()),
        fx.expected["free_joint"].as_int(),
        "{case} free_joint"
    );

    let body_names: Vec<&str> = fx.expected["body_names"]
        .as_str()
        .split_whitespace()
        .collect();
    assert_eq!(
        body_names.len(),
        model.body_count(),
        "{case} body_names count"
    );
    for (index, name) in body_names.iter().enumerate() {
        assert_eq!(
            model.body(index).unwrap().name(),
            *name,
            "{case} body {index} name"
        );
    }

    let parents = fx.expected["parents"].as_vector();
    let (_, _, origin_positions) = fx.expected["origin_positions"].as_matrix();
    let (_, _, origin_quaternions) = fx.expected["origin_quaternions"].as_matrix();
    let joint_kinds: Vec<char> = fx.expected["joint_kinds"].as_str().chars().collect();
    let (_, _, axes) = fx.expected["axes"].as_matrix();
    let (_, _, anchors) = fx.expected["anchors"].as_matrix();
    let zero_offsets = fx.expected["zero_offsets"].as_vector();
    let limited = fx.expected["limited"].as_vector();
    let (_, _, ranges) = fx.expected["ranges"].as_matrix();
    let armatures = fx.expected["armatures"].as_vector();
    let dampings = fx.expected["dampings"].as_vector();
    let friction_losses = fx.expected["friction_losses"].as_vector();
    let spring_references = fx.expected["spring_references"].as_vector();
    let spring_stiffnesses = fx.expected["spring_stiffnesses"].as_vector();
    let masses = fx.expected["masses"].as_vector();
    let (_, _, centers_of_mass) = fx.expected["centers_of_mass"].as_matrix();
    let (_, _, rotational_inertias) = fx.expected["rotational_inertias"].as_matrix();

    for index in 0..model.body_count() {
        let body = model.body(index).unwrap();
        let ctx = format!("{case} body {index}");

        let expected_parent = parents[index] as i64;
        let got_parent = body.parent().map_or(-1, |p| p as i64);
        assert_eq!(got_parent, expected_parent, "{ctx} parent");

        let translation = body.pose().translation();
        for axis in 0..3 {
            assert!(
                close(translation[axis], origin_positions[3 * index + axis], t),
                "{ctx} origin_position[{axis}]"
            );
        }

        // A quaternion and its negative name the same turn.
        let quaternion = body.pose().rotation().quaternion().as_array();
        let want = [
            origin_quaternions[4 * index],
            origin_quaternions[4 * index + 1],
            origin_quaternions[4 * index + 2],
            origin_quaternions[4 * index + 3],
        ];
        let flip = if quaternion[0] < 0.0 { -1.0 } else { 1.0 };
        let want_flip = if want[0] < 0.0 { -1.0 } else { 1.0 };
        for component in 0..4 {
            assert!(
                close(flip * quaternion[component], want_flip * want[component], t),
                "{ctx} origin_quaternion[{component}]"
            );
        }

        assert!(close(body.inertia().mass(), masses[index], t), "{ctx} mass");
        let center_of_mass = body.inertia().center_of_mass();
        for axis in 0..3 {
            assert!(
                close(center_of_mass[axis], centers_of_mass[3 * index + axis], t),
                "{ctx} center_of_mass[{axis}]"
            );
        }
        let inertia = body.inertia().rotational_inertia();
        for row in 0..3 {
            for column in 0..3 {
                assert!(
                    close(
                        inertia[(row, column)],
                        rotational_inertias[(3 * index + row) * 3 + column],
                        ROTATIONAL_INERTIA_TOLERANCE,
                    ),
                    "{ctx} rotational_inertia({row},{column})"
                );
            }
        }

        let expected_kind = joint_kinds[index];
        match (body.joint(), expected_kind) {
            (None, 'F') => {}
            (Some(joint), 'R') => assert_eq!(joint.kind(), JointKind::Revolute, "{ctx} kind"),
            (Some(joint), 'P') => assert_eq!(joint.kind(), JointKind::Prismatic, "{ctx} kind"),
            (joint, kind) => unreachable!("{ctx}: joint {joint:?} does not match kind {kind}"),
        }

        let Some(joint) = body.joint() else {
            continue;
        };

        let axis = joint.axis();
        for component in 0..3 {
            assert!(
                close(axis[component], axes[3 * index + component], t),
                "{ctx} axis[{component}]"
            );
        }
        let anchor = joint.anchor();
        for component in 0..3 {
            assert!(
                close(anchor[component], anchors[3 * index + component], t),
                "{ctx} anchor[{component}]"
            );
        }
        assert!(
            close(joint.zero_offset(), zero_offsets[index], t),
            "{ctx} zero_offset"
        );
        assert!(
            close(joint.armature(), armatures[index], t),
            "{ctx} armature"
        );
        assert!(close(joint.damping(), dampings[index], t), "{ctx} damping");
        assert!(
            close(joint.friction_loss(), friction_losses[index], t),
            "{ctx} friction_loss"
        );
        assert!(
            close(joint.spring_reference(), spring_references[index], t),
            "{ctx} spring_reference"
        );
        assert!(
            close(joint.spring_stiffness(), spring_stiffnesses[index], t),
            "{ctx} spring_stiffness"
        );

        let expected_limits = if limited[index] == 0.0 {
            None
        } else {
            Some((ranges[2 * index], ranges[2 * index + 1]))
        };
        match (joint.limits(), expected_limits) {
            (None, None) => {}
            (Some((lower, upper)), Some((want_lower, want_upper))) => {
                assert!(close(lower, want_lower, t), "{ctx} limits lower");
                assert!(close(upper, want_upper, t), "{ctx} limits upper");
            }
            (got, want) => unreachable!("{ctx}: limits {got:?} does not match {want:?}"),
        }
    }

    // The converted tree: the same dynamics data, having survived `kinematic_tree`. Forward
    // kinematics never reads armature, damping, friction loss or the zero reference, so this is
    // the only place in the suite that would notice the conversion dropping them.
    let tree = model
        .kinematic_tree::<16, 16>()
        .unwrap_or_else(|e| unreachable!("{case}: kinematic_tree: {e}"));
    for index in 0..model.body_count() {
        let joint = tree.joint(index).unwrap();
        let ctx = format!("{case} tree slot {index}");
        assert!(
            close(joint.armature(), armatures[index], t),
            "{ctx} armature"
        );
        assert!(close(joint.damping(), dampings[index], t), "{ctx} damping");
        assert!(
            close(joint.friction_loss(), friction_losses[index], t),
            "{ctx} friction_loss"
        );
        assert!(
            close(joint.zero_offset(), zero_offsets[index], t),
            "{ctx} zero_offset"
        );
    }
}
