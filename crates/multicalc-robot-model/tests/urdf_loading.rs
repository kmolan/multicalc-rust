#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg(feature = "urdf")]

//! The URDF reader, worked against the hand-written models in `tests/models/`. Each one is small
//! and exists to reach one part of the subset, and each `bad_` one is refused by name.

use std::path::{Path, PathBuf};

use multicalc::kinematics::JointKind;
use multicalc::linear_algebra::Vector;
use multicalc::spatial::Quaternion;
use multicalc_robot_model::{ModelError, ModelFormat, RobotModel, RustSourceOptions};

/// Where the hand-written models live.
fn model_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/models")
        .join(name)
}

fn model(name: &str) -> RobotModel {
    multicalc_robot_model::urdf::load_path(&model_path(name))
        .unwrap_or_else(|e| panic!("loading {name}: {e}"))
}

/// The error one of the `bad_` models is refused with.
fn refusal(name: &str) -> ModelError {
    match multicalc_robot_model::urdf::load_path(&model_path(name)) {
        Ok(_) => panic!("{name} loaded, and it should not have"),
        Err(e) => e,
    }
}

#[test]
fn two_link_arm_reads_its_tree() {
    let arm = model("two_link_arm.urdf");
    assert_eq!(arm.format(), ModelFormat::Urdf);
    assert_eq!(arm.name(), "two_link_arm");
    assert_eq!(arm.body_count(), 3);

    let names: Vec<&str> = arm.bodies().iter().map(|body| body.name()).collect();
    assert_eq!(names, ["base_link", "upper_arm", "forearm"]);
    assert_eq!(arm.body(0).unwrap().parent(), None);
    assert_eq!(arm.body(1).unwrap().parent(), Some(0));
    assert_eq!(arm.body(2).unwrap().parent(), Some(1));
    assert!(!arm.has_floating_base());
}

#[test]
fn joint_origin_becomes_the_body_pose() {
    let arm = model("two_link_arm.urdf");

    // URDF puts the transform on the joint, so the top body sits at the origin.
    let root = arm.body(0).unwrap().pose();
    assert_eq!(root.translation().into_array(), [0.0; 3]);
    assert_eq!(root.rotation().quaternion(), Quaternion::identity());

    let shoulder = arm.body(1).unwrap().pose();
    for (axis, want) in [0.0, 0.0, 0.5].into_iter().enumerate() {
        assert!(
            (shoulder.translation()[axis] - want).abs() < 1e-15,
            "shoulder translation[{axis}]"
        );
    }
}

#[test]
fn rpy_becomes_the_right_turn() {
    let arm = model("two_link_arm.urdf");

    // A quarter turn about z takes the x direction onto the y direction.
    let turned = arm
        .body(1)
        .unwrap()
        .pose()
        .rotation()
        .act(Vector::new([1.0, 0.0, 0.0]));
    for (axis, want) in [0.0, 1.0, 0.0].into_iter().enumerate() {
        assert!((turned[axis] - want).abs() < 1e-12, "turned[{axis}]");
    }

    // And all three angles at once agree with the same three angles read straight.
    let three_angles = model("massless_frames.urdf");
    let stated = Quaternion::from_euler_zyx(0.0, 0.0, std::f64::consts::FRAC_PI_2);
    let read = three_angles.body(2).unwrap().pose().rotation().quaternion();
    let sign = if read.w() * stated.w() < 0.0 {
        -1.0
    } else {
        1.0
    };
    for (component, want) in [stated.w(), stated.x(), stated.y(), stated.z()]
        .into_iter()
        .enumerate()
    {
        let got = sign * [read.w(), read.x(), read.y(), read.z()][component];
        assert!((got - want).abs() < 1e-12, "quaternion[{component}]");
    }
}

#[test]
fn axis_defaults_to_x() {
    // URDF's default axis is x, where MJCF's is z.
    let kinds = model("every_joint_kind.urdf");
    let arm = model("two_link_arm.urdf");
    assert_eq!(
        arm.body(1).unwrap().joint().unwrap().axis().into_array(),
        [0.0, 0.0, 1.0]
    );
    let spins = kinds.body_named("spinning").unwrap().joint().unwrap();
    assert_eq!(spins.axis().into_array(), [0.0, 1.0, 0.0]);
}

#[test]
fn anchor_is_always_zero() {
    // A URDF joint sits at the origin of the link it drives, so it never carries an offset.
    for body in model("two_link_arm.urdf").bodies() {
        if let Some(joint) = body.joint() {
            assert_eq!(joint.anchor().into_array(), [0.0; 3], "{}", body.name());
        }
    }
}

#[test]
fn every_joint_kind_reads() {
    let kinds = model("every_joint_kind.urdf");
    assert_eq!(kinds.body_count(), 5);

    let kind_of = |name: &str| kinds.body_named(name).unwrap().joint().map(|j| j.kind());
    assert_eq!(kind_of("turning"), Some(JointKind::Revolute));
    assert_eq!(kind_of("spinning"), Some(JointKind::Continuous));
    assert_eq!(kind_of("sliding"), Some(JointKind::Prismatic));
    // A welded link carries no joint at all, the same as an MJCF body with none.
    assert_eq!(kind_of("welded"), None);
    assert_eq!(kinds.movable_joint_count(), 3);
}

#[test]
fn limits_come_across() {
    let kinds = model("every_joint_kind.urdf");
    let joint = |name: &str| kinds.body_named(name).unwrap().joint().unwrap();
    assert_eq!(joint("turning").limits(), Some((-1.0, 1.0)));
    assert_eq!(joint("sliding").limits(), Some((0.0, 2.0)));
    // A joint that turns round and round has nowhere to stop.
    assert_eq!(joint("spinning").limits(), None);
}

#[test]
fn dynamics_come_across() {
    let arm = model("two_link_arm.urdf");
    let shoulder = arm.body(1).unwrap().joint().unwrap();
    assert_eq!(shoulder.damping(), 0.25);
    assert_eq!(shoulder.friction_loss(), 0.5);

    // URDF has no equivalent of any of these.
    assert_eq!(shoulder.armature(), 0.0);
    assert_eq!(shoulder.zero_offset(), 0.0);
    assert_eq!(shoulder.spring_reference(), 0.0);
    assert_eq!(shoulder.spring_stiffness(), 0.0);

    // A joint stating no dynamics at all gets nothing that resists it.
    let elbow = arm.body(2).unwrap().joint().unwrap();
    assert_eq!(elbow.damping(), 0.125);
    assert_eq!(elbow.friction_loss(), 0.0);
}

#[test]
fn massless_links_read_as_no_inertia() {
    let frames = model("massless_frames.urdf");
    assert_eq!(frames.body(0).unwrap().inertia().unwrap().mass(), 3.0);
    assert_eq!(frames.body(1).unwrap().inertia(), None);
    assert_eq!(frames.body(2).unwrap().inertia(), None);
}

#[test]
fn massless_links_still_build_a_tree() {
    let frames = model("massless_frames.urdf");
    let tree = frames.kinematic_tree::<8, 8>().unwrap();
    assert_eq!(tree.len(), 3);

    let state = tree.forward_kinematics(&Vector::zeros()).unwrap();
    assert_eq!(state.len(), 3);
    for slot in 0..3 {
        assert!(state.pose(slot).unwrap().translation().is_finite());
    }
}

#[test]
fn mimic_is_recorded() {
    let gripper = model("mimic_gripper.urdf");
    assert!(
        gripper
            .body_named("finger1")
            .unwrap()
            .joint()
            .unwrap()
            .mimic()
            .is_none()
    );

    let follower = gripper.body_named("finger2").unwrap().joint().unwrap();
    let mimic = follower.mimic().unwrap();
    assert_eq!(mimic.joint(), "finger_joint1");
    assert_eq!(mimic.multiplier(), -1.0);
    assert_eq!(mimic.offset(), 0.01);
}

#[test]
fn mimic_refuses_the_whole_tree() {
    let gripper = model("mimic_gripper.urdf");
    assert_eq!(
        gripper.kinematic_tree::<8, 8>().unwrap_err(),
        ModelError::MimicJointInTree {
            joint: "finger_joint2".to_owned(),
            follows: "finger_joint1".to_owned(),
        }
    );
}

#[test]
fn mimic_allows_a_clean_chain() {
    let gripper = model("mimic_gripper.urdf");

    // The chain down to the finger that moves on its own has no following joint on it.
    let chain = gripper.kinematic_tree_to::<8, 8>("finger1").unwrap();
    assert_eq!(chain.len(), 2);

    assert!(matches!(
        gripper.kinematic_tree_to::<8, 8>("finger2"),
        Err(ModelError::MimicJointInTree { .. })
    ));
}

#[test]
fn mimic_refuses_rust_source() {
    let gripper = model("mimic_gripper.urdf");
    assert!(matches!(
        gripper.to_rust_source(&RustSourceOptions::new("gripper")),
        Err(ModelError::MimicJointInTree { .. })
    ));
}

#[test]
fn floating_joint_is_refused() {
    // Whether a robot is bolted down or free to move is settled by whoever loads it, so a file
    // stating it is refused rather than read.
    assert_eq!(
        refusal("bad_floating_joint.urdf"),
        ModelError::FreeJointNotAtRoot {
            body: "base_link".to_owned(),
        }
    );
}

#[test]
fn passed_over_sections_are_listed() {
    assert_eq!(
        model("passed_over_sections.urdf").ignored(),
        [
            "gazebo".to_owned(),
            "material".to_owned(),
            "transmission".to_owned()
        ]
    );
}

#[test]
fn visual_and_collision_are_skipped() {
    // They sit inside a link rather than at the top of the file, so they are skipped without
    // being named — and no mesh file is ever looked for.
    let shapes = model("visual_and_collision.urdf");
    assert!(shapes.ignored().is_empty());
    assert_eq!(shapes.body_count(), 2);
    assert_eq!(shapes.body(0).unwrap().inertia().unwrap().mass(), 2.0);
}

#[test]
fn safety_controller_does_not_change_limits() {
    // The soft pair a controller is told to keep inside is passed over; the hard pair is read.
    let safe = model("safety_controller.urdf");
    let shoulder = safe.body(1).unwrap().joint().unwrap();
    assert_eq!(shoulder.limits(), Some((-3.0, 3.0)));
}

#[test]
fn a_planar_joint_is_refused() {
    assert_eq!(
        refusal("bad_planar_joint.urdf"),
        ModelError::UnsupportedJoint {
            body: "plate".to_owned(),
            joint_type: "planar".to_owned(),
        }
    );
}

#[test]
fn a_model_with_no_top_is_refused() {
    assert_eq!(refusal("bad_no_root.urdf"), ModelError::MissingRootLink);
}

#[test]
fn two_links_at_the_top_are_refused() {
    assert_eq!(
        refusal("bad_two_roots.urdf"),
        ModelError::MultipleRootLinks {
            names: vec!["first_base".to_owned(), "second_base".to_owned()],
        }
    );
}

#[test]
fn a_joint_naming_a_missing_link_is_refused() {
    assert_eq!(
        refusal("bad_unknown_link.urdf"),
        ModelError::UnknownLink {
            joint: "shoulder".to_owned(),
            link: "missing_link".to_owned(),
        }
    );
}

#[test]
fn a_link_on_two_joints_is_refused() {
    assert_eq!(
        refusal("bad_two_parents.urdf"),
        ModelError::LinkHasTwoParents {
            link: "tip".to_owned(),
            joints: vec!["also_from_the_arm".to_owned(), "from_the_base".to_owned()],
        }
    );
}

#[test]
fn a_loop_of_joints_is_refused() {
    assert_eq!(
        refusal("bad_cycle.urdf"),
        ModelError::CyclicLinkage {
            link: "ring_first".to_owned(),
        }
    );
}

#[test]
fn a_joint_that_can_stop_needs_a_range() {
    assert_eq!(
        refusal("bad_revolute_no_limit.urdf"),
        ModelError::JointNeedsLimit {
            joint: "shoulder".to_owned(),
        }
    );
}

#[test]
fn an_axis_pointing_nowhere_is_refused() {
    assert_eq!(
        refusal("bad_zero_axis.urdf"),
        ModelError::BadAttribute {
            element: "axis".to_owned(),
            attribute: "xyz".to_owned(),
            value: "0 0 0".to_owned(),
        }
    );
}

#[test]
fn a_link_stating_no_mass_at_all_is_refused() {
    // Leaving the block out says the link carries no mass; stating zero states something a body
    // cannot have.
    assert!(matches!(
        refusal("bad_zero_mass.urdf"),
        ModelError::Inertia(multicalc::error::SpatialError::NonPositiveMass)
    ));
}

#[test]
fn a_document_that_is_not_a_robot_is_refused() {
    assert_eq!(
        refusal("bad_not_a_robot.urdf"),
        ModelError::UnexpectedRootElement {
            found: "mujoco".to_owned(),
        }
    );
}

#[test]
fn load_path_dispatches_on_extension() {
    // A `.urdf` file goes to the URDF reader whatever it holds, which is why a MuJoCo document
    // under that name is refused for its root element rather than read.
    let read = multicalc_robot_model::load_path(&model_path("two_link_arm.urdf")).unwrap();
    assert_eq!(read.format(), ModelFormat::Urdf);
    assert!(matches!(
        multicalc_robot_model::load_path(&model_path("bad_not_a_robot.urdf")),
        Err(ModelError::UnexpectedRootElement { .. })
    ));
}

#[test]
fn load_str_dispatches_on_root_element() {
    let urdf = std::fs::read_to_string(model_path("two_link_arm.urdf")).unwrap();
    assert_eq!(
        multicalc_robot_model::load_str(&urdf).unwrap().format(),
        ModelFormat::Urdf
    );

    assert!(matches!(
        multicalc_robot_model::load_str("<sdf version=\"1.6\"><model/></sdf>"),
        Err(ModelError::UnexpectedRootElement { found }) if found == "sdf"
    ));
}

#[cfg(feature = "mjcf")]
#[test]
fn load_str_reads_a_mujoco_document_too() {
    let xml = r#"<mujoco>
                   <worldbody>
                     <body><freejoint/><inertial mass="1" diaginertia="1 1 1"/></body>
                   </worldbody>
                 </mujoco>"#;
    assert_eq!(
        multicalc_robot_model::load_str(xml).unwrap().format(),
        ModelFormat::Mjcf
    );
}

#[test]
fn rust_source_renders_a_urdf_model() {
    // Writing a model out as Rust source reads the same `RobotModel` whichever file it came from,
    // so nothing in the codegen knows about either format.
    let source = model("two_link_arm.urdf")
        .to_rust_source(&RustSourceOptions::new("two_link_arm"))
        .unwrap();

    assert!(source.contains("fn two_link_arm()"), "{source}");
    assert!(source.contains("Joint::revolute("), "{source}");
    assert!(source.contains("JointParent::World"), "{source}");
    assert!(source.contains("JointParent::Joint(0)"), "{source}");
    assert!(source.contains("KinematicTree<3, 3, f32>"), "{source}");
    assert!(source.contains("with_damping(0.25)"), "{source}");
}
