#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Writing a model out as Rust source: what the emitted text contains, that it is stable across
//! runs, and that a floating-base model emits a `Joint::floating` push.

use multicalc_robot_model::mjcf::load_str;
use multicalc_robot_model::{GeneratedScalar, RustSourceOptions};

/// A body on a hinge, with the settings that show up literally in the emitted text.
const TWO_BODY_MODEL: &str = r#"<mujoco>
  <worldbody>
    <body name="base">
      <inertial mass="1" diaginertia="1 1 1"/>
      <body name="arm" pos="0 0 1">
        <joint axis="0 0 1" armature="0.1"/>
        <inertial mass="1" diaginertia="1 1 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>"#;

#[test]
fn emits_the_joint_settings_and_capacity() {
    let model = load_str(TWO_BODY_MODEL).unwrap();
    let options = RustSourceOptions::new("two_body_arm").with_scalar(GeneratedScalar::F32);
    let source = model.to_rust_source(&options).unwrap();

    assert!(source.contains("with_armature(0.1)"), "{source}");
    assert!(source.contains("JointParent::Joint(0)"), "{source}");
    assert!(source.contains("KinematicTree<2, 2, f32>"), "{source}");
}

#[test]
fn emitting_the_same_model_twice_gives_identical_text() {
    let model = load_str(TWO_BODY_MODEL).unwrap();
    let options = RustSourceOptions::new("two_body_arm");

    let first = model.to_rust_source(&options).unwrap();
    let second = model.to_rust_source(&options).unwrap();
    assert_eq!(first, second);
}

#[test]
fn emits_a_floating_base() {
    let model = load_str(
        r#"<mujoco><worldbody><body name="drone"><freejoint/><inertial mass="1" diaginertia="1 1 1"/></body></worldbody></mujoco>"#,
    )
    .unwrap();
    let options = RustSourceOptions::new("drone");
    let source = model.to_rust_source(&options).unwrap();

    assert!(source.contains("Joint::floating("), "{source}");
    assert!(source.contains("KinematicTree<1, 7, f32>"), "{source}");
}
