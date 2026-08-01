#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Loading behaviour, driven by small hand-written models: what is read, what is worked out from
//! the shapes, and what is refused by name.

use multicalc_mjcf::{MjcfError, RigidBodyModel, load_str};

/// A model file holding whatever the case needs inside its `<worldbody>`.
#[must_use]
fn model(inner: &str) -> String {
    format!("<mujoco><worldbody>{inner}</worldbody></mujoco>")
}

#[must_use]
fn load(inner: &str) -> RigidBodyModel {
    load_str(&model(inner)).unwrap()
}

#[must_use]
fn refuse(inner: &str) -> MjcfError {
    load_str(&model(inner)).unwrap_err()
}

/// The three numbers down the diagonal of how the body resists being spun.
#[must_use]
fn diagonal(body: &RigidBodyModel) -> [f64; 3] {
    let inertia = body.inertia().rotational_inertia();
    [inertia[(0, 0)], inertia[(1, 1)], inertia[(2, 2)]]
}

fn assert_close(actual: f64, expected: f64, label: &str) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "{label}: {actual} is not {expected}"
    );
}

#[test]
fn reads_mass_properties_a_file_states() {
    let body = load(
        r#"<body name="drone"><freejoint/><inertial pos="0 0 0" mass="2" diaginertia="1 2 3"/></body>"#,
    );

    assert_eq!(body.name(), "drone");
    assert!(body.has_free_joint());
    assert_eq!(body.inertia().mass(), 2.0);
    assert_eq!(diagonal(&body), [1.0, 2.0, 3.0]);
}

#[test]
fn turns_stated_inertia_into_the_body_axes() {
    // The three numbers run along the axes of a frame given by `quat`. A quarter turn about z
    // swaps which of the body's own axes the first two describe.
    let body = load(
        r#"<body><freejoint/><inertial mass="2" diaginertia="1 2 3" quat="0.7071067811865476 0 0 0.7071067811865476"/></body>"#,
    );

    let [first, second, third] = diagonal(&body);
    assert_close(first, 2.0, "first");
    assert_close(second, 1.0, "second");
    assert_close(third, 3.0, "third");
}

#[test]
fn names_a_body_the_file_leaves_unnamed() {
    let body = load(r#"<body><freejoint/><inertial mass="1" diaginertia="1 1 1"/></body>"#);
    assert_eq!(body.name(), "body");
}

#[test]
fn refuses_a_file_that_does_not_hold_exactly_one_body() {
    assert_eq!(
        load_str("<mujoco></mujoco>").unwrap_err(),
        MjcfError::MissingWorldbody
    );
    assert_eq!(refuse(""), MjcfError::NoBodies);

    let one = r#"<body><freejoint/><inertial mass="1" diaginertia="1 1 1"/></body>"#;
    assert_eq!(
        refuse(&format!("{one}{one}")),
        MjcfError::MultipleBodies { count: 2 }
    );
}

#[test]
fn refuses_a_joint_that_is_not_free() {
    assert_eq!(
        refuse(r#"<body name="arm"><joint type="hinge" axis="0 0 1"/></body>"#),
        MjcfError::UnsupportedJoint {
            body: "arm".to_owned(),
            joint_type: "hinge".to_owned(),
        }
    );

    // A joint that names no type is a hinge as far as MuJoCo is concerned, and is refused as one.
    assert_eq!(
        refuse(r#"<body name="arm"><joint axis="0 0 1"/></body>"#),
        MjcfError::UnsupportedJoint {
            body: "arm".to_owned(),
            joint_type: "hinge".to_owned(),
        }
    );
}

#[test]
fn refuses_a_body_with_no_joint_at_all() {
    assert_eq!(
        refuse(r#"<body name="fixed"><inertial mass="1" diaginertia="1 1 1"/></body>"#),
        MjcfError::MissingFreeJoint {
            body: "fixed".to_owned(),
        }
    );
}

#[test]
fn refuses_a_body_with_nothing_to_weigh() {
    assert_eq!(
        refuse(r#"<body name="empty"><freejoint/></body>"#),
        MjcfError::NoInertiaSource {
            body: "empty".to_owned(),
        }
    );
}

#[test]
fn refuses_a_shape_carrying_mass_it_cannot_measure() {
    assert_eq!(
        refuse(
            r#"<body name="pill"><freejoint/><geom type="capsule" size="0.1 0.4" mass="1"/></body>"#
        ),
        MjcfError::UnsupportedGeomType {
            body: "pill".to_owned(),
            geom_type: "capsule".to_owned(),
        }
    );

    assert_eq!(
        refuse(r#"<body name="hull"><freejoint/><geom type="mesh" mesh="x" mass="1"/></body>"#),
        MjcfError::MeshInertiaUnsupported {
            body: "hull".to_owned(),
        }
    );
}

#[test]
fn skips_a_shape_that_carries_no_mass_before_looking_at_its_form() {
    // The capsule would be refused if it carried mass, so this also shows the order of the checks.
    let body = load(
        r#"<body><freejoint/><geom type="capsule" size="0.1 0.4" mass="0"/><geom type="box" size="1 1 1" mass="6"/></body>"#,
    );
    assert_eq!(body.inertia().mass(), 6.0);
}

#[test]
fn refuses_an_attribute_that_does_not_hold_numbers() {
    assert_eq!(
        refuse(r#"<body><freejoint/><geom type="box" size="not a number" mass="1"/></body>"#),
        MjcfError::BadAttribute {
            element: "geom".to_owned(),
            attribute: "size".to_owned(),
            value: "not a number".to_owned(),
        }
    );
}

#[test]
fn refuses_a_class_the_file_never_defines() {
    assert_eq!(
        refuse(r#"<body><freejoint/><geom class="nowhere" mass="1"/></body>"#),
        MjcfError::UndefinedClass {
            name: "nowhere".to_owned(),
        }
    );
}

#[test]
fn refuses_a_file_that_pulls_in_another_file() {
    assert_eq!(
        load_str(r#"<mujoco><include file="other.xml"/><worldbody/></mujoco>"#).unwrap_err(),
        MjcfError::IncludeUnsupported
    );
}

#[test]
fn refuses_text_that_is_not_well_formed_xml() {
    assert!(matches!(load_str("<mujoco>"), Err(MjcfError::Xml(_))));
}

#[test]
fn refuses_stated_mass_properties_that_do_not_describe_a_body() {
    assert!(matches!(
        load_str(&model(
            r#"<body><freejoint/><inertial mass="0" diaginertia="1 1 1"/></body>"#
        )),
        Err(MjcfError::Inertia(_))
    ));
}

#[test]
fn measures_a_box() {
    // A box resists being spun about each axis by a third of its mass times the squares of the
    // two half-widths across from that axis.
    let body = load(r#"<body><freejoint/><geom type="box" size="1 2 3" mass="6"/></body>"#);

    assert_eq!(body.inertia().mass(), 6.0);
    assert_eq!(diagonal(&body), [26.0, 20.0, 10.0]);
}

#[test]
fn measures_a_sphere() {
    // A sphere reaches the same distance every way, so it is as hard to spin about one axis as
    // another: two fifths of its mass times the square of its radius.
    let body = load(r#"<body><freejoint/><geom type="sphere" size="2" mass="10"/></body>"#);

    assert_eq!(body.inertia().mass(), 10.0);
    assert_eq!(diagonal(&body), [16.0, 16.0, 16.0]);
}

#[test]
fn reads_a_shape_that_names_no_form_as_a_sphere() {
    // A shape that says nothing about its form or its mass is a sphere of the standard density,
    // which is what MuJoCo assumes. Its mass is that density times the room it takes up.
    let body = load(r#"<body><freejoint/><geom size="1"/></body>"#);

    let expected_mass = 1000.0 * 4.0 / 3.0 * std::f64::consts::PI;
    assert!((body.inertia().mass() - expected_mass).abs() < 1e-9);

    let [first, second, third] = diagonal(&body);
    for spin in [first, second, third] {
        assert!((spin - 0.4 * expected_mass).abs() < 1e-9, "{spin}");
    }
}

#[test]
fn measures_an_ellipsoid() {
    // The same pattern, with a fifth of the mass rather than a third.
    let body = load(r#"<body><freejoint/><geom type="ellipsoid" size="1 2 3" mass="5"/></body>"#);

    assert_eq!(body.inertia().mass(), 5.0);
    assert_eq!(diagonal(&body), [13.0, 10.0, 5.0]);
}

#[test]
fn combines_the_shapes_a_body_is_built_from() {
    // Two unit boxes a metre either side of the origin. They balance at the origin, each keeps its
    // own 2/3 about every axis, and moving the reference point a metre along x adds the mass times
    // one to the two axes across from that move but nothing to x itself.
    let body = load(
        r#"<body><freejoint/><geom type="box" size="1 1 1" pos="-1 0 0" mass="1"/><geom type="box" size="1 1 1" pos="1 0 0" mass="1"/></body>"#,
    );

    assert_eq!(body.inertia().mass(), 2.0);
    assert_eq!(body.inertia().center_of_mass().into_array(), [0.0; 3]);

    let [first, second, third] = diagonal(&body);
    assert_close(first, 4.0 / 3.0, "first");
    assert_close(second, 10.0 / 3.0, "second");
    assert_close(third, 10.0 / 3.0, "third");

    let inertia = body.inertia().rotational_inertia();
    for (row, column) in [(0, 1), (0, 2), (1, 2)] {
        assert_close(inertia[(row, column)], 0.0, "off the diagonal");
    }
}

#[test]
fn inherits_settings_through_nested_default_blocks() {
    // The shape names only the inner class and its own mass, so its form and size have to come
    // down the block chain, and the mass it states has to beat the zero the outer block sets.
    let body = load_str(
        r#"<mujoco>
             <default>
               <default class="outer">
                 <geom mass="0"/>
                 <default class="inner">
                   <geom type="ellipsoid" size="1 1 1"/>
                 </default>
               </default>
             </default>
             <worldbody>
               <body childclass="outer">
                 <freejoint/>
                 <geom class="inner" mass="5"/>
               </body>
             </worldbody>
           </mujoco>"#,
    )
    .unwrap();

    assert_eq!(body.inertia().mass(), 5.0);
    assert_eq!(diagonal(&body), [2.0, 2.0, 2.0]);
}
