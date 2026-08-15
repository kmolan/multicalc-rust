#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Loading behaviour, driven by small hand-written models: what is read, what is worked out from
//! the shapes, and what is refused by name.

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

use multicalc::kinematics::JointKind;
use multicalc_mjcf::{MjcfError, RobotModel, load_str};

/// A model file holding whatever the case needs inside its `<worldbody>`.
#[must_use]
fn model(inner: &str) -> String {
    format!("<mujoco><worldbody>{inner}</worldbody></mujoco>")
}

#[must_use]
fn load(inner: &str) -> RobotModel {
    load_str(&model(inner)).unwrap()
}

#[must_use]
fn refuse(inner: &str) -> MjcfError {
    load_str(&model(inner)).unwrap_err()
}

/// The three numbers down the diagonal of how the first body resists being spun.
#[must_use]
fn diagonal(loaded: &RobotModel) -> [f64; 3] {
    let inertia = loaded.body(0).unwrap().inertia().rotational_inertia();
    [inertia[(0, 0)], inertia[(1, 1)], inertia[(2, 2)]]
}

/// The parts of the file the loader did not read, as plain text for comparing against.
#[must_use]
fn ignored(loaded: &RobotModel) -> Vec<&str> {
    loaded.ignored().iter().map(String::as_str).collect()
}

fn assert_close(actual: f64, expected: f64, label: &str) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "{label}: {actual} is not {expected}"
    );
}

/// Against a number MuJoCo itself produced. The two do not add their terms in the same order, so
/// the last bits can differ and the comparison is made relative to the size of the number.
fn assert_golden(actual: f64, expected: f64, label: &str) {
    let tolerance = 1e-12 * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance,
        "{label}: {actual} is not MuJoCo's {expected}"
    );
}

#[test]
fn reads_mass_properties_a_file_states() {
    let model = load(
        r#"<body name="drone"><freejoint/><inertial pos="0 0 0" mass="2" diaginertia="1 2 3"/></body>"#,
    );

    let body = model.body(0).unwrap();
    assert_eq!(body.name(), "drone");
    assert!(model.has_floating_base());
    assert_eq!(body.inertia().mass(), 2.0);
    assert_eq!(diagonal(&model), [1.0, 2.0, 3.0]);
}

#[test]
fn turns_stated_inertia_into_the_body_axes() {
    // The three numbers run along the axes of a frame given by `quat`. A quarter turn about z
    // swaps which of the body's own axes the first two describe.
    let model = load(
        r#"<body><freejoint/><inertial mass="2" diaginertia="1 2 3" quat="0.7071067811865476 0 0 0.7071067811865476"/></body>"#,
    );

    let [first, second, third] = diagonal(&model);
    assert_close(first, 2.0, "first");
    assert_close(second, 1.0, "second");
    assert_close(third, 3.0, "third");
}

#[test]
fn names_a_body_the_file_leaves_unnamed() {
    let model = load(r#"<body><freejoint/><inertial mass="1" diaginertia="1 1 1"/></body>"#);
    assert_eq!(model.body(0).unwrap().name(), "body");
}

#[test]
fn refuses_a_file_with_no_worldbody_or_no_bodies() {
    assert_eq!(
        load_str("<mujoco></mujoco>").unwrap_err(),
        MjcfError::MissingWorldbody
    );
    assert_eq!(refuse(""), MjcfError::NoBodies);
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
            r#"<body name="ground"><freejoint/><geom type="hfield" hfield="terrain" mass="1"/></body>"#
        ),
        MjcfError::UnsupportedGeomType {
            body: "ground".to_owned(),
            geom_type: "hfield".to_owned(),
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
    // The height field would be refused if it carried mass, so this also shows the order of the
    // checks.
    let model = load(
        r#"<body><freejoint/><geom type="hfield" hfield="terrain" mass="0"/><geom type="box" size="1 1 1" mass="6"/></body>"#,
    );
    assert_eq!(model.body(0).unwrap().inertia().mass(), 6.0);
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
fn refuses_an_include_from_text() {
    // `load_str` has no file to resolve an `<include>` against, so it is refused rather than
    // partly read. `load_path` follows the same include successfully; that is `includes.rs`.
    assert_eq!(
        load_str(r#"<mujoco><worldbody><include file="other.xml"/></worldbody></mujoco>"#)
            .unwrap_err(),
        MjcfError::IncludeNeedsFile
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
    let model = load(r#"<body><freejoint/><geom type="box" size="1 2 3" mass="6"/></body>"#);

    assert_eq!(model.body(0).unwrap().inertia().mass(), 6.0);
    assert_eq!(diagonal(&model), [26.0, 20.0, 10.0]);
}

#[test]
fn measures_a_sphere() {
    // A sphere reaches the same distance every way, so it is as hard to spin about one axis as
    // another: two fifths of its mass times the square of its radius.
    let model = load(r#"<body><freejoint/><geom type="sphere" size="2" mass="10"/></body>"#);

    assert_eq!(model.body(0).unwrap().inertia().mass(), 10.0);
    assert_eq!(diagonal(&model), [16.0, 16.0, 16.0]);
}

#[test]
fn reads_a_shape_that_names_no_form_as_a_sphere() {
    // A shape that says nothing about its form or its mass is a sphere of the standard density,
    // which is what MuJoCo assumes. Its mass is that density times the room it takes up.
    let model = load(r#"<body><freejoint/><geom size="1"/></body>"#);

    let expected_mass = 1000.0 * 4.0 / 3.0 * std::f64::consts::PI;
    assert!((model.body(0).unwrap().inertia().mass() - expected_mass).abs() < 1e-9);

    let [first, second, third] = diagonal(&model);
    for spin in [first, second, third] {
        assert!((spin - 0.4 * expected_mass).abs() < 1e-9, "{spin}");
    }
}

#[test]
fn measures_an_ellipsoid() {
    // The same pattern, with a fifth of the mass rather than a third.
    let model = load(r#"<body><freejoint/><geom type="ellipsoid" size="1 2 3" mass="5"/></body>"#);

    assert_eq!(model.body(0).unwrap().inertia().mass(), 5.0);
    assert_eq!(diagonal(&model), [13.0, 10.0, 5.0]);
}

#[test]
fn measures_a_cylinder() {
    // A cylinder does not spread its mass alike along its axis and across it, so unlike the shapes
    // above it takes a different fraction on different axes: a quarter of the square of the radius
    // either way across, a third of the square of the half-length along. About its own axis that
    // is m·r²/2, and across it m·(3r² + 4h²)/12.
    let model = load(r#"<body><freejoint/><geom type="cylinder" size="1 3" mass="12"/></body>"#);

    assert_eq!(model.body(0).unwrap().inertia().mass(), 12.0);
    assert_eq!(diagonal(&model), [39.0, 39.0, 6.0]);
}

#[test]
fn measures_a_capsule() {
    // A capsule is a barrel of half-length 1 with a hemisphere of radius 1 on each end, so of its
    // 10 kg the barrel carries 6 and the caps 4. About the axis that is 6·r²/2 + 4·(2r²/5) = 4.6;
    // across it the caps also have to be carried a half-length out from the middle, giving
    // 6·(r²/4 + h²/3) + 4·(2r²/5 + h² + 3rh/4) = 12.1.
    let model = load(r#"<body><freejoint/><geom type="capsule" size="1 1" mass="10"/></body>"#);

    assert_eq!(model.body(0).unwrap().inertia().mass(), 10.0);
    let [first, second, third] = diagonal(&model);
    assert_golden(first, 12.1, "first");
    assert_golden(second, 12.1, "second");
    assert_golden(third, 4.6, "third");
}

/// Every number below is `mujoco.MjModel.from_xml_string` compiling the same geom, read out of
/// `body_mass` and `body_inertia` (MuJoCo 3.11.0). They are the point of the exercise: the closed
/// forms are only worth having if they agree with what the simulator these files are written for
/// works the same shape out to be.
#[test]
fn measures_capsules_and_cylinders_as_mujoco_compiles_them() {
    let cases: [(&str, f64, [f64; 3]); 4] = [
        (
            r#"<geom type="capsule" size="0.1 0.5" mass="4"/>"#,
            4.0,
            [
                0.440_117_647_058_823_5,
                0.440_117_647_058_823_5,
                0.019_529_411_764_705_885,
            ],
        ),
        // A shape stating no mass is the standard density times the room it takes up, so these
        // also pin the volumes: 2πr²h for the barrel, and 4πr³/3 for the two caps together.
        (
            r#"<geom type="cylinder" size="0.5 2"/>"#,
            3_141.592_653_589_793,
            [
                4_385.139_745_635_753,
                4_385.139_745_635_753,
                392.699_081_698_724_1,
            ],
        ),
        (
            r#"<geom type="capsule" size="0.5 2"/>"#,
            3_665.191_429_188_092,
            [
                6_924.593_807_287_501_5,
                6_924.593_807_287_501_5,
                445.058_959_258_554_07,
            ],
        ),
        // Almost all cap and hardly any barrel, which is a ball of radius 2 give or take a
        // nanometre. That the answer arrives at the sphere's own 16 is what says the caps are
        // being carried out to where they sit by the right amount.
        (
            r#"<geom type="capsule" size="2 1e-9" mass="10"/>"#,
            10.0,
            [
                16.000_000_010_500_003,
                16.000_000_010_500_003,
                16.000_000_003,
            ],
        ),
    ];

    for (geom, mass, expected) in cases {
        let model = load(&format!(r#"<body><freejoint/>{geom}</body>"#));
        assert_golden(model.body(0).unwrap().inertia().mass(), mass, geom);
        for (spin, want) in diagonal(&model).into_iter().zip(expected) {
            assert_golden(spin, want, geom);
        }
    }
}

#[test]
fn reads_a_shape_that_states_where_its_axis_starts_and_stops() {
    // A capsule 0.6 long about the z axis, written by its ends rather than as a half-length of 0.3
    // with a facing to go with it. Nothing about it is turned, so it is the plain measurement.
    let model = load(
        r#"<body><freejoint/><geom type="capsule" size="0.1" fromto="0 0 -0.3 0 0 0.3" mass="4"/></body>"#,
    );

    let body = model.body(0).unwrap();
    assert_golden(body.inertia().mass(), 4.0, "mass");
    assert_eq!(body.inertia().center_of_mass().into_array(), [0.0; 3]);
    for (spin, want) in diagonal(&model).into_iter().zip([
        0.191_090_909_090_909_1,
        0.191_090_909_090_909_1,
        0.019_272_727_272_727_275,
    ]) {
        assert_golden(spin, want, "on the z axis");
    }
}

/// The ends carry a facing as well as a length, and a shape lying across two axes at once is where
/// that has to be right: it is no longer hard to spin about the body's own axes, and the corner
/// terms are what say so. Both goldens are MuJoCo's compile of the same geom, its `body_inertia`
/// turned back into the body's axes through `body_iquat`.
#[test]
fn turns_a_shape_onto_the_line_between_its_ends() {
    let model = load(
        r#"<body><freejoint/><geom type="capsule" size="0.1" fromto="0 0 0 0.4 0 0.3" mass="4"/></body>"#,
    );

    let body = model.body(0).unwrap();
    assert_golden(body.inertia().mass(), 4.0, "mass");
    // Halfway along the line between the two ends.
    for (place, want) in body
        .inertia()
        .center_of_mass()
        .into_array()
        .into_iter()
        .zip([0.2, 0.0, 0.15])
    {
        assert_golden(place, want, "where it sits");
    }

    let expected = [
        [0.064_631_578_947_368_4, 0.0, -0.060_631_578_947_368_41],
        [0.0, 0.145_473_684_210_526_3, 0.0],
        [-0.060_631_578_947_368_41, 0.0, 0.100_000_000_000_000_03],
    ];
    let inertia = body.inertia().rotational_inertia();
    for (row, wanted) in expected.into_iter().enumerate() {
        for (column, want) in wanted.into_iter().enumerate() {
            assert_golden(inertia[(row, column)], want, "against MuJoCo");
        }
    }
}

#[test]
fn measures_a_cylinder_stated_by_its_ends() {
    // Its axis runs along none of the body's, and both ends sit away from the origin, so this
    // leans on the length, the placement and the facing all at once.
    let model = load(
        r#"<body><freejoint/><geom type="cylinder" size="0.05" fromto="1 1 1 2 3 5" mass="7"/></body>"#,
    );

    let body = model.body(0).unwrap();
    assert_golden(body.inertia().mass(), 7.0, "mass");
    for (place, want) in body
        .inertia()
        .center_of_mass()
        .into_array()
        .into_iter()
        .zip([1.5, 2.0, 3.0])
    {
        assert_golden(place, want, "where it sits");
    }

    let expected = [
        [
            11.671_25,
            -1.166_250_000_000_001_3,
            -2.332_499_999_999_999_6,
        ],
        [-1.166_250_000_000_001_1, 9.921_875, -4.664_999_999_999_999],
        [-2.332_5, -4.665, 2.924_374_999_999_999_5],
    ];
    let inertia = body.inertia().rotational_inertia();
    for (row, wanted) in expected.into_iter().enumerate() {
        for (column, want) in wanted.into_iter().enumerate() {
            assert_golden(inertia[(row, column)], want, "against MuJoCo");
        }
    }
}

#[test]
fn takes_the_ends_of_an_axis_down_a_default_block() {
    // MuJoCo lets `fromto` be inherited like any other geom setting, so a model can state a link's
    // ends in a class and name the class on the shape. Read off the element alone it would be
    // missed, and the shape would look like a capsule with no length at all.
    let model = load_str(
        r#"<mujoco>
             <default>
               <default class="link">
                 <geom type="capsule" size="0.1" fromto="0 0 -0.3 0 0 0.3"/>
               </default>
             </default>
             <worldbody>
               <body><freejoint/><geom class="link" mass="4"/></body>
             </worldbody>
           </mujoco>"#,
    )
    .unwrap();

    assert_golden(model.body(0).unwrap().inertia().mass(), 4.0, "mass");
    assert_golden(diagonal(&model)[2], 0.019_272_727_272_727_275, "about z");
}

#[test]
fn refuses_ends_that_say_nothing_the_loader_can_use() {
    // Both ends in one place pin down no direction, and no length either.
    assert_eq!(
        refuse(
            r#"<body><freejoint/><geom type="capsule" size="0.1" fromto="1 2 3 1 2 3" mass="4"/></body>"#
        ),
        MjcfError::BadAttribute {
            element: "geom".to_owned(),
            attribute: "fromto".to_owned(),
            value: "1 2 3 1 2 3".to_owned(),
        }
    );

    // Ends and a position are two answers to where the shape sits, and they need not agree.
    assert_eq!(
        refuse(
            r#"<body name="arm"><freejoint/><geom type="capsule" size="0.1" fromto="0 0 0 0 0 1" pos="9 9 9" mass="4"/></body>"#
        ),
        MjcfError::ConflictingPlacement {
            body: "arm".to_owned(),
        }
    );

    // MuJoCo reads ends on boxes and ellipsoids too, by a rule this loader has not checked against
    // the compiler, so they are refused by name rather than guessed at.
    assert_eq!(
        refuse(
            r#"<body name="link"><freejoint/><geom type="box" size="0.1 0.2" fromto="0 0 0 0 0 1" mass="4"/></body>"#
        ),
        MjcfError::UnsupportedFromTo {
            body: "link".to_owned(),
            geom_type: "box".to_owned(),
        }
    );
}

#[test]
fn refuses_a_capsule_or_cylinder_that_is_not_sized_by_a_radius_and_a_half_length() {
    // Both take exactly two numbers. One is what a geom written with `fromto` leaves behind, and
    // three is a box's size on the wrong shape; neither can be measured, and guessing at either
    // would put a wrong mass into the model.
    for size in ["0.1", "0.1 0.2 0.3", ""] {
        for form in ["capsule", "cylinder"] {
            assert_eq!(
                refuse(&format!(
                    r#"<body><freejoint/><geom type="{form}" size="{size}" mass="1"/></body>"#
                )),
                MjcfError::BadAttribute {
                    element: "geom".to_owned(),
                    attribute: "size".to_owned(),
                    value: size.to_owned(),
                },
                "{form} sized {size:?}"
            );
        }
    }
}

#[test]
fn combines_the_shapes_a_body_is_built_from() {
    // Two unit boxes a metre either side of the origin. They balance at the origin, each keeps its
    // own 2/3 about every axis, and moving the reference point a metre along x adds the mass times
    // one to the two axes across from that move but nothing to x itself.
    let model = load(
        r#"<body><freejoint/><geom type="box" size="1 1 1" pos="-1 0 0" mass="1"/><geom type="box" size="1 1 1" pos="1 0 0" mass="1"/></body>"#,
    );

    let body = model.body(0).unwrap();
    assert_eq!(body.inertia().mass(), 2.0);
    assert_eq!(body.inertia().center_of_mass().into_array(), [0.0; 3]);

    let [first, second, third] = diagonal(&model);
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
    let model = load_str(
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

    assert_eq!(model.body(0).unwrap().inertia().mass(), 5.0);
    assert_eq!(diagonal(&model), [2.0, 2.0, 2.0]);
}

#[test]
fn records_the_parts_of_a_file_it_does_not_read() {
    // Neither section changes a mass, so both are passed over — but the model that comes back says
    // so rather than leaving the caller to guess how much of the file was used.
    let model = load_str(
        r#"<mujoco>
             <worldbody>
               <body name="drone">
                 <freejoint/>
                 <inertial mass="1" diaginertia="1 1 1"/>
               </body>
             </worldbody>
             <tendon>
               <spatial name="cable"/>
             </tendon>
             <actuator>
               <motor name="thrust" gear="0 0 1 0 0 0"/>
             </actuator>
           </mujoco>"#,
    )
    .unwrap();

    assert_eq!(model.body(0).unwrap().name(), "drone");
    assert_eq!(ignored(&model), ["actuator", "tendon"]);
}

#[test]
fn records_nothing_for_a_file_it_reads_whole() {
    let model = load(r#"<body><freejoint/><inertial mass="1" diaginertia="1 1 1"/></body>"#);
    assert!(ignored(&model).is_empty(), "{:?}", ignored(&model));
}

#[test]
fn reads_a_chain_of_bodies() {
    let model = load(
        r#"<body><inertial mass="1" diaginertia="1 1 1"/>
             <body pos="0 0 1"><joint axis="0 1 0"/><inertial mass="1" diaginertia="1 1 1"/></body>
           </body>"#,
    );

    assert_eq!(model.body_count(), 2);
    assert_eq!(model.body(1).unwrap().parent(), Some(0));

    let joint = model.body(1).unwrap().joint().unwrap();
    assert_eq!(joint.kind(), JointKind::Continuous);
    assert_eq!(joint.axis().into_array(), [0.0, 1.0, 0.0]);
}

#[test]
fn an_explicitly_unlimited_hinge_is_continuous() {
    let model =
        load(r#"<body><joint limited="false"/><inertial mass="1" diaginertia="1 1 1"/></body>"#);
    assert_eq!(
        model.body(0).unwrap().joint().unwrap().kind(),
        JointKind::Continuous
    );
}

#[test]
fn a_limited_hinge_stays_revolute() {
    let model = load(
        r#"<body><joint limited="true" range="-1 1"/><inertial mass="1" diaginertia="1 1 1"/></body>"#,
    );
    assert_eq!(
        model.body(0).unwrap().joint().unwrap().kind(),
        JointKind::Revolute
    );
}

#[test]
fn a_body_without_a_joint_is_welded() {
    let model = load(r#"<body><inertial mass="1" diaginertia="1 1 1"/></body>"#);
    assert!(model.body(0).unwrap().joint().is_none());

    let tree = model.kinematic_tree::<2, 2>().unwrap();
    assert_eq!(tree.joint(0).unwrap().kind(), JointKind::Fixed);
}

#[test]
fn reads_joint_data_a_class_supplies() {
    let model = load_str(
        r#"<mujoco>
             <compiler angle="radian"/>
             <default>
               <default class="arm">
                 <joint armature="0.1" damping="1" frictionloss="0.2" ref="0.3"
                        springref="0.4" stiffness="5" range="-1 1"/>
               </default>
             </default>
             <worldbody>
               <body childclass="arm">
                 <inertial mass="1" diaginertia="1 1 1"/>
                 <body pos="0 0 1"><joint/><inertial mass="1" diaginertia="1 1 1"/></body>
               </body>
             </worldbody>
           </mujoco>"#,
    )
    .unwrap();

    let joint = model.body(1).unwrap().joint().unwrap();
    assert_eq!(joint.armature(), 0.1);
    assert_eq!(joint.damping(), 1.0);
    assert_eq!(joint.friction_loss(), 0.2);
    assert_eq!(joint.zero_offset(), 0.3);
    assert_eq!(joint.spring_reference(), 0.4);
    assert_eq!(joint.spring_stiffness(), 5.0);
    assert_eq!(joint.limits(), Some((-1.0, 1.0)));
}

#[test]
fn a_child_class_reaches_every_body_below_it() {
    // `childclass` is stated once, on the outermost body, and the joint that reads it sits two
    // levels further down.
    let model = load_str(
        r#"<mujoco>
             <compiler angle="radian"/>
             <default>
               <default class="arm">
                 <joint armature="0.1" damping="1"/>
               </default>
             </default>
             <worldbody>
               <body childclass="arm">
                 <inertial mass="1" diaginertia="1 1 1"/>
                 <body pos="0 0 1">
                   <inertial mass="1" diaginertia="1 1 1"/>
                   <body pos="0 0 1"><joint/><inertial mass="1" diaginertia="1 1 1"/></body>
                 </body>
               </body>
             </worldbody>
           </mujoco>"#,
    )
    .unwrap();

    let joint = model.body(2).unwrap().joint().unwrap();
    assert_eq!(joint.armature(), 0.1);
    assert_eq!(joint.damping(), 1.0);
}

#[test]
fn angles_are_degrees_unless_the_file_says_otherwise() {
    let model = load_str(
        r#"<mujoco>
             <compiler angle="degree"/>
             <worldbody>
               <body><inertial mass="1" diaginertia="1 1 1"/>
                 <body pos="0 0 1"><joint range="-90 90" ref="45"/><inertial mass="1" diaginertia="1 1 1"/></body>
               </body>
             </worldbody>
           </mujoco>"#,
    )
    .unwrap();

    let joint = model.body(1).unwrap().joint().unwrap();
    let (lower, upper) = joint.limits().unwrap();
    assert_close(lower, -FRAC_PI_2, "lower limit");
    assert_close(upper, FRAC_PI_2, "upper limit");
    assert_close(joint.zero_offset(), FRAC_PI_4, "zero offset");
}

#[test]
fn a_sliding_joint_keeps_its_range_in_metres() {
    let model = load_str(
        r#"<mujoco>
             <compiler angle="degree"/>
             <worldbody>
               <body><inertial mass="1" diaginertia="1 1 1"/>
                 <body pos="0 0 1"><joint type="slide" range="0 0.04"/><inertial mass="1" diaginertia="1 1 1"/></body>
               </body>
             </worldbody>
           </mujoco>"#,
    )
    .unwrap();

    let joint = model.body(1).unwrap().joint().unwrap();
    assert_eq!(joint.kind(), JointKind::Prismatic);
    assert_eq!(joint.limits(), Some((0.0, 0.04)));
}

#[test]
fn reads_a_full_inertia_tensor() {
    let model = load(r#"<body><inertial mass="2" fullinertia="1 2 3 0.1 0.2 0.3"/></body>"#);

    let expected = [[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]];
    let inertia = model.body(0).unwrap().inertia().rotational_inertia();
    for (row, wanted) in expected.into_iter().enumerate() {
        for (column, want) in wanted.into_iter().enumerate() {
            assert_close(inertia[(row, column)], want, "fullinertia entry");
        }
    }
}

#[test]
fn refuses_a_ball_joint() {
    assert_eq!(
        refuse(
            r#"<body name="arm"><joint type="ball"/><inertial mass="1" diaginertia="1 1 1"/></body>"#
        ),
        MjcfError::UnsupportedJoint {
            body: "arm".to_owned(),
            joint_type: "ball".to_owned(),
        }
    );
}

#[test]
fn refuses_two_joints_on_one_body() {
    assert_eq!(
        refuse(
            r#"<body name="arm"><joint/><joint/><inertial mass="1" diaginertia="1 1 1"/></body>"#
        ),
        MjcfError::MultipleJoints {
            body: "arm".to_owned(),
            count: 2,
        }
    );
}

#[test]
fn refuses_a_free_joint_below_the_top() {
    assert_eq!(
        refuse(
            r#"<body><inertial mass="1" diaginertia="1 1 1"/>
                 <body name="forearm" pos="0 0 1"><freejoint/><inertial mass="1" diaginertia="1 1 1"/></body>
               </body>"#
        ),
        MjcfError::FreeJointNotAtRoot {
            body: "forearm".to_owned(),
        }
    );
}

#[test]
fn refuses_a_turn_written_as_angles() {
    assert_eq!(
        refuse(r#"<body euler="0 0 1"><inertial mass="1" diaginertia="1 1 1"/></body>"#),
        MjcfError::UnsupportedOrientation {
            element: "body".to_owned(),
            attribute: "euler".to_owned(),
        }
    );
}

#[test]
fn refuses_a_limited_joint_with_no_range() {
    assert_eq!(
        refuse(
            r#"<body name="arm"><joint limited="true"/><inertial mass="1" diaginertia="1 1 1"/></body>"#
        ),
        MjcfError::LimitsNeedRange {
            body: "arm".to_owned(),
        }
    );
}

#[test]
fn refuses_a_tip_the_model_does_not_have() {
    let model = load(r#"<body><inertial mass="1" diaginertia="1 1 1"/></body>"#);
    assert_eq!(
        model.kinematic_tree_to::<4, 4>("gripper").unwrap_err(),
        MjcfError::UnknownBody {
            name: "gripper".to_owned(),
        }
    );
}

#[test]
fn refuses_a_model_too_big_for_the_tree() {
    let model = load(
        r#"<body><inertial mass="1" diaginertia="1 1 1"/>
             <body pos="0 0 1"><inertial mass="1" diaginertia="1 1 1"/>
               <body pos="0 0 1"><inertial mass="1" diaginertia="1 1 1"/></body>
             </body>
           </body>"#,
    );
    assert_eq!(
        model.kinematic_tree::<2, 2>().unwrap_err(),
        MjcfError::TreeCapacityExceeded {
            needed: 3,
            capacity: 2,
        }
    );
}

#[test]
fn builds_a_floating_base_as_a_tree() {
    let model =
        load(r#"<body name="drone"><freejoint/><inertial mass="1" diaginertia="1 1 1"/></body>"#);
    let tree = model.kinematic_tree::<1, 7>().unwrap();
    assert_eq!(tree.joint(0).unwrap().kind(), JointKind::Floating);
}
