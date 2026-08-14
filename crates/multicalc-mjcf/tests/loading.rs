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

/// The parts of the file the loader did not read, as plain text for comparing against.
#[must_use]
fn ignored(body: &RigidBodyModel) -> Vec<&str> {
    body.ignored().iter().map(String::as_str).collect()
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
    let body = load(
        r#"<body><freejoint/><geom type="hfield" hfield="terrain" mass="0"/><geom type="box" size="1 1 1" mass="6"/></body>"#,
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
fn measures_a_cylinder() {
    // A cylinder does not spread its mass alike along its axis and across it, so unlike the shapes
    // above it takes a different fraction on different axes: a quarter of the square of the radius
    // either way across, a third of the square of the half-length along. About its own axis that
    // is m·r²/2, and across it m·(3r² + 4h²)/12.
    let body = load(r#"<body><freejoint/><geom type="cylinder" size="1 3" mass="12"/></body>"#);

    assert_eq!(body.inertia().mass(), 12.0);
    assert_eq!(diagonal(&body), [39.0, 39.0, 6.0]);
}

#[test]
fn measures_a_capsule() {
    // A capsule is a barrel of half-length 1 with a hemisphere of radius 1 on each end, so of its
    // 10 kg the barrel carries 6 and the caps 4. About the axis that is 6·r²/2 + 4·(2r²/5) = 4.6;
    // across it the caps also have to be carried a half-length out from the middle, giving
    // 6·(r²/4 + h²/3) + 4·(2r²/5 + h² + 3rh/4) = 12.1.
    let body = load(r#"<body><freejoint/><geom type="capsule" size="1 1" mass="10"/></body>"#);

    assert_eq!(body.inertia().mass(), 10.0);
    let [first, second, third] = diagonal(&body);
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
        let body = load(&format!(r#"<body><freejoint/>{geom}</body>"#));
        assert_golden(body.inertia().mass(), mass, geom);
        for (spin, want) in diagonal(&body).into_iter().zip(expected) {
            assert_golden(spin, want, geom);
        }
    }
}

#[test]
fn reads_a_shape_that_states_where_its_axis_starts_and_stops() {
    // A capsule 0.6 long about the z axis, written by its ends rather than as a half-length of 0.3
    // with a facing to go with it. Nothing about it is turned, so it is the plain measurement.
    let body = load(
        r#"<body><freejoint/><geom type="capsule" size="0.1" fromto="0 0 -0.3 0 0 0.3" mass="4"/></body>"#,
    );

    assert_golden(body.inertia().mass(), 4.0, "mass");
    assert_eq!(body.inertia().center_of_mass().into_array(), [0.0; 3]);
    for (spin, want) in diagonal(&body).into_iter().zip([
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
    let body = load(
        r#"<body><freejoint/><geom type="capsule" size="0.1" fromto="0 0 0 0.4 0 0.3" mass="4"/></body>"#,
    );

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
    let body = load(
        r#"<body><freejoint/><geom type="cylinder" size="0.05" fromto="1 1 1 2 3 5" mass="7"/></body>"#,
    );

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
    let body = load_str(
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

    assert_golden(body.inertia().mass(), 4.0, "mass");
    assert_golden(diagonal(&body)[2], 0.019_272_727_272_727_275, "about z");
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

#[test]
fn records_the_parts_of_a_file_it_does_not_read() {
    // Neither section changes a mass, so both are passed over — but the model that comes back says
    // so rather than leaving the caller to guess how much of the file was used.
    let body = load_str(
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

    assert_eq!(body.name(), "drone");
    assert_eq!(ignored(&body), ["actuator", "tendon"]);
}

#[test]
fn records_nothing_for_a_file_it_reads_whole() {
    let body = load(r#"<body><freejoint/><inertial mass="1" diaginertia="1 1 1"/></body>"#);
    assert!(ignored(&body).is_empty(), "{:?}", ignored(&body));
}
