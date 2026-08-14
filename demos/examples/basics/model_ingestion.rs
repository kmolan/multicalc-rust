//! Loads a MuJoCo model file and reports the body's mass, balance point, and how hard it is to
//! spin, along with the free joint's state layout.
//!
//! Run with: `cargo run -p multicalc-demos --example model_ingestion`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::Path;

use multicalc::spatial::{FreeJointState, Twist};

fn report(label: &str, value: f64, exact: f64) {
    assert!((value - exact).abs() < 1e-9, "{label}: |err| too large");
    println!(
        "  {label:<22} = {value:>12.8}   (exact {exact:>12.8}, |err| {:.0e})",
        (value - exact).abs()
    );
}

fn main() {
    // (1) Read the model. The file ships with this repository under its own upstream licence.
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../third_party/menagerie/skydio_x2/x2.xml");
    let model = multicalc_mjcf::load_path(&path).unwrap();
    let body = model.body_named("x2").unwrap();

    println!("Model: {}", model.name());
    assert!(
        model.has_floating_base(),
        "the body should hang off the world by a free joint"
    );
    println!("  free joint             = yes");
    let position = body.pose().translation();
    println!(
        "  sits at                  ({:.3}, {:.3}, {:.3}) m",
        position[0], position[1], position[2]
    );

    // (2) The file states no mass of its own — every number below is worked out from the shapes
    // the body is built from: four rotor discs and a hull.
    let inertia = body.inertia();
    println!("\nMass, worked out from the shapes");
    report("mass (kg)", inertia.mass(), 1.325);

    let center_of_mass = inertia.center_of_mass();
    println!("\nWhere it balances (m)");
    report("x", center_of_mass[0], 0.0);
    report("y", center_of_mass[1], 0.0);
    report("z", center_of_mass[2], 0.053962264150943406);

    // (3) How hard it is to spin, about the point it balances at. The front rotors sit three
    // centimetres above the rear pair, so the body does not spin cleanly about its own axes and
    // the two corner terms are not zero.
    let spin = inertia.rotational_inertia();
    println!("\nResistance to being spun (kg·m²)");
    for row in 0..3 {
        println!(
            "  [{:>15.12} {:>15.12} {:>15.12}]",
            spin[(row, 0)],
            spin[(row, 1)],
            spin[(row, 2)]
        );
    }
    report("corner term (0, 2)", spin[(0, 2)], -0.0021);
    report("corner term (2, 0)", spin[(2, 0)], -0.0021);

    // (4) The free joint's own numbers: where the body is and how it is moving, written flat.
    let state = FreeJointState::new(body.pose(), Twist::zeros());
    let place = state.generalized_position();
    println!("\nFree joint state, as loose numbers");
    println!(
        "  position               = ({:.3}, {:.3}, {:.3}) m",
        place[0], place[1], place[2]
    );
    println!(
        "  orientation            = ({:.3}, {:.3}, {:.3}, {:.3})",
        place[3], place[4], place[5], place[6]
    );
    assert_eq!(FreeJointState::<f64>::GENERALIZED_POSITION_DIMENSION, 7);
    assert_eq!(FreeJointState::<f64>::GENERALIZED_VELOCITY_DIMENSION, 6);
    println!("  place numbers          = 7, motion numbers = 6");

    // Those numbers are the whole state: reading them back gives the same body again.
    let same = FreeJointState::from_generalized_vectors(place, state.generalized_velocity())
        .expect("the seven numbers should describe a usable state");
    let back = same.generalized_position();
    println!("\nReading the numbers back gives the same state");
    for index in 0..7 {
        report(&format!("number[{index}]"), back[index], place[index]);
    }
}
