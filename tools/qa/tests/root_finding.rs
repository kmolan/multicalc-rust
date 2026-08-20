#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks the scalar and system root finders against scipy.optimize goldens.
//!
//! Each fixture names a problem and a solver; the problem comes from the shared
//! registry so its formula is identical to the one scipy solved, and its
//! parameters are read from the fixture so the numbers are shared. Every case also
//! re-checks `f_at_probe`, which fails loudly if the Rust and Python formulas ever
//! diverge.

use multicalc::linear_algebra::Vector;
use multicalc::numerical_derivative::{AutoDiffMulti, AutoDiffSingle};
use multicalc::root_finding::{Bisection, Newton, NewtonSystem};
use multicalc::scalar::{ScalarFn, VectorFn};
use multicalc_qa::load::*;
use multicalc_qa::problems::*;
use multicalc_qa::schema::*;

fn run_scalar<F: ScalarFn>(f: &F, fixture: &Fixture) {
    let tolerance = fixture.tolerances.f64;
    let root = match fixture.inputs["solver"].as_str() {
        "bisection" => {
            let bracket = fixture.inputs["bracket"].as_vector();
            Bisection::default()
                .solve(f, bracket[0], bracket[1])
                .unwrap()
                .root
        }
        "newton" => {
            let initial_guess = fixture.inputs["start"].as_scalar();
            Newton::<AutoDiffSingle>::default()
                .solve(f, initial_guess)
                .unwrap()
                .root
        }
        "damped_newton" => {
            let initial_guess = fixture.inputs["start"].as_scalar();
            Newton::<AutoDiffSingle>::default()
                .with_backtracking(true)
                .solve(f, initial_guess)
                .unwrap()
                .root
        }
        other => panic!("unknown scalar solver {other}"),
    };
    assert_scalar(root, &fixture.expected["root"], tolerance, "root");
    let probe = fixture.inputs["probe"].as_scalar();
    assert_scalar(
        f.eval::<f64>(probe),
        &fixture.expected["f_at_probe"],
        tolerance,
        "f_at_probe",
    );
}

fn run_system<F: VectorFn<N, N>, const N: usize>(f: &F, fixture: &Fixture) {
    let tolerance = fixture.tolerances.f64;
    let initial_guess = to_vector::<N>(&fixture.inputs["start"]).into_array();
    let root = NewtonSystem::<AutoDiffMulti>::default()
        .solve(f, &initial_guess)
        .unwrap()
        .root;
    assert_vector(
        &Vector::new(root),
        &fixture.expected["root"],
        tolerance,
        "root",
    );
    let probe = to_vector::<N>(&fixture.inputs["probe"]).into_array();
    assert_vector(
        &Vector::new(f.eval::<f64>(&probe)),
        &fixture.expected["f_at_probe"],
        tolerance,
        "f_at_probe",
    );
}

#[test]
fn root_finding() {
    for fixture in load_dir("root_finding") {
        match fixture.inputs["problem"].as_str() {
            "root_wien" => run_scalar(&Wien, &fixture),
            "root_sigmoid" => run_scalar(&Sigmoid, &fixture),
            "root_kepler" => run_scalar(
                &Kepler {
                    eccentricity: fixture.inputs["e"].as_scalar(),
                    mean_anomaly: fixture.inputs["m"].as_scalar(),
                },
                &fixture,
            ),
            "root_colebrook" => run_scalar(
                &Colebrook {
                    reynolds: fixture.inputs["reynolds"].as_scalar(),
                    rel_roughness: fixture.inputs["rel_roughness"].as_scalar(),
                },
                &fixture,
            ),
            "sys_two_link" => run_system::<_, 2>(
                &TwoLinkArm {
                    first_link: fixture.inputs["l1"].as_scalar(),
                    second_link: fixture.inputs["l2"].as_scalar(),
                    target_x: fixture.inputs["px"].as_scalar(),
                    target_y: fixture.inputs["py"].as_scalar(),
                },
                &fixture,
            ),
            "sys_circle_hyperbola" => run_system::<_, 2>(&CircleHyperbola, &fixture),
            "sys_equilibrium" => run_system::<_, 3>(&Equilibrium, &fixture),
            other => panic!("unknown problem {other}"),
        }
    }
}
