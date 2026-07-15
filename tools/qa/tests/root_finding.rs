#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks the scalar and system root finders against scipy.optimize goldens.
//!
//! Each fixture names a problem and a solver; the problem comes from the shared
//! registry so its formula is identical to the one scipy solved, and its
//! parameters are read from the fixture so the numbers are shared. Every case also
//! re-checks `f_at_probe`, which fails loudly if the Rust and Python formulas ever
//! diverge.

use multicalc::linear_algebra::Vector;
use multicalc::numerical_derivative::autodiff::{AutoDiffMulti, AutoDiffSingle};
use multicalc::root_finding::{Bisection, Newton, NewtonSystem};
use multicalc::scalar::{ScalarFn, VectorFn};
use multicalc_qa::load::*;
use multicalc_qa::problems::*;
use multicalc_qa::schema::*;

fn run_scalar<F: ScalarFn>(f: &F, fx: &Fixture) {
    let t = fx.tolerances.get("f64", "host");
    let root = match fx.inputs["solver"].as_str() {
        "bisection" => {
            let br = fx.inputs["bracket"].as_vector();
            Bisection::default().solve(f, br[0], br[1]).unwrap().root
        }
        "newton" => {
            let x0 = fx.inputs["start"].as_scalar();
            Newton::<AutoDiffSingle>::default().solve(f, x0).unwrap().root
        }
        "damped_newton" => {
            let x0 = fx.inputs["start"].as_scalar();
            Newton::<AutoDiffSingle>::default()
                .with_backtracking(true)
                .solve(f, x0)
                .unwrap()
                .root
        }
        other => panic!("unknown scalar solver {other}"),
    };
    assert_scalar(root, &fx.expected["root"], t, "root");
    let probe = fx.inputs["probe"].as_scalar();
    assert_scalar(f.eval::<f64>(probe), &fx.expected["f_at_probe"], t, "f_at_probe");
}

fn run_system<F: VectorFn<N, N>, const N: usize>(f: &F, fx: &Fixture) {
    let t = fx.tolerances.get("f64", "host");
    let x0 = to_vector::<N>(&fx.inputs["start"]).into_array();
    let root = NewtonSystem::<AutoDiffMulti>::default().solve(f, &x0).unwrap().root;
    assert_vector(&Vector::new(root), &fx.expected["root"], t, "root");
    let probe = to_vector::<N>(&fx.inputs["probe"]).into_array();
    assert_vector(
        &Vector::new(f.eval::<f64>(&probe)),
        &fx.expected["f_at_probe"],
        t,
        "f_at_probe",
    );
}

#[test]
fn root_finding() {
    for fx in load_dir("fixtures/v1/root_finding") {
        match fx.inputs["problem"].as_str() {
            "root_wien" => run_scalar(&Wien, &fx),
            "root_sigmoid" => run_scalar(&Sigmoid, &fx),
            "root_kepler" => run_scalar(
                &Kepler {
                    e: fx.inputs["e"].as_scalar(),
                    m: fx.inputs["m"].as_scalar(),
                },
                &fx,
            ),
            "root_colebrook" => run_scalar(
                &Colebrook {
                    reynolds: fx.inputs["reynolds"].as_scalar(),
                    rel_roughness: fx.inputs["rel_roughness"].as_scalar(),
                },
                &fx,
            ),
            "sys_two_link" => run_system::<_, 2>(
                &TwoLinkArm {
                    l1: fx.inputs["l1"].as_scalar(),
                    l2: fx.inputs["l2"].as_scalar(),
                    px: fx.inputs["px"].as_scalar(),
                    py: fx.inputs["py"].as_scalar(),
                },
                &fx,
            ),
            "sys_circle_hyperbola" => run_system::<_, 2>(&CircleHyperbola, &fx),
            "sys_equilibrium" => run_system::<_, 3>(&Equilibrium, &fx),
            other => panic!("unknown problem {other}"),
        }
    }
}
