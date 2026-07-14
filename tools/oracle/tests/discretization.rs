#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks expm / ZOH / Van Loan / q_discrete_white_noise against scipy goldens.

use multicalc::discretization::{q_discrete_white_noise, van_loan, zoh};
use multicalc_oracle::load::*;
use multicalc_oracle::schema::*;

fn run_expm<const N: usize>(fx: &Fixture) {
    let a = to_matrix::<N, N>(&fx.inputs["A"]);
    let t = fx.tolerances.get("f64", "host");
    assert_matrix(&a.expm().unwrap(), &fx.expected["expm"], t, "expm");
}

#[test]
fn expm() {
    for fx in load_dir("fixtures/v1/discretization") {
        if fx.inputs["kind"].as_str() != "expm" {
            continue;
        }
        let (n, ..) = fx.inputs["A"].as_matrix();
        match n {
            2 => run_expm::<2>(&fx),
            3 => run_expm::<3>(&fx),
            4 => run_expm::<4>(&fx),
            5 => run_expm::<5>(&fx),
            n => panic!("unregistered expm size {n}"),
        }
    }
}

fn run_zoh<const N: usize, const M: usize, const NM: usize>(fx: &Fixture) {
    let a = to_matrix::<N, N>(&fx.inputs["A"]);
    let b = to_matrix::<N, M>(&fx.inputs["B"]);
    let dt = fx.inputs["dt"].as_scalar();
    let t = fx.tolerances.get("f64", "host");
    let (f, g) = zoh::<N, M, NM, f64>(a, b, dt).unwrap();
    assert_matrix(&f, &fx.expected["F"], t, "F");
    assert_matrix(&g, &fx.expected["G"], t, "G");
}

#[test]
fn zoh_cases() {
    for fx in load_dir("fixtures/v1/discretization") {
        if fx.inputs["kind"].as_str() != "zoh" {
            continue;
        }
        let (n, m, _) = fx.inputs["A"].as_matrix();
        let mcols = fx.inputs["B"].as_matrix().1;
        match (n, mcols) {
            (2, 1) => run_zoh::<2, 1, 3>(&fx),
            (3, 2) => run_zoh::<3, 2, 5>(&fx),
            (4, 2) => run_zoh::<4, 2, 6>(&fx),
            s => panic!("unregistered zoh shape {s:?} (m={m})"),
        }
    }
}

fn run_van_loan<const N: usize, const N2: usize>(fx: &Fixture) {
    let a = to_matrix::<N, N>(&fx.inputs["A"]);
    let qc = to_matrix::<N, N>(&fx.inputs["Qc"]);
    let dt = fx.inputs["dt"].as_scalar();
    let t = fx.tolerances.get("f64", "host");
    let (f, qd) = van_loan::<N, N2, f64>(a, qc, dt).unwrap();
    assert_matrix(&f, &fx.expected["F"], t, "F");
    assert_matrix(&qd, &fx.expected["Qd"], t, "Qd");
}

#[test]
fn van_loan_cases() {
    for fx in load_dir("fixtures/v1/discretization") {
        if fx.inputs["kind"].as_str() != "van_loan" {
            continue;
        }
        let (n, ..) = fx.inputs["A"].as_matrix();
        match n {
            2 => run_van_loan::<2, 4>(&fx),
            3 => run_van_loan::<3, 6>(&fx),
            n => panic!("unregistered van_loan size {n}"),
        }
    }
}

#[test]
fn qdwn_cases() {
    for fx in load_dir("fixtures/v1/discretization") {
        if fx.inputs["kind"].as_str() != "qdwn" {
            continue;
        }
        let dt = fx.inputs["dt"].as_scalar();
        let var = fx.inputs["variance"].as_scalar();
        let t = fx.tolerances.get("f64", "host");
        match fx.inputs["dim"].as_int() {
            2 => assert_matrix(
                &q_discrete_white_noise::<2, f64>(dt, var),
                &fx.expected["Q"],
                t,
                "Q2",
            ),
            3 => assert_matrix(
                &q_discrete_white_noise::<3, f64>(dt, var),
                &fx.expected["Q"],
                t,
                "Q3",
            ),
            4 => assert_matrix(
                &q_discrete_white_noise::<4, f64>(dt, var),
                &fx.expected["Q"],
                t,
                "Q4",
            ),
            d => panic!("unregistered qdwn dim {d}"),
        }
    }
}
