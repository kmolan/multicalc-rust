#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks expm / ZOH / Van Loan / q_discrete_white_noise against scipy goldens.

use multicalc::discretization::{q_discrete_white_noise, van_loan, zoh};
use multicalc_qa::load::*;
use multicalc_qa::schema::*;

fn run_expm<const N: usize>(fixture: &Fixture) {
    let a = to_matrix::<N, N>(&fixture.inputs["A"]);
    let tolerance = fixture.tolerances.f64;
    assert_matrix(
        &a.expm().unwrap(),
        &fixture.expected["expm"],
        tolerance,
        "expm",
    );
}

#[test]
fn expm() {
    for fixture in load_dir("discretization") {
        if fixture.inputs["kind"].as_str() != "expm" {
            continue;
        }
        let (n, _) = fixture.inputs["A"].shape();
        match n {
            2 => run_expm::<2>(&fixture),
            3 => run_expm::<3>(&fixture),
            4 => run_expm::<4>(&fixture),
            5 => run_expm::<5>(&fixture),
            n => panic!("unregistered expm size {n}"),
        }
    }
}

fn run_zoh<const N: usize, const M: usize, const NM: usize>(fixture: &Fixture) {
    let a = to_matrix::<N, N>(&fixture.inputs["A"]);
    let b = to_matrix::<N, M>(&fixture.inputs["B"]);
    let timestep = fixture.inputs["dt"].as_scalar();
    let tolerance = fixture.tolerances.f64;
    let (state_transition, input_matrix) = zoh::<N, M, NM, f64>(a, b, timestep).unwrap();
    assert_matrix(&state_transition, &fixture.expected["F"], tolerance, "F");
    assert_matrix(&input_matrix, &fixture.expected["G"], tolerance, "G");
}

#[test]
fn zoh_cases() {
    for fixture in load_dir("discretization") {
        if fixture.inputs["kind"].as_str() != "zoh" {
            continue;
        }
        let (n, _) = fixture.inputs["A"].shape();
        let (_, input_cols) = fixture.inputs["B"].shape();
        match (n, input_cols) {
            (2, 1) => run_zoh::<2, 1, 3>(&fixture),
            (3, 2) => run_zoh::<3, 2, 5>(&fixture),
            (4, 2) => run_zoh::<4, 2, 6>(&fixture),
            shape => panic!("unregistered zoh shape {shape:?}"),
        }
    }
}

fn run_van_loan<const N: usize, const N2: usize>(fixture: &Fixture) {
    let a = to_matrix::<N, N>(&fixture.inputs["A"]);
    let process_noise = to_matrix::<N, N>(&fixture.inputs["Qc"]);
    let timestep = fixture.inputs["dt"].as_scalar();
    let tolerance = fixture.tolerances.f64;
    let (state_transition, discrete_noise) =
        van_loan::<N, N2, f64>(a, process_noise, timestep).unwrap();
    assert_matrix(&state_transition, &fixture.expected["F"], tolerance, "F");
    assert_matrix(&discrete_noise, &fixture.expected["Qd"], tolerance, "Qd");
}

#[test]
fn van_loan_cases() {
    for fixture in load_dir("discretization") {
        if fixture.inputs["kind"].as_str() != "van_loan" {
            continue;
        }
        let (n, _) = fixture.inputs["A"].shape();
        match n {
            2 => run_van_loan::<2, 4>(&fixture),
            3 => run_van_loan::<3, 6>(&fixture),
            n => panic!("unregistered van_loan size {n}"),
        }
    }
}

#[test]
fn qdwn_cases() {
    for fixture in load_dir("discretization") {
        if fixture.inputs["kind"].as_str() != "qdwn" {
            continue;
        }
        let timestep = fixture.inputs["dt"].as_scalar();
        let var = fixture.inputs["variance"].as_scalar();
        let tolerance = fixture.tolerances.f64;
        match fixture.inputs["dim"].as_int() {
            2 => assert_matrix(
                &q_discrete_white_noise::<2, f64>(timestep, var),
                &fixture.expected["Q"],
                tolerance,
                "Q2",
            ),
            3 => assert_matrix(
                &q_discrete_white_noise::<3, f64>(timestep, var),
                &fixture.expected["Q"],
                tolerance,
                "Q3",
            ),
            4 => assert_matrix(
                &q_discrete_white_noise::<4, f64>(timestep, var),
                &fixture.expected["Q"],
                tolerance,
                "Q4",
            ),
            dim => panic!("unregistered qdwn dim {dim}"),
        }
    }
}
