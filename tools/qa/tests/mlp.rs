#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks the MLP forward pass against numpy goldens, loading the network the way a trained one
//! arrives: one flat block of numbers walked by `ParameterCursor`.
//!
//! The policy case is a 22 → 64 → 64 → 4 quadrotor controller, 5,892 parameters. Its two hidden
//! activations are stored alongside the action, so a wrong answer names the layer it started in
//! rather than only reporting that the four action numbers came out wrong. Each case runs twice,
//! once at each precision: the goldens stay f64, and the f32 pass is the one an embedded caller
//! would actually make.

use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::mlp_inference::{Activation, ParameterCursor};
use multicalc::scalar::Numeric;
use multicalc_qa::load::*;
use multicalc_qa::schema::*;

/// The widths of the policy: observation, both hidden layers, and the action.
const OBSERVATION: usize = 22;
const HIDDEN: usize = 64;
const ACTION: usize = 4;

/// Observations the policy fixture carries. Different rows switch different rectifiers off,
/// so the batch reaches activation patterns a single observation would not.
const SAMPLE_COUNT: usize = 8;

/// Outputs in the activation table, one per pre-activation sum the fixture is built on.
const SUM_COUNT: usize = 7;

#[test]
fn mlp_goldens() {
    let fixtures = load_dir("mlp");
    let mut checked = 0;
    for fixture in &fixtures {
        match fixture.case.as_str() {
            "activations" => check_activations(fixture),
            "policy_22x64x64x4" => check_policy(fixture),
            other => panic!("no check registered for mlp fixture {other}"),
        }
        checked += 1;
    }
    assert_eq!(checked, 2, "expected two mlp fixtures, found {checked}");
}

/// Asserts a layer's outputs match one row of a golden matrix.
fn assert_row<T: Numeric + Into<f64>>(
    got: &[T],
    want: &Value,
    row: usize,
    tolerance: Tol,
    ctx: &str,
) {
    let (rows, cols, data) = want.as_matrix();
    assert_eq!((rows, cols), (SAMPLE_COUNT, got.len()), "{ctx}: shape");
    for (index, &value) in got.iter().enumerate() {
        let (got, want) = (value.into(), data[row * cols + index]);
        assert!(
            close(got, want, tolerance),
            "{ctx}[{row}][{index}]: got {got}, want {want}, tol {tolerance:?}"
        );
    }
}

/// Asserts a layer's outputs match a golden vector.
fn assert_outputs<T: Numeric + Into<f64>>(got: &[T], want: &Value, tolerance: Tol, ctx: &str) {
    let data = want.as_vector();
    assert_eq!(data.len(), got.len(), "{ctx}: length");
    for (index, &value) in got.iter().enumerate() {
        let (got, want) = (value.into(), data[index]);
        assert!(
            close(got, want, tolerance),
            "{ctx}[{index}]: got {got}, want {want}, tol {tolerance:?}"
        );
    }
}

/// One sample's way through the policy: both hidden activations, then the action.
type Sample<T> = (Vector<HIDDEN, T>, Vector<HIDDEN, T>, Vector<ACTION, T>);

/// Loads the three layers from one flat buffer and runs every observation through them.
///
/// Generic over the scalar, so the f32 rerun walks this code rather than a transcription of it.
fn run_policy<T: Numeric>(
    parameters: &[T],
    observations: &Matrix<SAMPLE_COUNT, OBSERVATION, T>,
) -> Vec<Sample<T>> {
    let mut cursor = ParameterCursor::new(parameters);
    let hidden_1 = cursor
        .try_take_layer::<HIDDEN, OBSERVATION>(Activation::Relu)
        .unwrap();
    let hidden_2 = cursor
        .try_take_layer::<HIDDEN, HIDDEN>(Activation::Relu)
        .unwrap();
    let output = cursor
        .try_take_layer::<ACTION, HIDDEN>(Activation::Tanh)
        .unwrap();
    // Walking the buffer to the end is a check in itself: the three shapes declared above account
    // for every number the export holds, with none left over and none read twice.
    assert!(
        cursor.is_empty(),
        "{} parameters left after the last layer",
        cursor.remaining()
    );

    (0..SAMPLE_COUNT)
        .map(|sample| {
            let observation = Vector::<OBSERVATION, T>::from_fn(|i| observations[(sample, i)]);
            let first = hidden_1.forward(observation.view());
            let second = hidden_2.forward(first.view());
            let action = output.forward(second.view());
            (first, second, action)
        })
        .collect()
}

/// Checks every sample's two hidden activations and its action against the goldens. `label`
/// distinguishes the two precisions in a failure message.
fn check_samples<T: Numeric + Into<f64>>(
    fixture: &Fixture,
    samples: &[Sample<T>],
    tolerance: Tol,
    label: &str,
) {
    assert_eq!(samples.len(), SAMPLE_COUNT, "samples run");
    for (sample, (first, second, action)) in samples.iter().enumerate() {
        for (key, got) in [
            ("hidden_1", first.as_slice()),
            ("hidden_2", second.as_slice()),
            ("action", action.as_slice()),
        ] {
            let ctx = format!("{key}{label}");
            assert_row(got, &fixture.expected[key], sample, tolerance, &ctx);
        }
    }
}

fn check_policy(fixture: &Fixture) {
    let observations = &fixture.inputs["observations"];
    let parameters = fixture.inputs["parameters"].as_vector();
    let sampled = to_matrix::<SAMPLE_COUNT, OBSERVATION>(observations);
    check_samples(
        fixture,
        &run_policy(&parameters, &sampled),
        fixture.tolerances.f64,
        "",
    );

    // The same network at the precision a Cortex-M would run it in. The parameters are narrowed
    // once on load rather than converted per inference, which is what an embedded caller does too.
    let tolerance32 = fixture
        .tolerances
        .f32
        .expect("the policy fixture carries an f32 tolerance");
    let parameters32: Vec<f32> = parameters.iter().map(|&x| x as f32).collect();
    let observations32 = to_matrix_f32::<SAMPLE_COUNT, OBSERVATION>(observations);
    check_samples(
        fixture,
        &run_policy(&parameters32, &observations32),
        tolerance32,
        " f32",
    );
}

/// The one-layer activation table, read from the same flat buffer for each activation in turn.
fn run_table<T: Numeric>(
    parameters: &[T],
    observation: Vector<1, T>,
    activation: Activation,
) -> Vector<SUM_COUNT, T> {
    let mut cursor = ParameterCursor::new(parameters);
    let layer = cursor.try_take_layer::<SUM_COUNT, 1>(activation).unwrap();
    assert!(cursor.is_empty(), "the table is one layer and nothing else");
    layer.forward(observation.view())
}

/// Every activation over the same seven pre-activation sums: both signs, zero, and a magnitude far
/// enough out that tanh has saturated.
///
/// One input says nothing about the weighted sum — the policy case covers that — but it pins what
/// each activation does at the values where they differ, which 5,892 numbers would bury.
fn check_activations(fixture: &Fixture) {
    let parameters = fixture.inputs["parameters"].as_vector();
    let observation = to_vector::<1>(&fixture.inputs["observation"]);
    let parameters32: Vec<f32> = parameters.iter().map(|&x| x as f32).collect();
    let observation32 = Vector::<1, f32>::from_fn(|i| observation[i] as f32);
    let tolerance32 = fixture
        .tolerances
        .f32
        .expect("the activation fixture carries an f32 tolerance");

    for (name, activation) in [
        ("relu", Activation::Relu),
        ("tanh", Activation::Tanh),
        ("identity", Activation::Identity),
    ] {
        let golden = &fixture.expected[name];
        let got = run_table(&parameters, observation, activation);
        assert_outputs(got.as_slice(), golden, fixture.tolerances.f64, name);
        // f32 matters most here: tanh at ±8 is where the narrower type has least left to spend.
        let got32 = run_table(&parameters32, observation32, activation);
        let ctx = format!("{name} f32");
        assert_outputs(got32.as_slice(), golden, tolerance32, &ctx);
    }
}
