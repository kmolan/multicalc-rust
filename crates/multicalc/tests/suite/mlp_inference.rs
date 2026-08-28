use multicalc::linear_algebra::Vector;
use multicalc::mlp_inference::{Activation, Layer};

#[test]
fn relu_clamps_a_negative_sum_and_leaves_a_positive_one() {
    assert_eq!(Activation::Relu.apply(-1.0), 0.0);
    assert_eq!(Activation::Relu.apply(0.0), 0.0);
    assert_eq!(Activation::Relu.apply(2.5), 2.5);
}

#[test]
fn two_layers_chain_hidden_output_into_the_next_input() {
    let hidden_weights = [0.5, -0.5, 1.0, 0.0, -1.0, 2.0]; // 3x2
    let hidden_biases = [0.0, 1.0, -1.0];
    let output_weights = [1.0, 1.0, 1.0]; // 1x3
    let output_biases = [0.5];

    let hidden =
        Layer::<3, 2>::try_from_slices(&hidden_weights, &hidden_biases, Activation::Relu).unwrap();
    let output =
        Layer::<1, 3>::try_from_slices(&output_weights, &output_biases, Activation::Identity)
            .unwrap();

    let input = Vector::new([2.0, 1.0]);
    let activations = hidden.forward(input.view()).unwrap();

    // The third unit's sum is -1.0, so ReLU switches it off.
    assert_eq!(activations.into_array(), [0.5, 3.0, 0.0]);
    assert_eq!(
        output.forward(activations.view()).unwrap().into_array(),
        [4.0]
    );
}

#[test]
fn zero_weights_leave_only_the_biases() {
    let weights = [0.0; 6];
    let biases = [1.0, -2.0, 3.0];
    let layer = Layer::<3, 2>::try_from_slices(&weights, &biases, Activation::Identity).unwrap();

    let output = layer.forward(Vector::new([7.0, -9.0]).view()).unwrap();

    assert_eq!(output.into_array(), biases);
}

#[test]
fn identity_weights_and_zero_biases_pass_the_input_through() {
    let weights = [1.0, 0.0, 0.0, 1.0]; // 2x2 identity
    let biases = [0.0, 0.0];
    let layer = Layer::<2, 2>::try_from_slices(&weights, &biases, Activation::Identity).unwrap();

    let input = Vector::new([2.5, -4.0]);

    assert_eq!(layer.forward(input.view()).unwrap(), input);
}

#[test]
fn a_row_reads_the_whole_input_rather_than_summing_its_own_weights() {
    // Rows sum to 6 and 15, which the dot products must not be mistaken for.
    let weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let biases = [0.0, 0.0];
    let layer = Layer::<2, 3>::try_from_slices(&weights, &biases, Activation::Identity).unwrap();

    let output = layer
        .forward(Vector::new([10.0, 20.0, 30.0]).view())
        .unwrap();

    assert_eq!(output.into_array(), [140.0, 320.0]);
}

#[test]
fn relu_never_returns_a_negative_activation() {
    let weights = [-1.0, -1.0, -1.0, -1.0];
    let biases = [-5.0, -5.0];
    let layer = Layer::<2, 2>::try_from_slices(&weights, &biases, Activation::Relu).unwrap();

    let output = layer.forward(Vector::new([3.0, 4.0]).view()).unwrap();

    assert!(output.as_slice().iter().all(|value| *value >= 0.0));
}

#[test]
fn a_slice_too_short_for_the_declared_shape_is_rejected() {
    let weights = [1.0, 2.0, 3.0];
    let biases = [0.0, 0.0];
    assert!(Layer::<2, 2>::try_from_slices(&weights, &biases, Activation::Relu).is_err());
}
