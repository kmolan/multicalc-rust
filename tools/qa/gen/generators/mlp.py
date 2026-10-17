"""Forward-pass goldens for MLP policy inference, computed in numpy.

The policy case borrows its shape from the quadrotor controller in "Learning to
Fly in Seconds" (arXiv:2311.13081): a 22-component observation, two 64-wide ReLU
hidden layers, and four rotor commands through a tanh. Only the interface is
borrowed — the parameters here are sampled from a fixed seed, not anyone's
trained weights.

The parameters are written as one flat vector, weights row-major then biases,
one layer after the next. That is the order a trained network is exported in and
the order `ParameterCursor` walks, so a fixture that loads is also a check that
the offsets line up. The two hidden activations are stored alongside the action,
which costs 128 numbers on top of 5,892 and turns "the four action numbers are
wrong" into "layer 2 is wrong, layer 1 was fine".

Every layer is worked out a second time index by index before it is written. The
vectorized form folds the transpose into a single BLAS call, which is exactly
where a row-major mix-up would hide; the two disagreeing means the golden is
wrong rather than the crate.
"""

import numpy as np

import schema

# The observation and action widths of the quadrotor policy, and the hidden width
# it uses for both layers.
OBSERVATION = 22
HIDDEN = 64
ACTION = 4

# Observations per fixture. Different rows switch different ReLU units off, so a
# batch covers activation patterns a single row would not reach.
SAMPLE_COUNT = 8

# Pre-activation sums the activation table is built on: both signs, zero, and a
# magnitude far enough out that tanh has saturated.
SUMS = (-8.0, -2.0, -0.5, 0.0, 0.5, 2.0, 8.0)

ACTIVATIONS = {
    "relu": lambda x: np.maximum(x, 0.0),
    "tanh": np.tanh,
    "identity": lambda x: x,
}


def _tol():
    """The f32 bound is two orders tighter than the 1e-3 the decomposition fixtures
    use, because a forward pass has nothing in it that loses precision the way a
    factorization does: it is a fixed sequence of multiplies and adds, so the f32
    answer sits within about 1e-6 of the f64 golden. Left at 1e-3 the f32 leg would
    pass through a regression a thousand times larger than anything it can produce."""
    return {"f64": schema.tol(1e-11, 1e-10), "f32": schema.tol(1e-5, 1e-5)}


def _sample_rotation(rng):
    """A uniformly random rotation, as the Q of a Gaussian 3×3.

    The signs of R's diagonal are divided out so the factor is unique, and a
    column is negated when the determinant comes out negative, which turns a
    reflection into the rotation it mirrors."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0.0:
        q[:, 0] = -q[:, 0]
    return q


def _sample_observations(rng):
    """One row per sample, laid out the way the paper's policy reads it: position
    (3), the rotation matrix row-major (9), linear velocity (3), angular velocity
    (3), and the previous action (4). The ranges are a quadrotor's: metres,
    metres per second, radians per second, and commands already normalized to
    [-1, 1]."""
    rows = []
    for _ in range(SAMPLE_COUNT):
        rows.append(np.concatenate([
            rng.uniform(-2.0, 2.0, size=3),
            _sample_rotation(rng).reshape(9),
            rng.uniform(-3.0, 3.0, size=3),
            rng.uniform(-6.0, 6.0, size=3),
            rng.uniform(-1.0, 1.0, size=4),
        ]))
    return np.array(rows)


def _sample_layer(rng, outputs, inputs, gain):
    """He-scaled weights, which keep the pre-activation spread near one as the
    layers compose, times `gain` to place the last layer inside tanh's linear
    region rather than saturated at ±1 on every sample.

    The biases are small but never zero: a layer that read its bias run at the
    wrong offset, or skipped it, would pass against zeros."""
    weights = gain * rng.normal(0.0, np.sqrt(2.0 / inputs), size=(outputs, inputs))
    biases = rng.uniform(-0.1, 0.1, size=outputs)
    return weights, biases


def _forward(weights, biases, activation, inputs):
    """One layer over a batch of rows: `activation(input · weightsᵀ + biases)`."""
    return ACTIVATIONS[activation](inputs @ weights.T + biases)


def _forward_by_hand(weights, biases, activation, inputs):
    """The same layer written out index by index, as the independent check."""
    outputs, width = weights.shape
    sums = np.empty((inputs.shape[0], outputs))
    for sample in range(inputs.shape[0]):
        for row in range(outputs):
            total = biases[row]
            for col in range(width):
                total += weights[row, col] * inputs[sample, col]
            sums[sample, row] = total
    return ACTIVATIONS[activation](sums)


def _run_network(layers, inputs):
    """Every layer's output in order, checking each against the by-hand form.

    The two differ only in the order the products are summed, so they agree to
    the last few bits rather than exactly."""
    outputs = []
    for weights, biases, activation in layers:
        result = _forward(weights, biases, activation, inputs)
        np.testing.assert_allclose(
            result, _forward_by_hand(weights, biases, activation, inputs),
            rtol=1e-12, atol=1e-12,
        )
        outputs.append(result)
        inputs = result
    return outputs


def _flatten(layers):
    """The layers as one flat buffer: each one's weights row-major, then its
    biases, in the order they are applied."""
    flat = []
    for weights, biases, _ in layers:
        flat.extend(weights.reshape(-1))
        flat.extend(biases)
    return np.array(flat)


def _policy(out, rng, meta):
    layers = [
        (*_sample_layer(rng, HIDDEN, OBSERVATION, 1.0), "relu"),
        (*_sample_layer(rng, HIDDEN, HIDDEN, 1.0), "relu"),
        (*_sample_layer(rng, ACTION, HIDDEN, 0.5), "tanh"),
    ]
    observations = _sample_observations(rng)
    hidden_1, hidden_2, action = _run_network(layers, observations)

    inputs = {
        "kind": schema.string("policy"),
        "parameters": schema.vector(_flatten(layers)),
        "observations": schema.matrix(observations),
    }
    expected = {
        "hidden_1": schema.matrix(hidden_1),
        "hidden_2": schema.matrix(hidden_2),
        "action": schema.matrix(action),
    }
    schema.write_fixture(
        out, "mlp", f"policy_{OBSERVATION}x{HIDDEN}x{HIDDEN}x{ACTION}", meta, _tol(),
        inputs, expected,
        equation="a = tanh(W₃·relu(W₂·relu(W₁·o + b₁) + b₂) + b₃)",
        operations=[f"MLP forward pass, {OBSERVATION}→{HIDDEN}→{HIDDEN}→{ACTION}"],
    )


def _activations(out, meta):
    """Each activation on the same seven pre-activation sums.

    A one-input layer says nothing about the weighted sum — the policy case
    covers that — but it pins what every activation does to a negative, to zero,
    and to a saturating magnitude, where the policy's 5,892 numbers would hide
    it. The weight and the bias each supply half the sum and by different
    factors, so a layer that read the two runs the other way round lands
    somewhere else instead of on the same answer."""
    observation = np.array([2.0])
    weights = np.array([[total * 0.25] for total in SUMS])
    biases = np.array([total * 0.5 for total in SUMS])

    inputs = {
        "kind": schema.string("activations"),
        "parameters": schema.vector(_flatten([(weights, biases, "identity")])),
        "observation": schema.vector(observation),
    }
    expected = {}
    for name in ACTIVATIONS:
        [result] = _run_network([(weights, biases, name)], observation.reshape(1, -1))
        expected[name] = schema.vector(result[0])
    schema.write_fixture(
        out, "mlp", "activations", meta, _tol(), inputs, expected,
        equation="y = activation(w·x + b), sums −8, −2, −0.5, 0, 0.5, 2, 8",
        operations=["Activation table, relu / tanh / identity"],
    )


def run(out, seed):
    rng = np.random.default_rng(seed)
    meta = schema.metadata(
        "mlp", seed,
        "He-scaled normal weights, biases uniform in [-0.1, 0.1]; observations are a "
        "quadrotor state with a uniformly random rotation",
        libraries=("numpy",),
        reference="numpy {numpy}",
    )
    _policy(out, rng, meta)
    _activations(out, meta)
