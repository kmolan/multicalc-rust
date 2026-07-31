"""Signal-processing goldens from scipy.signal, built around whole filtering jobs.

Each case is one realistic pipeline rather than one function call, so a fixture that passes says
a job a caller would actually run comes out right end to end.
"""

import numpy as np
import scipy.signal

import schema

SAMPLE_RATE_HZ = 1000.0
DT = 1.0 / SAMPLE_RATE_HZ
# Frequencies the response is checked at, spanning well below the cutoff to near the limit.
PROBE_HZ = [1.0, 5.0, 20.0, 50.0, 80.0, 120.0, 180.0, 200.0, 300.0, 400.0, 450.0]


def _time(count):
    return np.arange(count) * DT


# A rate reading from a drone: a slow manoeuvre with the motor's vibration and its first two
# harmonics riding on top, which is what the conditioning chain has to clean up.
GYRO_INPUT = (np.sin(2 * np.pi * 3 * _time(128))
              + 0.4 * np.sin(2 * np.pi * 80 * _time(128))
              + 0.2 * np.sin(2 * np.pi * 160 * _time(128))
              + 0.1 * np.sin(2 * np.pi * 240 * _time(128)))
# The same reading before the notch frequency is known: a band-pass isolates the tone so its
# frequency can be tracked.
VIBRATION_INPUT = np.sin(2 * np.pi * 5 * _time(64)) + 0.8 * np.sin(2 * np.pi * 180 * _time(64))
# An accelerometer with a slow bias wandering under the signal of interest.
DRIFT_INPUT = 0.5 * np.sin(2 * np.pi * 0.5 * _time(64)) + 0.3 * np.sin(2 * np.pi * 30 * _time(64))
# An encoder counting up a curve, with a small ripple standing in for measurement noise.
ENCODER_INPUT = 0.5 * _time(60) ** 2 + 0.001 * np.sin(2 * np.pi * 120 * _time(60))
# A range finder pointed at a wall, with three returns off dust or a passing edge.
LIDAR_INPUT = 2.0 + 0.3 * np.sin(2 * np.pi * 3 * _time(60))
for _spike_at, _spike_value in ((12, 8.0), (31, 0.4), (47, 9.5)):
    LIDAR_INPUT[_spike_at] = _spike_value

# The two section sharpnesses a 4th-order Butterworth is built from, so the Rust side designs its
# own sections rather than reading stored weights.
BUTTERWORTH_QUALITY_FACTORS = [1.0 / (2.0 * np.cos(np.pi / 8.0)),
                               1.0 / (2.0 * np.cos(3.0 * np.pi / 8.0))]


def _tol():
    return {"f64": schema.tol(1e-11, 1e-10)}


def _curve_fit_tol():
    """A looser absolute floor for the curve fit, because the reference is noisier than it.

    The second-derivative weights are scaled by 2/dt², which is two million here, so SciPy's own
    rounding noise lands around 1e-10 in absolute terms — and where the exact weight is zero, that
    noise is all there is. multicalc returns the exact zero, so an absolute floor tighter than the
    reference's own precision would reject the better answer. The relative term still does the real
    work: these weights run to 1e5, where it allows 1e-5."""
    return {"f64": schema.tol(1e-9, 1e-10)}


def _response(sections, worN):
    """Magnitude and phase of a chain of `(b, a)` pairs at each probe frequency."""
    combined = np.ones(len(worN), dtype=complex)
    for b, a in sections:
        _, response = scipy.signal.freqz(b, a, worN=worN)
        combined = combined * response
    return np.abs(combined), np.angle(combined)


def _filtered(sections, signal):
    """One signal through a chain of `(b, a)` pairs, in order."""
    output = np.asarray(signal, dtype=float)
    for b, a in sections:
        output = scipy.signal.lfilter(b, a, output)
    return output


def _gyro_conditioning_chain(out, meta):
    """Motor-harmonic notches followed by a 4th-order low-pass: a drone's gyro chain."""
    fundamental_hz, quality_factor, notch_sections = 80.0, 4.0, 3
    low_pass_hz = 120.0

    notches = [scipy.signal.iirnotch(fundamental_hz * harmonic, quality_factor, fs=SAMPLE_RATE_HZ)
               for harmonic in range(1, notch_sections + 1)]
    # SciPy's own 4th-order low-pass, as the two sections it factors into. The Rust side rebuilds
    # these from its own design function and the two stored sharpnesses, so the comparison is
    # against SciPy's arithmetic rather than a restatement of multicalc's.
    sos = scipy.signal.butter(4, low_pass_hz, btype="low", fs=SAMPLE_RATE_HZ, output="sos")
    low_pass = [(row[:3], row[3:]) for row in sos]

    chain = notches + low_pass
    magnitude, phase = _response(chain, [2 * np.pi * f * DT for f in PROBE_HZ])
    fundamental_b, fundamental_a = notches[0]

    inputs = {
        "kind": schema.string("gyro_chain"),
        "fundamental_hz": schema.scalar(fundamental_hz),
        "quality_factor": schema.scalar(quality_factor),
        "notch_sections": schema.integer(notch_sections),
        "low_pass_hz": schema.scalar(low_pass_hz),
        "low_pass_quality_factors": schema.vector(BUTTERWORTH_QUALITY_FACTORS),
        "dt": schema.scalar(DT),
        "probe_hz": schema.vector(PROBE_HZ),
        "input": schema.vector(GYRO_INPUT),
    }
    expected = {
        "notch_feed_forward": schema.vector(fundamental_b / fundamental_a[0]),
        "notch_feedback": schema.vector(fundamental_a[1:] / fundamental_a[0]),
        "magnitude": schema.vector(magnitude),
        "phase": schema.vector(phase),
        "output": schema.vector(_filtered(chain, GYRO_INPUT)),
    }
    schema.write_fixture(out, "signal_processing", "gyro_conditioning_chain", meta, _tol(),
                         inputs, expected,
                         equation="Notches on 80/160/240 Hz into a 4th-order 120 Hz low-pass, 1 kHz",
                         operations=["Gyro conditioning chain: notch weights, response, and "
                                     "filtered output"])


def _single_section(out, meta, case, design, label, frequency_hz, quality_factor, coefficients,
                    signal, equation):
    """One designed section: its weights, its response, and a signal run through it."""
    b, a = coefficients
    magnitude, phase = _response([(b, a)], [2 * np.pi * f * DT for f in PROBE_HZ])
    inputs = {
        "kind": schema.string("biquad_section"),
        "design": schema.string(design),
        "frequency_hz": schema.scalar(frequency_hz),
        "quality_factor": schema.scalar(quality_factor),
        "dt": schema.scalar(DT),
        "probe_hz": schema.vector(PROBE_HZ),
        "input": schema.vector(signal),
    }
    expected = {
        "feed_forward": schema.vector(b / a[0]),
        "feedback": schema.vector(a[1:] / a[0]),
        "magnitude": schema.vector(magnitude),
        "phase": schema.vector(phase),
        "output": schema.vector(_filtered([(b, a)], signal)),
    }
    schema.write_fixture(out, "signal_processing", case, meta, _tol(), inputs, expected,
                         equation=equation,
                         operations=[f"{label}: weights, response, and filtered output"])


def _vibration_band_pass(out, meta):
    """Isolating the motor tone so its frequency can be tracked."""
    _single_section(out, meta, "motor_vibration_band_pass", "band_pass",
                    "Motor-vibration band-pass", 180.0, 4.0,
                    scipy.signal.iirpeak(180.0, 4.0, fs=SAMPLE_RATE_HZ), VIBRATION_INPUT,
                    "Band-pass on 180 Hz, sharpness 4, 1 kHz")


def _drift_removal_high_pass(out, meta):
    """Taking a slow bias out from under an accelerometer reading."""
    root_half = 1.0 / np.sqrt(2.0)
    _single_section(out, meta, "accelerometer_drift_high_pass", "high_pass",
                    "Drift-removal high-pass", 10.0, root_half,
                    scipy.signal.butter(2, 10.0, btype="high", fs=SAMPLE_RATE_HZ), DRIFT_INPUT,
                    "2nd-order high-pass, 10 Hz, sharpness 0.707, 1 kHz")


def _encoder_curve_fit(out, meta):
    """Position, rate, and acceleration off one noisy encoder, read both ways."""
    window, terms = 11, 3
    inputs = {
        "kind": schema.string("savitzky_golay"),
        "window": schema.integer(window),
        "polynomial_terms": schema.integer(terms),
        "dt": schema.scalar(DT),
        "input": schema.vector(ENCODER_INPUT),
    }
    expected = {}
    recent = ENCODER_INPUT[-window:]
    for position, read_position in (("latest", window - 1), ("centered", (window - 1) // 2)):
        # SciPy's polyorder is one less than multicalc's term count; use="dot" puts the weights in
        # the order they multiply an oldest-to-newest window; delta folds in the per-second scaling,
        # so the deriv=1 and deriv=2 rows already carry the 1/dt and 2/dt² multicalc's rows carry.
        rows = [scipy.signal.savgol_coeffs(window, polyorder=terms - 1, deriv=derivative, delta=DT,
                                           pos=read_position, use="dot")
                for derivative in (0, 1, 2)]
        expected[f"{position}_smoothing_weights"] = schema.vector(rows[0])
        expected[f"{position}_first_derivative_weights"] = schema.vector(rows[1])
        expected[f"{position}_second_derivative_weights"] = schema.vector(rows[2])
        expected[f"{position}_value"] = schema.scalar(np.dot(rows[0], recent))
        expected[f"{position}_first_derivative"] = schema.scalar(np.dot(rows[1], recent))
        expected[f"{position}_second_derivative"] = schema.scalar(np.dot(rows[2], recent))
    schema.write_fixture(out, "signal_processing", "encoder_curve_fit", meta, _curve_fit_tol(),
                         inputs, expected,
                         equation=f"{terms}-term curve across {window} samples, read at the newest "
                                  f"sample and at the middle",
                         operations=["Encoder curve fit: weights, value, rate, and acceleration, "
                                     "both read positions"])


def _lidar_despiking(out, meta, numpy_version):
    """A range trace with three bad returns, through a median and through a mean."""
    window = 5
    half = (window - 1) // 2
    first_compared_index = window - 1
    # medfilt centres its window on each sample while multicalc's covers the samples up to and
    # including it, so its output is shifted by half a window. Only indices from window-1 on are
    # kept, which is exactly the range where neither side reaches medfilt's zero padding.
    centered = scipy.signal.medfilt(LIDAR_INPUT, kernel_size=window)
    median_output = [centered[i - half] for i in range(first_compared_index, len(LIDAR_INPUT))]
    mean_output = [float(np.mean(LIDAR_INPUT[i - window + 1:i + 1]))
                   for i in range(first_compared_index, len(LIDAR_INPUT))]

    inputs = {
        "kind": schema.string("window_filters"),
        "window": schema.integer(window),
        "first_compared_index": schema.integer(first_compared_index),
        "input": schema.vector(LIDAR_INPUT),
    }
    expected = {
        "median_output": schema.vector(median_output),
        "mean_output": schema.vector(mean_output),
    }
    # A sliding mean is not a scipy.signal routine, so the mean half is checked against NumPy and
    # the accuracy table says so.
    both = {**meta, "reference": f"SciPy {meta['libraries']['scipy']} / NumPy {numpy_version}"}
    schema.write_fixture(out, "signal_processing", "lidar_despiking", both, _tol(), inputs,
                         expected,
                         equation=f"Middle value and mean of the last {window} samples",
                         operations=[f"Lidar despiking: {window}-wide running median and moving "
                                     f"average"])


def run(out, seed):
    meta = schema.metadata(
        "signal_processing", seed,
        "fixed 1 kHz sequences: a gyro reading with motor harmonics, an accelerometer with a slow "
        "bias, an encoder counting up a curve, and a range trace with three bad returns",
        libraries=("numpy", "scipy"), reference="SciPy {scipy}",
    )
    _gyro_conditioning_chain(out, meta)
    _vibration_band_pass(out, meta)
    _drift_removal_high_pass(out, meta)
    _encoder_curve_fit(out, meta)
    _lidar_despiking(out, meta, np.__version__)
