"""Biquad design, response, and filtered-output goldens from scipy.signal."""

import numpy as np
import scipy.signal

import schema

SAMPLE_RATE_HZ = 1000.0
DT = 1.0 / SAMPLE_RATE_HZ
# Frequencies the response is checked at, spanning well below the cutoff to near the limit.
PROBE_HZ = [1.0, 5.0, 20.0, 50.0, 80.0, 120.0, 180.0, 200.0, 300.0, 400.0, 450.0]
# A slow component plus a fast one, so a low-pass and a notch both have something to do.
INPUT = [np.sin(2 * np.pi * 5 * n * DT) + 0.8 * np.sin(2 * np.pi * 180 * n * DT)
         for n in range(64)]


def _tol():
    return {"f64": schema.tol(1e-11, 1e-10)}


def _response(b, a):
    """Magnitude and phase at each probe frequency, from one section's weights."""
    _, response = scipy.signal.freqz(b, a, worN=[2 * np.pi * f * DT for f in PROBE_HZ])
    return np.abs(response), np.angle(response)


def _section(out, meta, case, design, label, frequency_hz, quality_factor, coefficients, equation):
    """Records one designed section: its weights, its response, and its filtered output.

    `design` is what the Rust suite matches on to pick a design function; `label` is how the
    accuracy table reads."""
    b, a = coefficients
    magnitude, phase = _response(b, a)
    inputs = {
        "kind": schema.string("biquad_section"),
        "design": schema.string(design),
        "frequency_hz": schema.scalar(frequency_hz),
        "quality_factor": schema.scalar(quality_factor),
        "dt": schema.scalar(DT),
        "probe_hz": schema.vector(PROBE_HZ),
        "input": schema.vector(INPUT),
    }
    expected = {
        "feed_forward": schema.vector(b / a[0]),
        "feedback": schema.vector(a[1:] / a[0]),
        "magnitude": schema.vector(magnitude),
        "phase": schema.vector(phase),
        "output": schema.vector(scipy.signal.lfilter(b, a, INPUT)),
    }
    schema.write_fixture(out, "signal_processing", case, meta, _tol(), inputs, expected,
                         equation=equation,
                         operations=[f"Biquad {label} weights",
                                     f"Biquad {label} magnitude and phase",
                                     f"Biquad {label} filtered output"])


def _sections(out, meta):
    root_half = 1.0 / np.sqrt(2.0)
    _section(out, meta, "biquad_low_pass_50hz", "low_pass", "low-pass", 50.0, root_half,
             scipy.signal.butter(2, 50.0, btype="low", fs=SAMPLE_RATE_HZ),
             "2nd-order low-pass, 50 Hz, sharpness 0.707, 1 kHz")
    _section(out, meta, "biquad_high_pass_50hz", "high_pass", "high-pass", 50.0, root_half,
             scipy.signal.butter(2, 50.0, btype="high", fs=SAMPLE_RATE_HZ),
             "2nd-order high-pass, 50 Hz, sharpness 0.707, 1 kHz")
    _section(out, meta, "biquad_band_pass_180hz", "band_pass", "band-pass", 180.0, 4.0,
             scipy.signal.iirpeak(180.0, 4.0, fs=SAMPLE_RATE_HZ),
             "2nd-order band-pass, 180 Hz, sharpness 4, 1 kHz")
    _section(out, meta, "biquad_notch_180hz", "notch", "notch", 180.0, 4.0,
             scipy.signal.iirnotch(180.0, 4.0, fs=SAMPLE_RATE_HZ),
             "2nd-order notch, 180 Hz, sharpness 4, 1 kHz")


def _cascade(out, meta, case, design, sections, extra_inputs, response, output, equation,
             operations):
    """Records a chain of sections by what the chain does, not by its section weights.

    SciPy's section ordering is its own business and need not match the order multicalc
    builds them in, so comparing section by section would compare an ordering rather than a
    filter. Magnitude, phase, and filtered output do not depend on the order, and the
    per-section weights are already pinned by the single-section cases."""
    magnitude, phase = response
    inputs = {
        "kind": schema.string("biquad_cascade"),
        "design": schema.string(design),
        "sections": schema.integer(sections),
        "dt": schema.scalar(DT),
        "probe_hz": schema.vector(PROBE_HZ),
        "input": schema.vector(INPUT),
        **extra_inputs,
    }
    expected = {
        "magnitude": schema.vector(magnitude),
        "phase": schema.vector(phase),
        "output": schema.vector(output),
    }
    schema.write_fixture(out, "signal_processing", case, meta, _tol(), inputs, expected,
                         equation=equation, operations=operations)


def _fourth_order_low_pass(out, meta):
    sos = scipy.signal.butter(4, 80.0, btype="low", fs=SAMPLE_RATE_HZ, output="sos")
    _, response = scipy.signal.sosfreqz(sos, worN=[2 * np.pi * f * DT for f in PROBE_HZ])
    # The two section sharpnesses a 4th-order Butterworth is built from, so the Rust side
    # designs its own sections rather than reading stored weights.
    quality_factors = [1.0 / (2.0 * np.cos(np.pi / 8.0)), 1.0 / (2.0 * np.cos(3.0 * np.pi / 8.0))]
    _cascade(out, meta, "biquad_cascade_fourth_order_low_pass_80hz", "low_pass", 2,
             {"frequency_hz": schema.scalar(80.0),
              "quality_factors": schema.vector(quality_factors)},
             (np.abs(response), np.angle(response)),
             scipy.signal.sosfilt(sos, INPUT),
             "4th-order low-pass, 80 Hz, two sections, 1 kHz",
             ["Biquad cascade 4th-order low-pass magnitude and phase",
              "Biquad cascade 4th-order low-pass filtered output"])


def _harmonic_notch(out, meta):
    # 80 Hz and its next two multiples. A 180 Hz fundamental would need a notch at 540 Hz,
    # past half of a 1 kHz sampling rate, which multicalc rejects outright.
    fundamental_hz, quality_factor, sections = 80.0, 4.0, 3
    magnitude = np.ones(len(PROBE_HZ))
    phase = np.zeros(len(PROBE_HZ))
    output = list(INPUT)
    for harmonic in range(1, sections + 1):
        b, a = scipy.signal.iirnotch(fundamental_hz * harmonic, quality_factor, fs=SAMPLE_RATE_HZ)
        section_magnitude, section_phase = _response(b, a)
        magnitude = magnitude * section_magnitude
        phase = phase + section_phase
        output = scipy.signal.lfilter(b, a, output)
    _cascade(out, meta, "biquad_harmonic_notch_80hz", "harmonic_notch", sections,
             {"frequency_hz": schema.scalar(fundamental_hz),
              "quality_factor": schema.scalar(quality_factor)},
             (magnitude, phase), output,
             "Notches on 80, 160, and 240 Hz, sharpness 4, 1 kHz",
             ["Biquad harmonic notch magnitude and phase",
              "Biquad harmonic notch filtered output"])


def run(out, seed):
    meta = schema.metadata(
        "signal_processing", seed,
        "fixed frequencies at a 1 kHz sampling rate; a 5 Hz plus 180 Hz input sequence",
        libraries=("numpy", "scipy"), reference="SciPy {scipy}",
    )
    _sections(out, meta)
    _fourth_order_low_pass(out, meta)
    _harmonic_notch(out, meta)
