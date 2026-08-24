import pytest

import multicalc_py

def identity(x, y):
    return [x, y]

def test_particle_filter_stays_near_measurement():
    filt = multicalc_py.ParticleFilter2x2(
        64,
        [0.0, 0.0],
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.01, 0.0], [0.0, 0.01]],
        [[0.1, 0.0], [0.0, 0.1]],
        1,
    )
    for _ in range(8):
        filt.predict(identity)
        filt.update(identity, [1.0, 2.0])
    mean = filt.mean()
    assert abs(mean[0] - 1.0) < 0.5
    assert abs(mean[1] - 2.0) < 0.5
