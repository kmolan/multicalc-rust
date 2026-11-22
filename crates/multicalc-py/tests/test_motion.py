import pytest

import multicalc_py

def test_polyline_length_and_lookahead():
    path = multicalc_py.PolylinePath8x2.try_from_points(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    )
    assert path.total_arc_length() == pytest.approx(2.0)
    point = path.lookahead_point(0.0, 0.5)
    assert len(point) == 2
