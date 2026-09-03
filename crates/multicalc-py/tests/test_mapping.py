import math

import pytest

import multicalc_py

def test_scan_middle_beam_ahead():
    scan = multicalc_py.ScanGeometry5(math.pi / 2, 4.0)
    assert scan.beam_angle(2) == pytest.approx(0.0)

def test_dynamic_grid_blocks_a_cell():
    grid = multicalc_py.DynamicOccupancyGrid(8, 8, 1.0, [0.0, 0.0])
    grid.occupy_point([3.5, 4.5])
    assert grid.is_occupied(4, 3) is True
    hit = grid.cast_ray([0.5, 4.5], 0.0, 10.0)
    assert hit is not None
