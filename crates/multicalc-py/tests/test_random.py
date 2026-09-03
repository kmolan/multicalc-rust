import multicalc_py

def test_pcg32_repeatable():
    first = [multicalc_py.Pcg32(1).next_unit() for _ in range(3)]
    second = [multicalc_py.Pcg32(1).next_unit() for _ in range(3)]
    assert first == second
    assert 0.0 <= first[0] < 1.0
    