from app import clamp


def test_upper_boundary_is_valid():
    assert clamp(10, 10) == 10
