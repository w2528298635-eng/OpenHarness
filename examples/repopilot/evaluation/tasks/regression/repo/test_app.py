from app import average


def test_fraction_is_preserved():
    assert average(5, 2) == 2.5


def test_even_average_still_works():
    assert average(4, 2) == 2
