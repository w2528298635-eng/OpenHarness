from app import parse_count


def test_none_is_zero():
    assert parse_count(None) == 0
