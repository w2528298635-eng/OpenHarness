from app import with_default


def test_does_not_mutate_input():
    source = ["a"]
    assert with_default(source) == ["a", "default"]
    assert source == ["a"]
