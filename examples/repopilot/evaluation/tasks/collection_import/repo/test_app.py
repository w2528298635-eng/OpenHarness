from app import greeting


def test_greeting():
    assert greeting("Ada") == "Hello, Ada"
