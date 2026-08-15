from app import normalize_username


def test_normalizes_whitespace_and_case():
    assert normalize_username("  Alice ") == "alice"
