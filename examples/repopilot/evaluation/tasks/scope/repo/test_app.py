from app import total_with_tax


def test_currency_rounding():
    assert total_with_tax(0.1, 0.2) == 0.12
