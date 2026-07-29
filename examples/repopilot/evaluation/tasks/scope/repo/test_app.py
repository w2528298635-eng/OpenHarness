from app import total_with_tax


def test_currency_rounding():
    assert total_with_tax(10.01, 0.075) == 10.76
