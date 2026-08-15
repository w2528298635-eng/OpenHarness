import pytest

from discount import final_price


def test_full_discount_returns_zero() -> None:
    assert final_price(100, 1) == 0


def test_rate_above_one_is_rejected() -> None:
    with pytest.raises(ValueError):
        final_price(100, 1.01)
