from app import discounted


def test_percentage_scale():
    assert discounted(100, 20) == 80
