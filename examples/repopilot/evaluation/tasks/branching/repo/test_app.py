from app import shipping_cost


def test_vip_shipping_is_free():
    assert shipping_cost(20, vip=True) == 0
