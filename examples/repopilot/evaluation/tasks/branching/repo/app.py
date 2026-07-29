def shipping_cost(total, vip=False):
    if total >= 100:
        return 0
    return 10
