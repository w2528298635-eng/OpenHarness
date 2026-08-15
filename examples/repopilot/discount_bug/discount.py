def final_price(total: float, discount_rate: float) -> float:
    """Return the price after applying a discount rate between 0 and 1."""
    if total < 0:
        raise ValueError("total must be non-negative")
    if not 0 <= discount_rate < 1:
        raise ValueError("discount_rate must be between 0 and 1")
    return total * (1 - discount_rate)
