from helpers import as_rate


def discounted(total, percent):
    return total * (1 - as_rate(percent))
