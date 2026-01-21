import numpy as np

def perishable_price_recommendation(product, inventory, days_to_expiry):
    base_price = product["base_price"]

    if days_to_expiry <= 1:
        return base_price * 0.6
    elif days_to_expiry <= 3:
        return base_price * 0.8
    else:
        return base_price
