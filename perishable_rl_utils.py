import numpy as np
import pickle
from perishable_env import PerishablePricingEnv

def load_q_table(product_name):
    filename = f"q_tables/q_{product_name.lower()}.pkl"
    with open(filename, 'rb') as f:
        Q = pickle.load(f)
    return Q

def discretize_inventory(inv):
    if inv < 20:
        return 0
    elif inv < 50:
        return 1
    else:
        return 2

def discretize_price(price, base_price):
    ratio = price / base_price
    if ratio < 0.8:
        return 0
    elif ratio < 1.1:
        return 1
    else:
        return 2

def discretize_state(state, base_price):
    inv, days, price = state
    inv_bin = discretize_inventory(inv)
    price_bin = discretize_price(price, base_price)
    return (inv_bin, int(days), price_bin)

def rl_price_recommendation(product, inventory, days_to_expiry,
                            price_elasticity, expiry_sensitivity):

    Q = load_q_table(product['name'])

    env = PerishablePricingEnv(
        product=product,
        price_elasticity=price_elasticity,
        expiry_sensitivity=expiry_sensitivity,
        initial_inventory=inventory,
        lambda_waste=30.0
    )

    env.inventory = inventory
    env.days_to_expiry = days_to_expiry
    env.price = product["base_price"]

    state = env._get_state()
    dstate = discretize_state(state, product["base_price"])

    if dstate in Q:
        action = np.argmax(Q[dstate])
    else:
        action = 2
        print("No dstate")

    next_state, reward, done, info = env.step(action)

    return info['price'], info['profit'], info['waste']
