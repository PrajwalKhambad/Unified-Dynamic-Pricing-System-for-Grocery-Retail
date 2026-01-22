import numpy as np
class PerishablePricingEnv:
    def __init__(self, product, price_elasticity, expiry_sensitivity, 
                 initial_inventory=100, lambda_waste=5.0):
        
        """
        product: dict with keys [id, base_price, cost, shelf_life]
        """

        self.product = product
        self.price_elasticity = price_elasticity
        self.expiry_sensitivity = expiry_sensitivity

        self.initial_inventory = initial_inventory
        self.lambda_waste = lambda_waste

        self.price_factors = [-0.2, -0.1, 0.0, 0.1]

        self.reset()
        
    # reset environment
    def reset(self):
        self.inventory = self.initial_inventory
        self.days_to_expiry = self.product['shelf_life']
        self.price = self.product['base_price']

        state = self._get_state()
        return state
    
    # get current state
    def _get_state(self):
        return np.array([
          self.inventory,
          self.days_to_expiry,
          self.price  
        ], dtype = np.float32)
    
    # demand simulation
    def _simulate_demand(self, price):
        base_demand = np.random.poisson(30)

        alpha = self.price_elasticity[self.product['id']]
        beta = self.expiry_sensitivity[self.product['id']]

        price_effect = np.exp(-alpha * (price / self.product['base_price']))
        expiry_effect = np.exp(-beta * (self.days_to_expiry / self.product['shelf_life']))

        expected_demand = base_demand * price_effect * expiry_effect
        demand = min(np.random.poisson(expected_demand), self.inventory)

        return demand
    
    # step function
    def step(self, action_id):
        price_change = self.price_factors[action_id]
        new_price = self.price * (1 + price_change)

        # enforce minimum margin
        min_price = self.product['cost'] * 1.05
        self.price = max(new_price, min_price)

        units_sold = self._simulate_demand(self.price)

        self.inventory -= units_sold

        if self.days_to_expiry == 0:
            waste = self.inventory
        else:
            waste = 0

        # compute reward
        profit = (self.price - self.product['cost']) * units_sold
        reward = profit - self.lambda_waste * waste

        self.inventory -= waste
        self.days_to_expiry += 1

        done = False

        if self.days_to_expiry<0 or self.initial_inventory<=0:
            done = True

        next_state = self._get_state()

        info = {
            'units_sold': units_sold,
            'waste': waste,
            'price': self.price,
            'profit': profit
        }

        return next_state, reward, done, info