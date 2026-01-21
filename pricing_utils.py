import numpy as np

def recommend_fmcg_price(row, model, features, price_range_pct=0.15, n_prices=15):
    base_price = row["selling_price"]
    cost = row["purchase_cost"]

    min_price = max(cost * 1.05, base_price * (1 - price_range_pct))
    max_price = base_price * (1 + price_range_pct)

    candidate_prices = np.linspace(min_price, max_price, n_prices)

    best_price = base_price
    best_revenue = 0

    for price in candidate_prices:
        temp = row.copy()
        temp["selling_price"] = price
        temp["discount_pct"] = (row["list_price"] - price) / row["list_price"] * 100

        X = temp[features].values.reshape(1, -1)
        predicted_demand = model.predict(X)[0]
        revenue = price * predicted_demand

        if revenue > best_revenue:
            best_revenue = revenue
            best_price = price

    return best_price, best_revenue
