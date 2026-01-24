import streamlit as st
import pandas as pd
import joblib
from pricing_utils import recommend_fmcg_price
from perishable_sim import perishable_price_recommendation
from perishable_rl_utils import rl_price_recommendation
import matplotlib.pyplot as plt

perishable_products = {
    "Milk":{"id": "P001", "name": "Milk", "cost": 30, "base_price": 40, "shelf_life": 7},
    "Yoghurt":{"id": "P002", "name": "Yogurt", "cost": 28, "base_price": 40, "shelf_life": 10},
    "Cheese":{"id": "P003", "name": "Cheese", "cost": 50, "base_price": 75, "shelf_life": 30},

    "Bread":{"id": "P004", "name": "Bread", "cost": 20, "base_price": 25, "shelf_life": 4},
    "Croissant":{"id": "P005", "name": "Croissant", "cost": 12, "base_price": 22, "shelf_life": 2},
    "Cake Slice":{"id": "P006", "name": "Cake Slice", "cost": 30, "base_price": 45, "shelf_life": 4},

    "Apple":{"id": "P007", "name": "Apple", "cost": 15, "base_price": 30, "shelf_life": 10},
    "Banana":{"id": "P008", "name": "Banana", "cost": 8, "base_price": 15, "shelf_life": 5},
    "Strawberry":{"id": "P009", "name": "Strawberry", "cost": 25, "base_price": 40, "shelf_life": 3},

    "Tomato":{"id": "P010", "name": "Tomato", "cost": 6, "base_price": 12, "shelf_life": 6},
    "Spinach":{"id": "P011", "name": "Spinach", "cost": 5, "base_price": 10, "shelf_life": 3},
    "Mushroom":{"id": "P012", "name": "Mushroom", "cost": 15, "base_price": 28, "shelf_life": 4},

    "Chicken":{"id": "P013", "name": "Chicken", "cost": 120, "base_price": 180, "shelf_life": 5},
    "Fish":{"id": "P014", "name": "Fish", "cost": 150, "base_price": 220, "shelf_life": 5}, #2
    "Paneer":{"id": "P015", "name": "Paneer", "cost": 90, "base_price": 140, "shelf_life": 7},
}

price_elasticity = {
    "P001": 1.2, "P002": 1.0, "P003": 0.6,
    "P004": 1.5, "P005": 1.8, "P006": 1.3,
    "P007": 0.8, "P008": 1.1, "P009": 1.6,
    "P010": 1.0, "P011": 1.9, "P012": 1.4,
    "P013": 0.7, "P014": 0.9, "P015": 0.8
}

expiry_sensitivity = {
    "P001": 1.0, "P002": 0.8, "P003": 0.3,
    "P004": 1.8, "P005": 2.0, "P006": 1.5,
    "P007": 0.5, "P008": 1.2, "P009": 2.2,
    "P010": 1.0, "P011": 2.5, "P012": 1.7,
    "P013": 1.3, "P014": 2.8, "P015": 1.1
}

st.set_page_config(
    page_title="Unified Dynamic Pricing System",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("fmcg_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("D:/NITK Projects/fmcg_dataset.csv")

model = load_model()
df = load_data()

FEATURES = [
    "selling_price", "discount_pct",
    "weekday", "is_weekend", "is_holiday",
    "temperature", "rain_mm",
    "lag_1", "lag_7", "rolling_mean_7",
    "promo_flag", "stock_on_hand"
]

st.markdown(
    "<h3 style='margin-bottom:5px;'>🛒 Unified Dynamic Pricing System for Grocery Retail</h3>",
    unsafe_allow_html=True
)

st.caption(
    "ML-based pricing engine for FMCG & Perishable grocery products "
    "using demand forecasting and optimization."
)
with st.expander("ℹ️ How does this system work?"):
    st.markdown("""
**What problem does this solve?**  
Retailers struggle to decide the *right price* for products every day.
Too high → low sales.  
Too low → low profit.

---

**What does this system do?**  
It recommends the *optimal price* that maximizes profit by:

- Predicting demand using machine learning  
- Testing multiple candidate prices  
- Choosing the price with highest expected profit  

---

**Two pricing engines are used:**

🛒 **FMCG (Non-Perishable Products)**  
- Uses real sales data  
- ML model predicts demand  
- Optimizes price for profit  

🥦 **Perishable Products**  
- Uses expiry-aware simulation  
- Minimizes waste + maximizes revenue  
- (Rule-based now, RL later)

---

**Key business question answered:**

> *“At what price should I sell this product today to maximize profit?”*
""")


engine = st.sidebar.selectbox("Select Pricing Engine", ["FMCG Pricing", "Perishable Pricing"])

# -----------------------
# FMCG MODULE
# -----------------------
if engine == "FMCG Pricing":
    st.subheader("FMCG Dynamic Pricing Engine")

    sku = st.selectbox("Select SKU", df["sku_id"].unique())
    store = st.selectbox("Select Store", df["store_id"].unique())

    row = df[(df["sku_id"] == sku) & (df["store_id"] == store)].iloc[-1]

    st.subheader("Current Product Snapshot")
    st.write(row[["sku_name", "category", "selling_price", "units_sold", "stock_on_hand"]])


    if st.button("Recommend Optimal Price"):
        best_price, best_revenue, best_profit, curve_df = recommend_fmcg_price(row, model, FEATURES)

        # -------------------------------
        # Key Metrics Panel
        # -------------------------------
        st.subheader("📊 Pricing Recommendation Summary")

        col1, col2, col3 = st.columns(3)

        current_revenue = row["selling_price"] * row["units_sold"]
        current_profit = (row["selling_price"] - row["purchase_cost"]) * row["units_sold"]

        col1.metric("Current Price", f"₹{row['selling_price']:.2f}")
        col2.metric("Recommended Price", f"₹{best_price:.2f}")
        col3.metric("Price Change", f"{(best_price-row['selling_price'])/row['selling_price']*100:.2f}%")

        st.markdown("### 💰 Business Impact")

        col4, col5, col6 = st.columns(3)
        col4.metric("Current Revenue", f"₹{current_revenue:.2f}")
        col5.metric("Expected Revenue", f"₹{best_revenue:.2f}")
        col6.metric("Revenue Uplift", f"{(best_revenue-current_revenue)/current_revenue*100:.2f}%")

        col7, col8 = st.columns(2)
        col7.metric("Current Profit", f"₹{current_profit:.2f}")
        col8.metric("Expected Profit", f"₹{best_profit:.2f}")

        # -------------------------------
        # Explanation Panel
        # -------------------------------
        st.info(
            "📌 How to interpret this:\n\n"
            "- The system tests multiple candidate prices around the current price.\n"
            "- For each price, it predicts expected demand using an ML model.\n"
            "- It then computes expected profit = (price − cost) × demand.\n"
            "- The recommended price is the one that maximizes long-term profit, "
            "not necessarily the highest price."
        )

        # -------------------------------
        # Plot Price–Demand–Revenue Curves
        # -------------------------------
        st.subheader("📈 Price–Demand–Revenue Relationship")

        fig, ax1 = plt.subplots(figsize=(8, 5))

        ax1.set_xlabel("Price")
        ax1.set_ylabel("Predicted Demand")
        ax1.plot(curve_df["price"].to_numpy(), curve_df["predicted_demand"].to_numpy(), label="Demand", marker="o")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Revenue / Profit")
        ax2.plot(curve_df["price"].to_numpy(), curve_df["revenue"].to_numpy(), label="Revenue", linestyle="--")
        ax2.plot(curve_df["price"].to_numpy(), curve_df["profit"].to_numpy(), label="Profit", linestyle=":")

        # Mark current and recommended price
        ax1.axvline(row["selling_price"], linestyle="--", alpha=0.6, label="Current Price")
        ax1.axvline(best_price, linestyle="-", alpha=0.9, label="Recommended Price")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        st.pyplot(fig)

        st.subheader("🔍 Before vs After Comparison")

        compare_df = pd.DataFrame({
            "Scenario": ["Current", "Optimized"],
            "Price": [row["selling_price"], best_price],
            "Expected Demand": [row["units_sold"], curve_df.loc[curve_df["price"].sub(best_price).abs().idxmin(), "predicted_demand"]],
            "Revenue": [current_revenue, best_revenue],
            "Profit": [current_profit, best_profit]
        })

        st.table(compare_df)



# -----------------------
# PERISHABLE MODULE
# -----------------------
if engine == "Perishable Pricing":
    st.subheader("🥦 Perishable Dynamic Pricing Engine")

    # Select product
    product_name = st.selectbox("Select Product", list(perishable_products.keys()))
    product = perishable_products[product_name]

    # Input current state
    # inventory = st.slider("Current Inventory", 0, 200, 100)
    inventory = st.selectbox("Inventory (boxed)", [10, 30, 80])
    # days_to_expiry = st.slider("Days to Expiry", 0, product["shelf_life"], product["shelf_life"])
    days_to_expiry = st.selectbox("Days to expiry", list(range(product['shelf_life'])))

    # Select pricing strategy
    strategy = st.radio("Select Pricing Strategy", ["Rule-Based", "RL-Based"])

    st.markdown("### 🔍 Current State")
    st.write({
        "Product": product_name,
        "Inventory": inventory,
        "Days to Expiry": days_to_expiry,
        "Base Price": product["base_price"]
    })

    if st.button("Recommend Perishable Price"):
        if strategy == "Rule-Based":
            price = perishable_price_recommendation(product, inventory, days_to_expiry)
            st.success(f"Rule-Based Recommended Price: ₹{price:.2f}")
            st.info("Heuristic pricing based only on expiry thresholds.")

        else:
            rl_price, rl_profit, rl_waste = rl_price_recommendation(
                product=product,
                inventory=inventory,
                days_to_expiry=days_to_expiry,
                price_elasticity=price_elasticity,
                expiry_sensitivity=expiry_sensitivity
            )

            st.success(f"RL Recommended Price: ₹{rl_price:.2f}")

            col1, col2 = st.columns(2)
            col1.metric("Expected Profit", f"₹{rl_profit:.2f}")
            col2.metric("Expected Waste", f"{rl_waste:.2f} units")

            st.info(
                "🤖 This price is chosen by a trained Q-learning agent that maximizes "
                "long-term profit while penalizing food waste."
            )