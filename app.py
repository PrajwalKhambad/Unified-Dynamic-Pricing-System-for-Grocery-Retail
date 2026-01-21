import streamlit as st
import pandas as pd
import joblib
from pricing_utils import recommend_fmcg_price
from perishable_sim import perishable_price_recommendation
import matplotlib.pyplot as plt

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
    st.header("🥦 Perishable Dynamic Pricing Engine")

    product = st.selectbox("Select Product", ["Milk", "Bread", "Apple"])

    base_prices = {"Milk": 30, "Bread": 25, "Apple": 20}

    inventory = st.slider("Inventory", 0, 200, 100)
    days_to_expiry = st.slider("Days to Expiry", 0, 10, 5)

    product_info = {
        "name": product,
        "base_price": base_prices[product]
    }

    recommended_price = perishable_price_recommendation(product_info, inventory, days_to_expiry)

    st.subheader("Recommended Price")
    st.success(f"₹{recommended_price:.2f}")

    st.caption("Note: This uses rule-based pricing. Replace with RL agent later.")
