import streamlit as st
import pandas as pd
import joblib
from pricing_utils import recommend_fmcg_price
from perishable_sim import perishable_price_recommendation

# -----------------------
# Load FMCG Components
# -----------------------
@st.cache_resource
def load_model():
    return joblib.load("fmcg_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("fmcg_dataset.csv")

model = load_model()
df = load_data()

FEATURES = [
    "selling_price", "discount_pct",
    "weekday", "is_weekend", "is_holiday",
    "temperature", "rain_mm",
    "lag_1", "lag_7", "rolling_mean_7",
    "promo_flag", "stock_on_hand"
]

# -----------------------
# Streamlit UI
# -----------------------
st.title("🛒 Unified Dynamic Pricing System for Grocery Retail")

engine = st.sidebar.selectbox("Select Pricing Engine", ["FMCG Pricing", "Perishable Pricing"])

# -----------------------
# FMCG MODULE
# -----------------------
if engine == "FMCG Pricing":
    st.header("📦 FMCG Dynamic Pricing Engine")

    sku = st.selectbox("Select SKU", df["sku_id"].unique())
    store = st.selectbox("Select Store", df["store_id"].unique())

    row = df[(df["sku_id"] == sku) & (df["store_id"] == store)].iloc[-1]

    st.subheader("Current Product Snapshot")
    st.write(row[["sku_name", "category", "selling_price", "units_sold", "stock_on_hand"]])

    if st.button("Recommend Optimal Price"):
        best_price, best_revenue = recommend_fmcg_price(row, model, FEATURES)

        st.success(f"Recommended Price: ₹{best_price:.2f}")
        st.info(f"Expected Revenue: ₹{best_revenue:.2f}")

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
