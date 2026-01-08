# streamlit_app/app.py
import sys
from pathlib import Path

# Ensure root project directory is in Python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.elasticity import estimate_loglog_ols
from src.simulator import find_revenue_max_price

# -------------------------------------------------
# Correct file paths
# -------------------------------------------------
DATA = ROOT / "data" / "online_retail_II.xlsx"
AGG_FILE = ROOT / "data" / "agg_weekly_per_sku.csv"

# Debug — shows Streamlit the exact path it is loading from
st.write("Current working directory:", Path.cwd())
st.write("Looking for:", AGG_FILE)
st.write("Data folder contents:", list((ROOT / "data").iterdir()))

# -------------------------------------------------
# Load Aggregated CSV
# -------------------------------------------------
@st.cache_data
def load_agg():
    return pd.read_csv(AGG_FILE, parse_dates=['week'])

agg = load_agg()

# SKUs
skus = sorted(agg['StockCode'].unique())
sku = st.sidebar.selectbox("Select SKU", skus)

sku_df = agg[agg['StockCode'] == sku].sort_values('week')

# -------------------------------------------------
# Page Title
# -------------------------------------------------
st.title(f"SKU: {sku} — Price Elasticity & Revenue Simulator")

# -------------------------------------------------
# Elasticity estimation
# -------------------------------------------------
res = estimate_loglog_ols(sku_df)
st.metric("Elasticity (beta on log price)", f"{res['elasticity']:.3f}")
st.write("95% CI:", res['ci'])

# -------------------------------------------------
# Scatter plot with trendline
# -------------------------------------------------
sku_df['log_q'] = np.log(sku_df['units'])
sku_df['log_p'] = np.log(sku_df['avg_price'])

fig = px.scatter(
    sku_df,
    x='log_p',
    y='log_q',
    trendline='ols',
    title='Log(price) vs log(quantity)'
)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# Revenue simulation
# -------------------------------------------------
p0 = sku_df['avg_price'].iloc[-1]
a = res['model'].params.get('Intercept', 0)
C = np.exp(a)

elasticity_boots = np.random.normal(
    res['elasticity'],
    scale=(res['ci'][1] - res['ci'][0]) / 4,
    size=500
)

sim = find_revenue_max_price(
    res['elasticity'],
    elasticity_boots,
    C,
    p0,
    step=max(0.01, p0 * 0.01)
)

grid = sim['grid']
rev = sim['rev_grid']
rev_df = pd.DataFrame({'price': grid, 'revenue': rev})

fig2 = px.line(
    rev_df,
    x='price',
    y='revenue',
    title='Price vs Expected Revenue'
)
fig2.add_vline(
    x=sim['best_price'],
    line_dash='dash',
    annotation_text=f"Recommended: {sim['best_price']:.2f}"
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown(f"**Recommended price (point estimate):** {sim['best_price']:.2f}")
st.markdown(f"**Recommended price (95% CI):** {sim['price_ci'][0]:.2f} — {sim['price_ci'][1]:.2f}")

# -------------------------------------------------
# Export recommendation
# -------------------------------------------------
if st.button("Export recommendation CSV"):
    out = pd.DataFrame([{
        'StockCode': sku,
        'recommended_price': sim['best_price'],
        'ci_lower': sim['price_ci'][0],
        'ci_upper': sim['price_ci'][1]
    }])
    out.to_csv('sku_recommendation.csv', index=False)
    st.success("Exported sku_recommendation.csv")
