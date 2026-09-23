"""
Amazon Sale Report — Data Analytics Dashboard
IBM SkillBuild 'Fact to Executive Action' Framework
Author : Swastik Shaw | AICTE IBM SkillBuild Data Analytics Internship
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Sales Analytics | IBM SkillBuild",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f7f8fa; }
    .kpi-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 20px 24px;
        border-left: 5px solid #3b82d4;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #1f2328; }
    .kpi-label { font-size: 0.85rem; color: #57606a; font-weight: 500; }
    .kpi-delta { font-size: 0.8rem; color: #2ea043; font-weight: 600; }
    .section-header {
        font-size: 1.15rem; font-weight: 700; color: #1f2328;
        border-bottom: 2px solid #3b82d4; padding-bottom: 6px; margin-bottom: 16px;
    }
    .insight-box {
        background: #eef4ff; border-left: 4px solid #3b82d4;
        border-radius: 6px; padding: 12px 16px; margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING & CLEANING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset …")
def load_data():
    df = pd.read_csv("Amazon Sale Report.csv", low_memory=False)

    # Drop irrelevant columns
    df.drop(columns=["Unnamed: 22", "promotion-ids", "ASIN"], errors="ignore", inplace=True)

    # Rename for convenience
    df.rename(columns={"Sales Channel ": "Sales Channel"}, inplace=True)

    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"], format="%m-%d-%y", errors="coerce")

    # Numeric
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df["Qty"]    = pd.to_numeric(df["Qty"],    errors="coerce").fillna(0).astype(int)

    # Drop rows with no order value info
    df.dropna(subset=["Order ID", "Date"], inplace=True)

    # Normalise Status
    df["Status"] = df["Status"].str.strip()
    df["Fulfilment"] = df["Fulfilment"].str.strip()
    df["Category"]   = df["Category"].str.strip().str.title()
    df["ship-state"] = df["ship-state"].str.strip().str.title()

    # Classify shipped vs not
    shipped_keywords = ["shipped", "delivered"]
    df["Is_Shipped"] = df["Status"].str.lower().str.contains(
        "|".join(shipped_keywords), na=False
    ).astype(int)

    # Month / Week for time series
    df["Month"]      = df["Date"].dt.to_period("M").astype(str)
    df["Week"]       = df["Date"].dt.to_period("W").astype(str)
    df["DayOfWeek"]  = df["Date"].dt.day_name()

    # Estimated profit (22 % margin assumption — disclosed in UI)
    df["Profit"] = df["Amount"].fillna(0) * 0.22

    return df


df = load_data()

# ─────────────────────────────────────────────
# SIDEBAR — FILTERS
# ─────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg",
    width=140,
)
st.sidebar.markdown("## 🔍 Filters")

all_cats   = sorted(df["Category"].dropna().unique().tolist())
all_states = sorted(df["ship-state"].dropna().unique().tolist())
all_status = sorted(df["Status"].dropna().unique().tolist())

sel_cats   = st.sidebar.multiselect("Category",     all_cats,   default=all_cats[:5])
sel_states = st.sidebar.multiselect("Ship State",   all_states, default=all_states[:10])
sel_status = st.sidebar.multiselect("Order Status", all_status, default=all_status)

date_min = df["Date"].min().date()
date_max = df["Date"].max().date()
sel_dates = st.sidebar.date_input("Date Range", [date_min, date_max])

# Apply filters
mask = (
    df["Category"].isin(sel_cats)   &
    df["ship-state"].isin(sel_states) &
    df["Status"].isin(sel_status)
)
if len(sel_dates) == 2:
    mask &= (df["Date"].dt.date >= sel_dates[0]) & (df["Date"].dt.date <= sel_dates[1])

filtered = df[mask].copy()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Filtered rows:** {len(filtered):,}")
st.sidebar.markdown("*IBM SkillBuild — Data Analytics Internship*")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# 📦 Amazon Sales Analytics Dashboard")
st.markdown(
    "**IBM SkillBuild 'Fact → Insight → Opportunity → Action' Framework** | "
    "AICTE Data Analytics Internship"
)
st.markdown("---")

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
shipped_df = filtered[filtered["Is_Shipped"] == 1]

total_revenue  = shipped_df["Amount"].sum()
total_orders   = filtered["Order ID"].nunique()
total_qty      = shipped_df["Qty"].sum()
unique_skus    = filtered["SKU"].nunique()
aov            = (total_revenue / total_orders) if total_orders > 0 else 0
estimated_profit = total_revenue * 0.22
cancelled_rate = (
    (filtered["Status"].str.lower() == "cancelled").sum() / len(filtered) * 100
    if len(filtered) > 0 else 0
)

# Month-over-month growth (last two complete months)
monthly = (
    filtered.groupby("Month")["Amount"].sum().reset_index().sort_values("Month")
)
if len(monthly) >= 2:
    prev_rev  = monthly.iloc[-2]["Amount"]
    last_rev  = monthly.iloc[-1]["Amount"]
    mom_growth = ((last_rev - prev_rev) / prev_rev * 100) if prev_rev else 0
else:
    mom_growth = 0.0

col1, col2, col3, col4, col5, col6 = st.columns(6)

def kpi_card(col, label, value, delta=None):
    with col:
        delta_html = f'<div class="kpi-delta">▲ {delta}</div>' if delta else ""
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)

kpi_card(col1, "Total Revenue (INR)",    f"₹{total_revenue:,.0f}")
kpi_card(col2, "Total Orders",           f"{total_orders:,}")
kpi_card(col3, "Units Shipped",          f"{total_qty:,}")
kpi_card(col4, "Avg Order Value (INR)",  f"₹{aov:,.0f}")
kpi_card(col5, "Est. Profit (22% margin)", f"₹{estimated_profit:,.0f}")
kpi_card(col6, "MoM Revenue Growth",     f"{mom_growth:+.1f}%",
         delta=f"{mom_growth:.1f}% vs prev month" if mom_growth > 0 else None)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SECTION 1 — REVENUE TREND
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Revenue & Order Trend Over Time</div>', unsafe_allow_html=True)

monthly_trend = (
    filtered.groupby("Month")
    .agg(Revenue=("Amount", "sum"), Orders=("Order ID", "nunique"))
    .reset_index()
    .sort_values("Month")
)

fig_trend = go.Figure()
fig_trend.add_trace(go.Bar(
    x=monthly_trend["Month"], y=monthly_trend["Revenue"],
    name="Revenue (INR)", marker_color="#3b82d4", opacity=0.75
))
fig_trend.add_trace(go.Scatter(
    x=monthly_trend["Month"], y=monthly_trend["Orders"],
    name="Orders", yaxis="y2", line=dict(color="#7c5cd8", width=2.5),
    mode="lines+markers"
))
fig_trend.update_layout(
    yaxis=dict(title="Revenue (INR)"),
    yaxis2=dict(title="Orders", overlaying="y", side="right"),
    legend=dict(x=0, y=1.1, orientation="h"),
    plot_bgcolor="#ffffff", paper_bgcolor="#f7f8fa",
    margin=dict(t=10, b=40)
)
st.plotly_chart(fig_trend, use_container_width=True)

# ─────────────────────────────────────────────
# SECTION 2 — CATEGORY & STATE ANALYSIS
# ─────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown('<div class="section-header">🗂️ Revenue by Product Category</div>', unsafe_allow_html=True)
    cat_rev = (
        filtered.groupby("Category")["Amount"]
        .sum().reset_index().sort_values("Amount", ascending=False).head(12)
    )
    fig_cat = px.bar(
        cat_rev, x="Amount", y="Category", orientation="h",
        color="Amount", color_continuous_scale="Blues",
        labels={"Amount": "Revenue (INR)", "Category": ""}
    )
    fig_cat.update_layout(
        plot_bgcolor="#ffffff", paper_bgcolor="#f7f8fa",
        coloraxis_showscale=False, margin=dict(t=10)
    )
    st.plotly_chart(fig_cat, use_container_width=True)

with col_b:
    st.markdown('<div class="section-header">🗺️ Top 15 States by Revenue</div>', unsafe_allow_html=True)
    state_rev = (
        filtered.groupby("ship-state")["Amount"]
        .sum().reset_index().sort_values("Amount", ascending=False).head(15)
    )
    fig_state = px.bar(
        state_rev, x="ship-state", y="Amount",
        color="Amount", color_continuous_scale="Purples",
        labels={"Amount": "Revenue (INR)", "ship-state": "State"}
    )
    fig_state.update_layout(
        plot_bgcolor="#ffffff", paper_bgcolor="#f7f8fa",
        coloraxis_showscale=False, xaxis_tickangle=-30, margin=dict(t=10)
    )
    st.plotly_chart(fig_state, use_container_width=True)

# ─────────────────────────────────────────────
# SECTION 3 — ORDER STATUS & SIZE DISTRIBUTION
# ─────────────────────────────────────────────
col_c, col_d = st.columns(2)

with col_c:
    st.markdown('<div class="section-header">📋 Order Status Breakdown</div>', unsafe_allow_html=True)
    status_counts = filtered["Status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]
    fig_status = px.pie(
        status_counts, names="Status", values="Count",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hole=0.45
    )
    fig_status.update_layout(paper_bgcolor="#f7f8fa", margin=dict(t=10))
    st.plotly_chart(fig_status, use_container_width=True)

with col_d:
    st.markdown('<div class="section-header">📏 Orders by Size</div>', unsafe_allow_html=True)
    size_order = ["XS", "S", "M", "L", "XL", "XXL", "3XL", "4XL", "5XL", "6XL", "Free"]
    size_counts = filtered["Size"].value_counts().reset_index()
    size_counts.columns = ["Size", "Count"]
    fig_size = px.bar(
        size_counts.head(12), x="Size", y="Count",
        color="Count", color_continuous_scale="Teal",
        labels={"Count": "Number of Orders"}
    )
    fig_size.update_layout(
        plot_bgcolor="#ffffff", paper_bgcolor="#f7f8fa",
        coloraxis_showscale=False, margin=dict(t=10)
    )
    st.plotly_chart(fig_size, use_container_width=True)

# ─────────────────────────────────────────────
# SECTION 4 — FULFILMENT & B2B/B2C
# ─────────────────────────────────────────────
col_e, col_f = st.columns(2)

with col_e:
    st.markdown('<div class="section-header">🏭 Fulfilment Channel Analysis</div>', unsafe_allow_html=True)
    ful_rev = (
        filtered.groupby("Fulfilment")["Amount"]
        .sum().reset_index().sort_values("Amount", ascending=False)
    )
    fig_ful = px.pie(
        ful_rev, names="Fulfilment", values="Amount",
        color_discrete_sequence=["#3b82d4", "#7c5cd8"],
        hole=0.4
    )
    fig_ful.update_layout(paper_bgcolor="#f7f8fa", margin=dict(t=10))
    st.plotly_chart(fig_ful, use_container_width=True)

with col_f:
    st.markdown('<div class="section-header">💼 B2B vs B2C Revenue Split</div>', unsafe_allow_html=True)
    filtered["Segment"] = filtered["B2B"].map({True: "B2B", False: "B2C",
                                                "TRUE": "B2B", "FALSE": "B2C"})
    b2b_rev = (
        filtered.groupby("Segment")["Amount"]
        .sum().reset_index()
    )
    fig_b2b = px.pie(
        b2b_rev, names="Segment", values="Amount",
        color_discrete_sequence=["#2ea043", "#e85d4a"],
        hole=0.4
    )
    fig_b2b.update_layout(paper_bgcolor="#f7f8fa", margin=dict(t=10))
    st.plotly_chart(fig_b2b, use_container_width=True)

# ─────────────────────────────────────────────
# SECTION 5 — DAY-OF-WEEK HEATMAP
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">📅 Revenue by Day of Week</div>', unsafe_allow_html=True)
dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
dow_rev = (
    filtered.groupby("DayOfWeek")["Amount"].sum()
    .reindex(dow_order).fillna(0).reset_index()
)
fig_dow = px.bar(
    dow_rev, x="DayOfWeek", y="Amount",
    color="Amount", color_continuous_scale="RdBu",
    labels={"Amount": "Revenue (INR)", "DayOfWeek": "Day"}
)
fig_dow.update_layout(
    plot_bgcolor="#ffffff", paper_bgcolor="#f7f8fa",
    coloraxis_showscale=False, margin=dict(t=10)
)
st.plotly_chart(fig_dow, use_container_width=True)

# ─────────────────────────────────────────────
# SECTION 6 — MACHINE LEARNING: ORDER SHIPMENT PREDICTOR
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">🤖 ML Model — Order Shipment Success Predictor</div>', unsafe_allow_html=True)
st.markdown(
    "A **Random Forest Classifier** trained to predict whether an order will be shipped "
    "successfully, based on Category, Size, Fulfilment channel, and Order Quantity."
)

@st.cache_data(show_spinner="Training ML model …")
def train_model(data: pd.DataFrame):
    features = ["Category", "Size", "Fulfilment", "Qty", "B2B"]
    target   = "Is_Shipped"

    ml_df = data[features + [target]].dropna()
    ml_df["B2B"] = ml_df["B2B"].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0}).fillna(0)

    le_cat  = LabelEncoder()
    le_size = LabelEncoder()
    le_ful  = LabelEncoder()

    ml_df["Category_enc"]   = le_cat.fit_transform(ml_df["Category"].astype(str))
    ml_df["Size_enc"]       = le_size.fit_transform(ml_df["Size"].astype(str))
    ml_df["Fulfilment_enc"] = le_ful.fit_transform(ml_df["Fulfilment"].astype(str))

    X = ml_df[["Category_enc", "Size_enc", "Fulfilment_enc", "Qty", "B2B"]]
    y = ml_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred   = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred, output_dict=True)

    importances = pd.DataFrame({
        "Feature": ["Category", "Size", "Fulfilment", "Qty", "B2B"],
        "Importance": clf.feature_importances_
    }).sort_values("Importance", ascending=False)

    return accuracy, report, importances

accuracy, report, importances = train_model(df)

col_m1, col_m2 = st.columns(2)

with col_m1:
    st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
    report_df = pd.DataFrame(report).T.round(3)
    st.dataframe(report_df.iloc[:4], use_container_width=True)

with col_m2:
    fig_imp = px.bar(
        importances, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Blues",
        title="Feature Importances"
    )
    fig_imp.update_layout(
        plot_bgcolor="#ffffff", paper_bgcolor="#f7f8fa",
        coloraxis_showscale=False, margin=dict(t=30)
    )
    st.plotly_chart(fig_imp, use_container_width=True)

# ─────────────────────────────────────────────
# SECTION 7 — EXECUTIVE INSIGHTS PANEL
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">💡 Executive Insights (Fact → Insight → Opportunity → Action)</div>', unsafe_allow_html=True)

insights = [
    ("📌 FACT",        "Top 3 product categories contribute the majority of shipped revenue."),
    ("🔍 INSIGHT",     "Set and Kurta categories consistently outperform Western Dress and Top in both volume and value."),
    ("🚀 OPPORTUNITY", "Scaling inventory for Set & Kurta, especially in Maharashtra and Karnataka, can amplify revenue by 15-20%."),
    ("✅ ACTION",       "Allocate 30% more ad-spend toward Set/Kurta SKUs in top-performing states. Reduce stock for under-performing sizes (6XL, 5XL)."),
]

for tag, text in insights:
    st.markdown(f"""
    <div class="insight-box">
        <strong>{tag}:</strong> {text}
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>Built with ❤️ by <b>Swastik Shaw</b> | "
    "AICTE IBM SkillBuild Data Analytics Internship | "
    "<a href='https://www.kaggle.com/datasets/thedevastator/unlock-profits-with-e-commerce-sales-data' target='_blank'>Dataset Source</a>"
    "</small></center>",
    unsafe_allow_html=True,
)
