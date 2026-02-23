import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pathlib import Path

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
DEFAULT_PATH = Path(r"C:\Users\kanta\Downloads\Online Retail.csv")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Online Retail Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CSS  –  clean light theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family:'Inter',sans-serif!important; background:#f0f2f8!important; color:#1a1d3a!important; }
#MainMenu, footer { visibility:hidden; }
.block-container { padding:1.4rem 2rem!important; max-width:100%!important; }

.ph { background:#fff; border-radius:16px; padding:18px 28px; margin-bottom:18px;
      box-shadow:0 2px 14px rgba(26,29,58,.07); display:flex; align-items:center; justify-content:space-between; }
.ph h1 { margin:0!important; font-size:21px!important; font-weight:700!important; color:#1a1d3a!important; }
.ph p  { margin:4px 0 0!important; font-size:13px!important; color:#8b90b8!important; }

.kpi { background:#fff; border-radius:14px; padding:18px 20px; text-align:center;
       box-shadow:0 2px 12px rgba(26,29,58,.07); transition:transform .18s,box-shadow .18s; }
.kpi:hover { transform:translateY(-3px); box-shadow:0 8px 24px rgba(46,65,212,.13); }
.kpi-lbl  { font-size:10px; letter-spacing:1px; text-transform:uppercase; color:#8b90b8; margin-bottom:5px; }
.kpi-val  { font-size:28px; font-weight:700; color:#1a1d3a; }
.kpi-sub  { font-size:11px; color:#8b90b8; margin-top:3px; }

.ct  { font-size:15px; font-weight:600; color:#1a1d3a; margin-bottom:2px; }
.cs  { font-size:12px; color:#8b90b8; margin-bottom:10px; }

.prog-row { display:flex; align-items:center; gap:10px; margin-bottom:10px; }
.prog-lbl { width:140px; font-size:12px; font-weight:500; color:#1a1d3a; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.prog-cnt { width:48px; font-size:12px; color:#8b90b8; text-align:right; }
.prog-wrap{ flex:1; background:#eef0f8; border-radius:999px; height:7px; }
.prog-fill{ height:7px; border-radius:999px; }
.prog-pct { width:34px; font-size:11px; color:#8b90b8; text-align:right; }

.dl  { display:flex; align-items:center; gap:5px; font-size:12px; color:#1a1d3a; margin-bottom:5px; }
.dot { width:10px; height:10px; border-radius:50%; display:inline-block; flex-shrink:0; }
.pill{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:11px; font-weight:600; background:#e8f0fe; color:#2a4ed4; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
BLUE = ["#1a1d8c","#2e41d4","#5068e8","#7b93f5","#a8b8fb","#d0d9fd","#e0e7ff"]
SEG_COLORS = {
    "Champions":        "#1a1d8c",
    "Loyal Customers":  "#2e41d4",
    "Potential Loyal":  "#5068e8",
    "Recent Customers": "#7b93f5",
    "Promising":        "#a8b8fb",
    "At Risk":          "#e86f7b",
    "Lost":             "#f09fa7",
}
_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#1a1d3a", size=12),
    margin=dict(t=8, b=8, l=0, r=0),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
)
def CL(**kw):
    """Merge base layout + per-chart overrides. Prevents duplicate-kwarg errors."""
    return {**_BASE, **kw}


REQUIRED_COLS = {"InvoiceNo","StockCode","Description","Quantity",
                 "InvoiceDate","UnitPrice","CustomerID","Country"}

# ─────────────────────────────────────────────
#  DATA LOADING & CLEANING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading Online Retail dataset…")
def load_default() -> pd.DataFrame:
    df = read_csv_smart(DEFAULT_PATH)
    return clean_df(df)

def read_csv_smart(src) -> pd.DataFrame:
    """Try utf-8 → latin-1 → cp1252 to handle £ and other non-ASCII bytes."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            if hasattr(src, "seek"):
                src.seek(0)
            return pd.read_csv(src, encoding=enc)
        except (UnicodeDecodeError, ValueError):
            continue
    if hasattr(src, "seek"):
        src.seek(0)
    return pd.read_csv(src, encoding="utf-8", errors="replace")

@st.cache_data(show_spinner="Processing uploaded file…")
def load_uploaded(raw_bytes: bytes, fname: str) -> pd.DataFrame:
    import io
    if fname.lower().endswith(".xlsx") or fname.lower().endswith(".xls"):
        try:
            import openpyxl  # noqa: F401
            df = pd.read_excel(io.BytesIO(raw_bytes), engine="openpyxl")
        except ImportError:
            st.error("❌ openpyxl not installed. Run: `pip install openpyxl`")
            st.stop()
    else:
        df = read_csv_smart(io.BytesIO(raw_bytes))
    return clean_df(df)

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df.dropna(subset=["CustomerID","InvoiceDate"], inplace=True)
    df["CustomerID"] = df["CustomerID"].astype(int)
    # Remove cancellations (InvoiceNo starting with C) and bad prices/qty
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["Revenue"]  = df["Quantity"] * df["UnitPrice"]
    df["Date"]     = df["InvoiceDate"].dt.date
    df["YearMonth"]= df["InvoiceDate"].dt.to_period("M").astype(str)
    df["Hour"]     = df["InvoiceDate"].dt.hour
    df["DayOfWeek"]= df["InvoiceDate"].dt.day_name()
    return df

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Data Source")
    uploaded = st.file_uploader(
        "Upload Excel / CSV",
        type=["xlsx","xls","csv"],
        help="Required: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country",
    )

    if uploaded:
        try:
            df_raw = load_uploaded(uploaded.read(), uploaded.name)
            miss = REQUIRED_COLS - set(df_raw.columns)
            if miss:
                st.error(f"❌ Missing: {', '.join(sorted(miss))}")
                df_raw = load_default()
            else:
                st.success(f"✅ {uploaded.name} — {len(df_raw):,} rows")
        except Exception as e:
            st.error(f"❌ {e}")
            df_raw = load_default()
    else:
        df_raw = load_default()
        st.info("🛒 Using Online Retail demo dataset")

    with st.expander("🔍 Preview data"):
        st.dataframe(df_raw.head(8), use_container_width=True)
        st.download_button(
            "⬇️ Download cleaned CSV",
            df_raw.to_csv(index=False).encode(),
            "online_retail_clean.csv","text/csv",
        )

    st.divider()
    st.markdown("### 🎛️ Filters")

    all_countries = sorted(df_raw["Country"].unique())
    top_countries = df_raw["Country"].value_counts().head(10).index.tolist()
    sel_countries = st.multiselect("Countries", all_countries, default=top_countries)

    d_min = df_raw["Date"].min()
    d_max = df_raw["Date"].max()
    date_range = st.date_input("Date Range", value=(d_min, d_max),
                               min_value=d_min, max_value=d_max)

    n_clusters = st.slider("RFM Clusters (k)", 2, 8, 4)

    st.divider()
    st.info("E-commerce RFM Analytics powered by KMeans clustering")

# ─────────────────────────────────────────────
#  APPLY FILTERS
# ─────────────────────────────────────────────
d1, d2 = (date_range[0], date_range[1]) if len(date_range) == 2 else (d_min, d_max)

df = df_raw[
    df_raw["Country"].isin(sel_countries) &
    (df_raw["Date"] >= d1) &
    (df_raw["Date"] <= d2)
].copy()

if df.empty:
    st.warning("⚠️ No data after filters. Adjust the sidebar.")
    st.stop()

# ─────────────────────────────────────────────
#  RFM COMPUTATION
# ─────────────────────────────────────────────
today = pd.Timestamp(df["InvoiceDate"].max().date())

rfm = df.groupby("CustomerID").agg(
    Recency   = ("InvoiceDate", lambda x: (today - x.max()).days),
    Frequency = ("InvoiceNo",   "nunique"),
    Monetary  = ("Revenue",     "sum"),
).reset_index()

# KMeans clustering
scaler    = StandardScaler()
X_scaled  = scaler.fit_transform(rfm[["Recency","Frequency","Monetary"]])
km        = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
rfm["Cluster"] = km.fit_predict(X_scaled).astype(str)

def rfm_segment(row, r_q, f_q, m_q):
    r = row["Recency"];  f = row["Frequency"];  m = row["Monetary"]
    if r <= r_q[0] and f >= f_q[1] and m >= m_q[1]:  return "Champions"
    elif f >= f_q[1] and m >= m_q[0]:                 return "Loyal Customers"
    elif r <= r_q[0] and f < f_q[0]:                  return "Recent Customers"
    elif r <= r_q[1] and m >= m_q[0]:                 return "Potential Loyal"
    elif r <= r_q[1]:                                  return "Promising"
    elif r > r_q[1] and f >= f_q[0]:                  return "At Risk"
    else:                                              return "Lost"

r_q = rfm["Recency"].quantile([.25,.75]).values
f_q = rfm["Frequency"].quantile([.25,.75]).values
m_q = rfm["Monetary"].quantile([.25,.75]).values

rfm["Segment"] = rfm.apply(rfm_segment, axis=1, args=(r_q, f_q, m_q))

# ─────────────────────────────────────────────
#  AGGREGATE KPIs
# ─────────────────────────────────────────────
total_revenue   = df["Revenue"].sum()
n_orders        = df["InvoiceNo"].nunique()
n_customers     = df["CustomerID"].nunique()
n_products      = df["StockCode"].nunique()
avg_order_val   = total_revenue / n_orders
avg_items_order = df.groupby("InvoiceNo")["Quantity"].sum().mean()

# Revenue by month
rev_month = (
    df.groupby("YearMonth")["Revenue"].sum()
    .reset_index().rename(columns={"YearMonth":"Month","Revenue":"Revenue"})
)

# Revenue by country
rev_country = (
    df.groupby("Country")["Revenue"].sum()
    .sort_values(ascending=False).head(10).reset_index()
)

# Top products
top_products = (
    df.groupby("Description")["Revenue"].sum()
    .sort_values(ascending=False).head(10).reset_index()
)
top_qty = (
    df.groupby("Description")["Quantity"].sum()
    .sort_values(ascending=False).head(10).reset_index()
)

# Hourly pattern
hourly = df.groupby("Hour")["Revenue"].sum().reset_index()
# Day of week
dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
rev_dow = (
    df.groupby("DayOfWeek")["Revenue"].sum()
    .reindex(dow_order).fillna(0).reset_index()
)

# Segment summary
seg_sum = (
    rfm.groupby("Segment")
    .agg(Customers=("CustomerID","count"),
         AvgRecency=("Recency","mean"),
         AvgFrequency=("Frequency","mean"),
         AvgMonetary=("Monetary","mean"),
         TotalRev=("Monetary","sum"))
    .reset_index()
    .sort_values("Customers",ascending=False)
)
seg_sum["pct"] = (seg_sum["Customers"]/seg_sum["Customers"].sum()*100).round(0).astype(int)

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def fmt_date(d):
    if hasattr(d,"strftime"):
        return d.strftime("%b %d, %Y").replace(" 0"," ")
    return str(d)

def safe_gradient(styler, subset, cmap="Blues"):
    """Apply background_gradient only when matplotlib is available."""
    try:
        import matplotlib  # noqa: F401
        return styler.background_gradient(subset=subset, cmap=cmap)
    except ImportError:
        return styler

def kpi_card(col, label, val, sub=""):
    col.markdown(
        f"<div class='kpi'><div class='kpi-lbl'>{label}</div>"
        f"<div class='kpi-val'>{val}</div>"
        + (f"<div class='kpi-sub'>{sub}</div>" if sub else "")
        + "</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  PAGE HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class='ph'>
  <div>
    <h1>🛒 Online Retail Analytics Dashboard</h1>
    <p>RFM Segmentation · E-commerce Trends · Customer Intelligence</p>
  </div>
  <div style='text-align:right'>
    <div style='font-size:13px;font-weight:600;color:#1a1d3a'>
        {fmt_date(d1)} – {fmt_date(d2)}
    </div>
    <div style='font-size:12px;color:#8b90b8'>
        {n_customers:,} Customers &nbsp;·&nbsp; £{total_revenue:,.0f} Revenue
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
#  ROW 0 – KPI CARDS
# ════════════════════════════════════════════════════════
k = st.columns(6, gap="medium")
kpi_card(k[0], "Total Revenue",   f"£{total_revenue/1000:.1f}k")
kpi_card(k[1], "Total Orders",    f"{n_orders:,}")
kpi_card(k[2], "Unique Customers",f"{n_customers:,}")
kpi_card(k[3], "Products Sold",   f"{n_products:,}")
kpi_card(k[4], "Avg Order Value", f"£{avg_order_val:.2f}")
kpi_card(k[5], "Avg Items/Order", f"{avg_items_order:.1f}")

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
#  ROW 1 – Monthly Revenue | Revenue by Country pie
# ════════════════════════════════════════════════════════
r1a, r1b = st.columns([1.6, 1], gap="medium")

with r1a:
    st.markdown("<div class='ct'>Monthly Revenue Trend</div>", unsafe_allow_html=True)
    st.markdown("<div class='cs'>Total revenue by month</div>", unsafe_allow_html=True)
    fig_rev = go.Figure()
    fig_rev.add_trace(go.Scatter(
        x=rev_month["Month"], y=rev_month["Revenue"],
        mode="lines+markers",
        line=dict(color="#2e41d4", width=2.5, shape="spline"),
        fill="tozeroy", fillcolor="rgba(46,65,212,0.08)",
        marker=dict(size=6, color="#2e41d4"),
        hovertemplate="<b>%{x}</b><br>£%{y:,.0f}<extra></extra>",
    ))
    fig_rev.update_layout(CL(
        height=250, showlegend=False,
        xaxis=dict(showgrid=False, color="#8b90b8", tickangle=-30, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="#eef0f8", color="#8b90b8",
                   tickprefix="£", tickfont=dict(size=10)),
    ))
    st.plotly_chart(fig_rev, use_container_width=True, config={"displayModeBar":False})

with r1b:
    st.markdown("<div class='ct'>Revenue by Country</div>", unsafe_allow_html=True)
    st.markdown("<div class='cs'>Top 10 countries (filtered)</div>", unsafe_allow_html=True)
    fig_cpie = px.pie(
        rev_country, values="Revenue", names="Country",
        color_discrete_sequence=BLUE, hole=0.5,
    )
    fig_cpie.update_traces(
        textinfo="percent", textposition="outside",
        hovertemplate="<b>%{label}</b><br>£%{value:,.0f}<extra></extra>",
    )
    fig_cpie.add_annotation(
        text=f"<b>£{rev_country['Revenue'].sum()/1000:.0f}k</b><br>Total",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=12, color="#1a1d3a", family="Inter"),
    )
    fig_cpie.update_layout(CL(
        height=250, showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10), orientation="v"),
    ))
    st.plotly_chart(fig_cpie, use_container_width=True, config={"displayModeBar":False})

# ════════════════════════════════════════════════════════
#  ROW 2 – Hourly Revenue | Day-of-week Revenue
# ════════════════════════════════════════════════════════
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
r2a, r2b = st.columns(2, gap="medium")

with r2a:
    st.markdown("<div class='ct'>Revenue by Hour of Day</div>", unsafe_allow_html=True)
    st.markdown("<div class='cs'>When do customers shop?</div>", unsafe_allow_html=True)
    fig_hr = px.bar(
        hourly, x="Hour", y="Revenue",
        color="Revenue",
        color_continuous_scale=[[0,"#eef0f8"],[1,"#1a1d8c"]],
        labels={"Revenue":"Revenue (£)"},
    )
    fig_hr.update_traces(hovertemplate="<b>%{x}:00h</b><br>£%{y:,.0f}<extra></extra>")
    fig_hr.update_layout(CL(
        height=240, coloraxis_showscale=False, showlegend=False,
        xaxis=dict(showgrid=False, color="#8b90b8", title="Hour"),
        yaxis=dict(showgrid=True, gridcolor="#eef0f8", color="#8b90b8",
                   tickprefix="£", title=""),
    ))
    st.plotly_chart(fig_hr, use_container_width=True, config={"displayModeBar":False})

with r2b:
    st.markdown("<div class='ct'>Revenue by Day of Week</div>", unsafe_allow_html=True)
    st.markdown("<div class='cs'>Which days drive the most revenue?</div>", unsafe_allow_html=True)
    fig_dow = px.bar(
        rev_dow, x="DayOfWeek", y="Revenue",
        color="Revenue",
        color_continuous_scale=[[0,"#eef0f8"],[1,"#2e41d4"]],
        labels={"DayOfWeek":"","Revenue":"Revenue (£)"},
    )
    fig_dow.update_traces(hovertemplate="<b>%{x}</b><br>£%{y:,.0f}<extra></extra>")
    fig_dow.update_layout(CL(
        height=240, coloraxis_showscale=False, showlegend=False,
        xaxis=dict(showgrid=False, color="#8b90b8", tickangle=-20, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="#eef0f8", color="#8b90b8",
                   tickprefix="£", title=""),
    ))
    st.plotly_chart(fig_dow, use_container_width=True, config={"displayModeBar":False})

# ════════════════════════════════════════════════════════
#  ROW 3 – Customer Segmentation header
# ════════════════════════════════════════════════════════
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
st.markdown(
    "<div class='ct' style='font-size:16px;margin-bottom:14px'>"
    "Customer Segmentation &nbsp;<span class='pill'>RFM Analysis</span></div>",
    unsafe_allow_html=True,
)

r3a, r3b = st.columns([1.1, 1], gap="medium")

with r3a:
    # Treemap
    fig_tree = px.treemap(
        seg_sum, path=["Segment"], values="Customers",
        color="Segment", color_discrete_map=SEG_COLORS,
        custom_data=["AvgMonetary","AvgFrequency","AvgRecency","TotalRev"],
    )
    fig_tree.update_traces(
        texttemplate="<b>%{label}</b><br>%{value} customers",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Customers: %{value}<br>"
            "Avg Revenue: £%{customdata[0]:,.0f}<br>"
            "Avg Frequency: %{customdata[1]:.1f}<br>"
            "Avg Recency: %{customdata[2]:.0f} days<extra></extra>"
        ),
        textfont=dict(size=12, color="white", family="Inter"),
        marker=dict(pad=dict(t=4,l=4,r=4,b=4)),
    )
    fig_tree.update_layout(CL(
        height=310, showlegend=False, margin=dict(t=0,b=0,l=0,r=0),
    ))
    st.plotly_chart(fig_tree, use_container_width=True, config={"displayModeBar":False})

with r3b:
    # R / F / M toggle
    rfm_choice = st.radio(
        "Metric", ["R — Recency", "F — Frequency", "M — Monetary"],
        horizontal=True, label_visibility="collapsed",
    )
    metric_map = {
        "R — Recency":   ("AvgRecency",   "days", True),
        "F — Frequency": ("AvgFrequency", "orders", False),
        "M — Monetary":  ("AvgMonetary",  "£",     False),
    }
    col_key, unit, lower_better = metric_map[rfm_choice]
    max_v = seg_sum[col_key].max() or 1

    rows_html = ""
    for _, row in seg_sum.iterrows():
        val  = row[col_key]
        fill = max(3, int((1 - val/max_v)*100) if lower_better else int(val/max_v*100))
        color = SEG_COLORS.get(row["Segment"],"#2e41d4")
        if unit == "£":
            fmt = f"£{val:,.0f}"
        else:
            fmt = f"{val:.0f} {unit}"
        rows_html += (
            f"<div class='prog-row'>"
            f"<span class='dl' style='width:150px'>"
            f"<span class='dot' style='background:{color}'></span>"
            f"<span style='font-size:12px;font-weight:500'>{row['Segment']}</span></span>"
            f"<span class='prog-cnt'>{row['Customers']}</span>"
            f"<div class='prog-wrap'><div class='prog-fill' style='width:{fill}%;background:{color}'></div></div>"
            f"<span class='prog-pct'>{row['pct']}%</span>"
            f"</div>"
        )
    st.markdown(f"<div style='padding-top:6px'>{rows_html}</div>", unsafe_allow_html=True)

    # mini table
    st.markdown("<br>", unsafe_allow_html=True)
    tbl = seg_sum[["Segment","Customers","AvgRecency","AvgFrequency","AvgMonetary"]].copy()
    tbl.columns = ["Segment","Customers","Recency (d)","Frequency","Monetary (£)"]
    tbl = tbl.set_index("Segment")
    st.dataframe(
        safe_gradient(
            tbl.style.format({"Recency (d)":"{:.0f}","Frequency":"{:.1f}","Monetary (£)":"£{:,.0f}"}),
            subset=["Monetary (£)"],
        ),
        use_container_width=True,
    )

# ════════════════════════════════════════════════════════
#  ROW 4 – RFM 3D scatter | Top products bar
# ════════════════════════════════════════════════════════
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
r4a, r4b = st.columns(2, gap="medium")

with r4a:
    st.markdown("<div class='ct'>RFM 3-D Scatter</div>", unsafe_allow_html=True)
    st.markdown("<div class='cs'>Coloured by segment</div>", unsafe_allow_html=True)
    fig_3d = px.scatter_3d(
        rfm, x="Recency", y="Frequency", z="Monetary",
        color="Segment", opacity=0.75,
        color_discrete_map=SEG_COLORS,
        hover_data={"CustomerID":True},
        labels={"Monetary":"Monetary (£)"},
    )
    fig_3d.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#1a1d3a", size=11),
        scene=dict(
            bgcolor="rgba(240,242,248,1)",
            xaxis=dict(gridcolor="#d0d9fd", color="#8b90b8"),
            yaxis=dict(gridcolor="#d0d9fd", color="#8b90b8"),
            zaxis=dict(gridcolor="#d0d9fd", color="#8b90b8"),
        ),
        height=360, margin=dict(t=0,b=0,l=0,r=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    st.plotly_chart(fig_3d, use_container_width=True, config={"displayModeBar":False})

with r4b:
    top_tab = st.radio(
        "Top products by", ["Revenue", "Quantity"],
        horizontal=True, label_visibility="visible",
    )
    st.markdown("<div class='cs'>Top 10 products</div>", unsafe_allow_html=True)
    src = top_products if top_tab == "Revenue" else top_qty
    ycol = "Revenue" if top_tab == "Revenue" else "Quantity"
    fig_tp = px.bar(
        src.sort_values(ycol), x=ycol, y="Description",
        orientation="h",
        color=ycol,
        color_continuous_scale=[[0,"#d0d9fd"],[1,"#1a1d8c"]],
        text=ycol,
        labels={"Description":""},
    )
    fmt_txt = "£%{text:,.0f}" if top_tab == "Revenue" else "%{text:,}"
    fig_tp.update_traces(
        texttemplate=fmt_txt,
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>" + ("£" if top_tab=="Revenue" else "") + "%{x:,}<extra></extra>",
    )
    fig_tp.update_layout(CL(
        height=360, coloraxis_showscale=False, showlegend=False,
        xaxis=dict(showgrid=True, gridcolor="#eef0f8", color="#8b90b8",
                   tickprefix="£" if top_tab=="Revenue" else ""),
        yaxis=dict(showgrid=False, color="#1a1d3a", tickfont=dict(size=10)),
    ))
    st.plotly_chart(fig_tp, use_container_width=True, config={"displayModeBar":False})

# ════════════════════════════════════════════════════════
#  ROW 5 – Revenue heatmap (Country x Month) | RFM dist
# ════════════════════════════════════════════════════════
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
r5a, r5b = st.columns([1.3, 1], gap="medium")

with r5a:
    st.markdown("<div class='ct'>Revenue Heatmap — Country × Month</div>", unsafe_allow_html=True)
    st.markdown("<div class='cs'>Top 8 countries × last months</div>", unsafe_allow_html=True)
    top8 = rev_country["Country"].head(8).tolist()
    hm_data = (
        df[df["Country"].isin(top8)]
        .groupby(["Country","YearMonth"])["Revenue"].sum()
        .reset_index()
        .pivot(index="Country", columns="YearMonth", values="Revenue")
        .fillna(0)
    )
    # keep last 12 months
    hm_data = hm_data[sorted(hm_data.columns)[-12:]]
    fig_hm = px.imshow(
        hm_data.round(0), text_auto=".0f",
        color_continuous_scale=[[0,"#eef0f8"],[0.5,"#5068e8"],[1,"#1a1d8c"]],
        labels=dict(color="Revenue (£)"), aspect="auto",
    )
    fig_hm.update_layout(CL(
        height=290, coloraxis_showscale=False, showlegend=False,
        xaxis=dict(color="#8b90b8", tickangle=-30, tickfont=dict(size=9)),
        yaxis=dict(color="#8b90b8"),
    ))
    st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar":False})

with r5b:
    st.markdown("<div class='ct'>RFM Distribution</div>", unsafe_allow_html=True)
    rfm_hist_metric = st.selectbox(
        "Metric", ["Recency","Frequency","Monetary"],
        label_visibility="collapsed",
    )
    pal = {"Recency":"#5068e8","Frequency":"#2e41d4","Monetary":"#1a1d8c"}
    fig_rfm_h = px.histogram(
        rfm, x=rfm_hist_metric, nbins=30,
        color_discrete_sequence=[pal[rfm_hist_metric]],
        labels={rfm_hist_metric: rfm_hist_metric + (" (days)" if rfm_hist_metric=="Recency" else
                                                     " (orders)" if rfm_hist_metric=="Frequency" else " (£)")},
    )
    fig_rfm_h.update_layout(CL(
        height=270, showlegend=False,
        xaxis=dict(showgrid=False, color="#8b90b8"),
        yaxis=dict(showgrid=True, gridcolor="#eef0f8", color="#8b90b8"),
    ))
    st.plotly_chart(fig_rfm_h, use_container_width=True, config={"displayModeBar":False})

# ════════════════════════════════════════════════════════
#  ROW 6 – Customer Lookup
# ════════════════════════════════════════════════════════
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
st.markdown("<div class='ct' style='font-size:16px;margin-bottom:12px'>🔎 Customer Lookup</div>",
            unsafe_allow_html=True)

cid_list = sorted(rfm["CustomerID"].unique())
lu_c, _ = st.columns([0.28, 0.72])
with lu_c:
    search_id = st.selectbox("Customer ID", cid_list, label_visibility="collapsed")

cust_tx  = df[df["CustomerID"] == search_id]
cust_rfm = rfm[rfm["CustomerID"] == search_id]

if cust_tx.empty:
    st.info("Customer not found in filtered data.")
else:
    rfm_row = cust_rfm.iloc[0]
    seg = rfm_row["Segment"]
    seg_color = SEG_COLORS.get(seg, "#2e41d4")

    kc = st.columns(6, gap="medium")
    for col, lbl, val in zip(kc, [
        "Segment","Recency","Frequency","Monetary","Country","Transactions",
    ],[
        seg,
        f"{rfm_row['Recency']} days",
        f"{rfm_row['Frequency']} orders",
        f"£{rfm_row['Monetary']:,.0f}",
        cust_tx["Country"].iloc[0],
        f"{len(cust_tx):,} lines",
    ]):
        col.markdown(
            f"<div style='background:#fff;border-radius:12px;padding:14px 16px;"
            f"box-shadow:0 2px 10px rgba(26,29,58,.07);text-align:center'>"
            f"<div style='font-size:10px;color:#8b90b8;letter-spacing:.8px;text-transform:uppercase'>{lbl}</div>"
            f"<div style='font-size:18px;font-weight:700;color:#1a1d3a;margin-top:4px'>{val}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<div style='margin-top:10px;padding:10px 18px;background:#e8f0fe;"
        f"border-radius:10px;font-size:13px;color:{seg_color};font-weight:600'>"
        f"🏷️ Segment: {seg} &nbsp;·&nbsp; Cluster {rfm_row['Cluster']}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    lc1, lc2 = st.columns([1.6, 1], gap="medium")
    with lc1:
        st.markdown("<div class='ct'>Purchase History</div>", unsafe_allow_html=True)
        cust_daily = (
            cust_tx.groupby("Date")["Revenue"].sum().reset_index()
            .rename(columns={"Date":"Date","Revenue":"Revenue"})
        )
        fig_ch = go.Figure(go.Scatter(
            x=cust_daily["Date"], y=cust_daily["Revenue"],
            mode="lines+markers",
            line=dict(color="#2e41d4", width=2),
            fill="tozeroy", fillcolor="rgba(46,65,212,0.07)",
            marker=dict(size=5),
            hovertemplate="<b>%{x}</b><br>£%{y:,.0f}<extra></extra>",
        ))
        fig_ch.update_layout(CL(
            height=230, showlegend=False,
            xaxis=dict(showgrid=False, color="#8b90b8"),
            yaxis=dict(showgrid=True, gridcolor="#eef0f8", color="#8b90b8", tickprefix="£"),
        ))
        st.plotly_chart(fig_ch, use_container_width=True, config={"displayModeBar":False})

    with lc2:
        st.markdown("<div class='ct'>Top Products Bought</div>", unsafe_allow_html=True)
        cprod = (
            cust_tx.groupby("Description")["Revenue"].sum()
            .sort_values(ascending=False).head(6).reset_index()
        )
        fig_cprod = px.pie(
            cprod, values="Revenue", names="Description",
            color_discrete_sequence=BLUE, hole=0.45,
        )
        fig_cprod.update_traces(textinfo="percent", textposition="outside",
                                 hovertemplate="<b>%{label}</b><br>£%{value:,.0f}<extra></extra>")
        fig_cprod.update_layout(CL(
            height=230, showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        ))
        st.plotly_chart(fig_cprod, use_container_width=True, config={"displayModeBar":False})

    st.markdown("<div class='ct' style='margin-top:2px'>Transaction Records</div>",
                unsafe_allow_html=True)
    show_cols = ["InvoiceDate","InvoiceNo","Description","Quantity","UnitPrice","Revenue","Country"]
    st.dataframe(
        safe_gradient(
            cust_tx[show_cols].sort_values("InvoiceDate", ascending=False)
            .reset_index(drop=True)
            .style.format({"UnitPrice":"£{:.2f}","Revenue":"£{:.2f}"}),
            subset=["Revenue"],
        ),
        use_container_width=True,
    )

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;font-size:12px;color:#b0b4d0;padding:8px 0'>"
    "🛒 Online Retail Analytics Dashboard &nbsp;·&nbsp; "
    "Dataset: UCI Online Retail &nbsp;·&nbsp; Built with Streamlit & Plotly"
    "</div>",
    unsafe_allow_html=True,
)
