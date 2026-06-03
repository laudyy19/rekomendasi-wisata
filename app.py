import streamlit as st
import pandas as pd

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.feature_engineering import create_content_feature
from src.model import build_model, get_recommendations


# =========================
# CONFIG
# =========================

PALETTE = {
    "bg": "#FAF7F2",
    "card": "#F3EDE3",
    "primary": "#7D9B76",
    "secondary": "#C4A882",
    "accent": "#C17F5E",
    "text": "#4A3F35",
    "muted": "#9E8E80",
}

st.set_page_config(
    page_title="Rekomendasi Wisata Indonesia",
    page_icon="🌿",
    layout="wide"
)

# =========================
# CSS
# =========================

st.markdown(
    f"""
<style>

html, body, [class*="css"], .stApp, .main, .block-container,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"] {{
    background-color:{PALETTE['bg']} !important;
    color:{PALETTE['text']} !important;
}}

section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {{
    background-color:{PALETTE['card']} !important;
}}

[data-testid="stSelectbox"] > div > div {{
    background-color:{PALETTE['card']} !important;
    color:{PALETTE['text']} !important;
}}

[data-testid="metric-container"] {{
    background-color:{PALETTE['card']} !important;
    border-radius:12px;
    padding:10px;
}}

.hero {{
    background: linear-gradient(
        135deg,
        {PALETTE['primary']},
        {PALETTE['secondary']}
    );

    border-radius:16px;
    padding:2rem;
    margin-bottom:1.5rem;
}}

.hero h1 {{
    color:white !important;
}}

.hero p {{
    color:white !important;
}}

.rec-card {{
    background:{PALETTE['card']};
    border-radius:12px;
    padding:1rem;
    margin-bottom:1rem;
    border-left:5px solid {PALETTE['primary']};
}}

.badge {{
    display:inline-block;
    padding:4px 10px;
    border-radius:999px;
    margin-right:5px;
    font-size:12px;
}}

.cat {{
    background:#dce8d8;
}}

.city {{
    background:#f0e2d0;
}}

.info {{
    background:#e8e0d5;
}}

.bar-wrap {{
    width:100%;
    height:7px;
    background:#ddd;
    border-radius:999px;
    margin-top:10px;
}}

.bar-fill {{
    height:7px;
    border-radius:999px;
}}

</style>
""",
    unsafe_allow_html=True,
)

# =========================
# LOAD DATA
# =========================

@st.cache_data
def get_data():

    df = load_data()
    df = preprocess_data(df)
    df = create_content_feature(df)

    return df


@st.cache_resource
def get_model(df):

    cosine_sim, indices = build_model(df)

    return cosine_sim, indices


try:

    df = get_data()

    cosine_sim, indices = get_model(df)

except Exception as e:

    st.error(f"Error : {e}")
    st.stop()

# =========================
# HERO
# =========================

st.markdown(
    """
<div class="hero">
<h1>🌿 Sistem Rekomendasi Wisata Indonesia</h1>
<p>
Content-Based Filtering menggunakan
TF-IDF dan Cosine Similarity
</p>
</div>
""",
    unsafe_allow_html=True,
)

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.header("🗺️ Filter")

    kota_list = ["Semua"] + sorted(df["City"].unique())

    filter_kota = st.selectbox(
        "Pilih Kota",
        kota_list
    )

    kategori_list = ["Semua"] + sorted(
        df["Category"].unique()
    )

    filter_kategori = st.selectbox(
        "Pilih Kategori",
        kategori_list
    )

    top_n = st.slider(
        "Jumlah Rekomendasi",
        3,
        10,
        5
    )

# =========================
# FILTER DATA
# =========================

df_filter = df.copy()

if filter_kota != "Semua":
    df_filter = df_filter[
        df_filter["City"] == filter_kota
    ]

if filter_kategori != "Semua":
    df_filter = df_filter[
        df_filter["Category"] == filter_kategori
    ]

# =========================
# SELECT PLACE
# =========================

c1, c2 = st.columns([3, 1])

with c1:

    pilihan = st.selectbox(
        "🔍 Pilih Destinasi Wisata",
        df_filter["Place_Name"].tolist()
    )

with c2:

    st.metric(
        "Total Tempat",
        len(df_filter)
    )

# =========================
# INFO TEMPAT
# =========================

selected = df[
    df["Place_Name"] == pilihan
].iloc[0]

a, b, c = st.columns(3)

a.metric("📍 Kota", selected["City"])
b.metric("🏷️ Kategori", selected["Category"])

if "Rating" in df.columns:
    c.metric(
        "⭐ Rating",
        round(float(selected["Rating"]), 1)
    )

st.markdown("---")

st.subheader(
    f"🧭 Top {top_n} Rekomendasi untuk {pilihan}"
)

# =========================
# REKOMENDASI
# =========================

hasil = get_recommendations(
    pilihan,
    cosine_sim,
    indices,
    df,
    top_n
)

for rank, row in enumerate(
    hasil.itertuples(),
    start=1
):

    score = row.Similarity

    persen = int(score * 100)

    if score >= 0.7:
        color = PALETTE["primary"]
    elif score >= 0.4:
        color = PALETTE["secondary"]
    else:
        color = PALETTE["accent"]

    price = (
        f"Rp {int(row.Price):,}"
        if row.Price > 0
        else "Gratis"
    )

    st.markdown(
        f"""
<div class="rec-card">

<b>#{rank} {row.Place_Name}</b>

<br><br>

<span class="badge cat">
{row.Category}
</span>

<span class="badge city">
📍 {row.City}
</span>

<span class="badge info">
⭐ {row.Rating}
</span>

<span class="badge info">
{price}
</span>

<br><br>

<b style="color:{color}">
Similarity : {score:.4f}
</b>

<div class="bar-wrap">
<div
class="bar-fill"
style="
width:{persen}%;
background:{color};
">
</div>
</div>

</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("---")

st.caption(
    "Sistem Rekomendasi Wisata Indonesia • Content Based Filtering • TF-IDF • Cosine Similarity"
)