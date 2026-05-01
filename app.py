import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
import re

# ================= CONFIG =================
st.set_page_config(page_title="YouTube VN Music Dashboard", layout="wide")
st.title("🎬 YouTube VN Music Intelligence Dashboard")

# ================= LOAD =================
df = pd.read_csv("data/vn_music_simplification.csv")

# ================= FEATURE ENGINEERING =================
df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce', utc=True)
df['hour'] = df['published_at'].dt.hour
df['day_of_week'] = df['published_at'].dt.day_name()

# parse duration ISO8601
def parse_duration(d):
    if pd.isna(d): return np.nan
    m = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', str(d))
    if not m: return np.nan
    h = int(m.group(1)) if m.group(1) else 0
    m_ = int(m.group(2)) if m.group(2) else 0
    s = int(m.group(3)) if m.group(3) else 0
    return h*3600 + m_*60 + s

df['duration_sec'] = df['duration'].apply(parse_duration)

# title length
df['title_length'] = df['title'].astype(str).apply(len)

# log view
df['log_view'] = np.log1p(df['view_count'])

# ================= FILTER =================
st.sidebar.header("🔎 Filter")

date_range = st.sidebar.date_input(
    "Date range",
    [df['published_at'].min().date(), df['published_at'].max().date()]
)

view_range = st.sidebar.slider(
    "View range",
    int(df['view_count'].min()),
    int(df['view_count'].max()),
    (int(df['view_count'].min()), int(df['view_count'].max()))
)

duration_range = st.sidebar.slider(
    "Duration (sec)",
    int(df['duration_sec'].fillna(0).min()),
    int(df['duration_sec'].fillna(0).max()),
    (0, int(df['duration_sec'].fillna(0).max()))
)
start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])

start_date = pd.Timestamp(start_date, tz="UTC")
end_date = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

filtered = df[
    (df['published_at'] >= start_date) &
    (df['published_at'] <= end_date) &
    (df['view_count'].between(*view_range)) &
    (df['duration_sec'].fillna(0).between(*duration_range))
].copy()

if filtered.empty:
    st.warning("No videos match the current filters.")
    st.stop()

# ================= OVERVIEW =================
st.header("🎯 Overview")

col1,col2,col3,col4,col5,col6 = st.columns(6)

col1.metric("Total Videos", len(filtered))
col2.metric("Time Range", f"{filtered['published_at'].min().date()} → {filtered['published_at'].max().date()}")
col3.metric("VN Content %", "100%")  # dataset fixed
col4.metric("Music %", "100%")       # category = 10
col5.metric("Mean Views", f"{filtered['view_count'].mean():,.0f}")
col6.metric("Median Views", f"{filtered['view_count'].median():,.0f}")

# distribution
fig = px.histogram(filtered, x='view_count', nbins=50, title="View Distribution")
st.plotly_chart(fig, use_container_width=True)

# ================= PERFORMANCE SEGMENT =================
st.header("🔥 Performance Segmentation")

q1 = filtered['view_count'].quantile(0.3)
q2 = filtered['view_count'].quantile(0.7)
q3 = filtered['view_count'].quantile(0.9)

def tier(v):
    if v >= q3: return "Viral"
    elif v >= q2: return "High"
    elif v >= q1: return "Medium"
    else: return "Low"

filtered['performance'] = filtered['view_count'].apply(tier)

# % viral
viral_pct = (filtered['performance']=="Viral").mean()*100
st.metric("🔥 % Viral Videos", f"{viral_pct:.2f}%")

col1,col2 = st.columns(2)

with col1:
    perf_counts = filtered['performance'].value_counts().rename_axis('performance').reset_index(name='count')
    fig = px.bar(perf_counts,
                 x='performance', y='count',
                 title="Video Count by Performance Tier")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.box(filtered, x='performance', y='duration_sec',
                 title="Duration by Performance")
    st.plotly_chart(fig, use_container_width=True)

# ================= TIMING =================
st.header("⏰ Best Time to Post")

heat = filtered.pivot_table(index='day_of_week', columns='hour',
                            values='view_count', aggfunc='mean')

fig = px.imshow(heat, title="Heatmap: Hour vs Day")
st.plotly_chart(fig, use_container_width=True)

# ================= STEPWISE MODEL =================
st.header("🤖 Stepwise Regression (Predict View)")

features = [
    'duration_sec',
    'tags_count',
    'channel_subscriber_count',
    'title_length',
    'channel_view_count',
    'channel_video_count'
]

df_model = filtered.copy()
df_model = df_model.replace([np.inf, -np.inf], np.nan)

# fill NA
for col in features:
    df_model[col] = df_model[col].fillna(df_model[col].median())

df_model = df_model.dropna(subset=['log_view'])

def forward_stepwise(data, target, candidates):
    selected = []
    log = []

    while candidates:
        scores = []
        for c in candidates:
            formula_cols = selected + [c]
            X = sm.add_constant(data[formula_cols])
            y = data[target]
            model = sm.OLS(y, X).fit()
            pval = model.pvalues[c]
            scores.append((c, pval))

        best = min(scores, key=lambda x: x[1])

        if best[1] < 0.05:
            selected.append(best[0])
            candidates.remove(best[0])
            log.append(f"Added {best[0]} (p={best[1]:.4f})")
        else:
            break

    X = sm.add_constant(data[selected])
    model = sm.OLS(data[target], X).fit()

    return selected, model, log

selected, model, log = forward_stepwise(df_model, 'log_view', features.copy())

col1,col2 = st.columns(2)

with col1:
    st.subheader("Stepwise Process")
    for l in log:
        st.code(l)

with col2:
    st.subheader("Model Summary")
    st.text(model.summary().tables[1])

# ================= CONFOUNDER =================
st.header("🧪 Confounder Analysis")

df_model['channel_size'] = pd.qcut(df_model['channel_subscriber_count'], 3, labels=['Small','Medium','Large'])

fig = px.scatter(
    df_model,
    x='hour',
    y='view_count',
    color='channel_size',
    trendline="ols",
    title="Hour vs View (Controlled by Channel Size)"
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
### 💡 Insight:
- Channel size có thể đảo chiều hoàn toàn kết luận về giờ đăng  
- Hour là **fake driver** nếu không control subscriber  
""")

# ================= CHATBOT =================
st.header("💬 AI Chatbot")

q = st.text_input("Ask something...")

if q:
    q = q.lower()

    if "viral" in q:
        st.write(f"🔥 Viral rate: {viral_pct:.2f}%")

    elif "time" in q:
        best_hour = filtered.groupby('hour')['view_count'].mean().idxmax()
        st.write(f"Best hour to post: {best_hour}")

    elif "best" in q:
        top = filtered.sort_values("view_count", ascending=False).iloc[0]
        st.write(f"Top video: {top['title']} ({top['view_count']:,} views)")

    else:
        st.write("Try: viral, time, best")