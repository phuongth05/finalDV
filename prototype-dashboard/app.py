import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm

# ================= CONFIG =================
st.set_page_config(
    page_title="YouTube Music Intelligence",
    layout="wide",
    page_icon="🎧"
)

# ================= STYLE =================
st.markdown("""
<style>
.big-title {
    font-size:28px !important;
    font-weight:700;
}
.metric-card {
    background-color:#111;
    padding:15px;
    border-radius:12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🎧 YouTube VN Music Analytics</p>', unsafe_allow_html=True)

# ================= LOAD =================
df = pd.read_csv("data/youtube_vn_music_cleaned.csv")

# ================= FEATURE ENGINEERING =================
df['video_publish_date'] = pd.to_datetime(df['video_publish_date'], errors='coerce')
df['hour'] = df['video_publish_date'].dt.hour
df['day'] = df['video_publish_date'].dt.day_name()

# basic features
df['title_length'] = df['video_title'].astype(str).apply(len)

# ================= SIDEBAR =================
st.sidebar.header("🔎 Filter")

date_range = st.sidebar.date_input(
    "Date range",
    [df['video_publish_date'].min(), df['video_publish_date'].max()]
)

view_range = st.sidebar.slider(
    "View range",
    int(df['video_view_count'].min()),
    int(df['video_view_count'].max()),
    (int(df['video_view_count'].min()), int(df['video_view_count'].max()))
)

# convert to date
df['publish_date'] = df['video_publish_date'].dt.date

filtered = df[
    (df['publish_date'] >= date_range[0]) &
    (df['publish_date'] <= date_range[1]) &
    (df['video_view_count'].between(*view_range))
].copy()

# ================= TABS =================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🔥 Performance",
    "⏰ Timing",
    "🤖 Modeling",
    "🧪 Confounder"
])

# ================= TAB 1 =================
with tab1:
    st.subheader("Overview")

    col1,col2,col3,col4,col5,col6 = st.columns(6)

    col1.metric("Videos", len(filtered))
    col2.metric("Mean Views", f"{filtered['video_view_count'].mean():,.0f}")
    col3.metric("Median Views", f"{filtered['video_view_count'].median():,.0f}")
    col4.metric("Max Views", f"{filtered['video_view_count'].max():,.0f}")
    col5.metric("Avg Tags", f"{filtered['video_tags_count'].mean():.1f}")
    col6.metric("Avg Subscribers", f"{filtered['channel_subscriber_count'].mean():,.0f}")

    st.markdown("### View Distribution")

    fig = px.histogram(filtered, x='video_view_count', nbins=60)
    st.plotly_chart(fig, use_container_width=True)

# ================= TAB 2 =================
with tab2:
    st.subheader("Performance Segmentation")

    q90 = filtered['video_view_count'].quantile(0.9)
    q70 = filtered['video_view_count'].quantile(0.7)

    def label(v):
        if v >= q90: return "Viral"
        elif v >= q70: return "High"
        else: return "Normal"

    filtered['tier'] = filtered['video_view_count'].apply(label)

    col1,col2 = st.columns(2)

    with col1:
        fig = px.pie(filtered, names='tier', title="Video Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(filtered, x='tier', y='channel_subscriber_count', title="Subscribers vs Performance")
        st.plotly_chart(fig, use_container_width=True)

# ================= TAB 3 =================
with tab3:
    st.subheader("Best Time to Post")

    heat = filtered.pivot_table(
        index='day',
        columns='hour',
        values='video_view_count',
        aggfunc='mean'
    )

    fig = px.imshow(heat, title="Heatmap: Day vs Hour")
    st.plotly_chart(fig, use_container_width=True)

# ================= TAB 4 =================
with tab4:
    st.subheader("Stepwise Regression")

    features = [
        'video_tags_count',
        'channel_subscriber_count',
        'title_length',
        'channel_view_count',
        'channel_video_count'
    ]

    df_model = filtered.copy()
    df_model = df_model.replace([np.inf, -np.inf], np.nan)

    for col in features:
        df_model[col] = df_model[col].fillna(df_model[col].median())

    df_model = df_model.dropna(subset=['video_view_count'])

    def stepwise(data, target, feats):
        selected = []
        log = []

        data = data.copy()
        # coerce candidate columns to numeric and replace infinities
        for c in list(feats) + [target]:
            if c in data.columns:
                data[c] = pd.to_numeric(data[c], errors='coerce')

        data = data.replace([np.inf, -np.inf], np.nan)

        while feats:
            scores = []
            for f in feats:
                cols = selected + [f]
                # prepare exog and endog and drop rows with missing values
                X = sm.add_constant(data[cols], has_constant='add')
                y = data[target]
                mask = X.notna().all(axis=1) & y.notna()
                if mask.sum() == 0:
                    scores.append((f, np.inf))
                    continue
                try:
                    model_try = sm.OLS(y[mask], X[mask]).fit()
                    pval = model_try.pvalues.get(f, np.inf)
                except Exception:
                    pval = np.inf
                scores.append((f, pval))

            best = min(scores, key=lambda x: x[1])

            if best[1] < 0.05:
                selected.append(best[0])
                feats.remove(best[0])
                log.append(f"Added {best[0]} (p={best[1]:.4f})")
            else:
                break

        # final model: use selected features (or constant-only if none)
        if selected:
            X_final = sm.add_constant(data[selected], has_constant='add')
        else:
            X_final = pd.DataFrame({'const': np.ones(len(data))}, index=data.index)

        y_final = data[target]
        mask_final = X_final.notna().all(axis=1) & y_final.notna()
        if mask_final.sum() == 0:
            final_model = None
        else:
            final_model = sm.OLS(y_final[mask_final], X_final[mask_final]).fit()

        return selected, final_model, log

    selected, model, log = stepwise(df_model, 'video_view_count', features.copy())

    col1,col2 = st.columns(2)

    with col1:
        st.write("### Selected Features")
        for l in log:
            st.code(l)

    with col2:
        st.write("### Model Summary")
        st.text(model.summary().tables[1])

# ================= TAB 5 =================
with tab5:
    st.subheader("Confounder Analysis")

    df_model['channel_size'] = pd.qcut(
        df_model['channel_subscriber_count'],
        3,
        labels=['Small','Medium','Large']
    )

    fig = px.scatter(
        df_model,
        x='hour',
        y='video_view_count',
        color='channel_size',
        trendline="ols"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### Insight:
    - Channel size làm lệch hoàn toàn kết luận về giờ đăng  
    - Timing không phải yếu tố causal thực sự  
    """)

# ================= CHATBOT =================
st.markdown("---")
st.subheader("💬 Chatbot")

q = st.text_input("Ask insight...")

if q:
    if "viral" in q:
        pct = (filtered['video_view_count'] >= q90).mean()*100
        st.write(f"🔥 Viral rate: {pct:.2f}%")

    elif "time" in q:
        best = filtered.groupby('hour')['video_view_count'].mean().idxmax()
        st.write(f"Best posting hour: {best}")

    else:
        st.write("Try: viral, time")