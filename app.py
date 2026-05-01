import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm

# ================= CONFIG =================
st.set_page_config(page_title="YouTube Content Intelligence", layout="wide")
st.title("🎬 YouTube Content Intelligence Dashboard")

# ================= LOAD DATA =================
df = pd.read_csv("data/vn_music_all.csv")

# ================= FEATURE ENGINEERING =================
df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')

# Engagement (safe)
df['engagement_rate'] = (
    (df['like_count'] + df['comment_count']) / df['view_count']
)
df['engagement_rate'] = df['engagement_rate'].replace([np.inf, -np.inf], np.nan)

# Duration (convert)
df['duration_sec'] = pd.to_numeric(df['duration'], errors='coerce')

# Language group
df['language_group'] = np.where(
    df['is_vietnamese_language_flag'] == 1,
    "Vietnamese",
    "Non-Vietnamese"
)

# ================= SIDEBAR =================
st.sidebar.header("🔎 Filters")

category = st.sidebar.multiselect(
    "Category",
    options=df['category_id'].dropna().unique(),
    default=df['category_id'].dropna().unique()
)

lang = st.sidebar.multiselect(
    "Language",
    options=df['language_group'].unique(),
    default=df['language_group'].unique()
)

accepted = st.sidebar.multiselect(
    "Accepted",
    options=df['accepted'].dropna().unique(),
    default=df['accepted'].dropna().unique()
)

filtered = df[
    (df['category_id'].isin(category)) &
    (df['language_group'].isin(lang)) &
    (df['accepted'].isin(accepted))
].copy()

# ================= SAFE CLEAN FUNCTION =================
def clean_for_model(data, cols):
    data = data.copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=cols)
    return data

# ================= TABS =================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Overview",
    "🌏 Language",
    "📊 Content",
    "🧪 Confounder",
    "🤖 Model",
    "💬 AI Chat"
])

# ================= TAB 1: OVERVIEW =================
with tab1:
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Videos", len(filtered))
    col2.metric("Accepted Rate", f"{filtered['accepted'].mean():.2%}" if len(filtered)>0 else "N/A")
    col3.metric("Vietnamese Ratio", f"{(filtered['language_group']=='Vietnamese').mean():.2%}" if len(filtered)>0 else "N/A")
    col4.metric("Avg Views", f"{filtered['view_count'].mean():,.0f}" if len(filtered)>0 else "N/A")

    # Accepted vs Rejected
    fig = px.pie(filtered, names='accepted', title="Accepted vs Rejected")
    st.plotly_chart(fig, use_container_width=True)

    # Filter reason FIXED
    reason_df = (
        filtered['filter_reason']
        .value_counts()
        .rename_axis('filter_reason')
        .reset_index(name='count')
    )

    fig2 = px.bar(reason_df, x='filter_reason', y='count', title="Filter Reasons")
    st.plotly_chart(fig2, use_container_width=True)


# ================= TAB 2: LANGUAGE =================
with tab2:
    st.subheader("Language Impact")

    fig = px.box(filtered, x='language_group', y='view_count')
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.box(filtered, x='has_vietnamese_diacritics', y='view_count')
    st.plotly_chart(fig2, use_container_width=True)


# ================= TAB 3: CONTENT =================
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            filtered,
            x='tags_count',
            y='view_count',
            color='category_id'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(filtered, x='category_id', y='view_count')
        st.plotly_chart(fig, use_container_width=True)

    fig3 = px.histogram(filtered, x='duration_sec', nbins=50)
    st.plotly_chart(fig3, use_container_width=True)


# ================= TAB 4: CONFOUNDER =================
with tab4:
    st.subheader("Confounder Analysis")

    cols = ['duration_sec', 'tags_count', 'view_count']
    model_df = clean_for_model(filtered, cols)

    if len(model_df) > 10:
        model_df['log_view'] = np.log1p(model_df['view_count'])

        X = model_df[['duration_sec', 'tags_count']]
        X = sm.add_constant(X)
        y = model_df['log_view']

        model = sm.OLS(y, X).fit()
        st.text(model.summary())
    else:
        st.warning("Not enough clean data for modeling")


# ================= TAB 5: MODEL =================
with tab5:
    st.subheader("Feature Importance")

    cols = ['duration_sec', 'tags_count', 'view_count', 'is_vietnamese_language_flag']
    model_df = clean_for_model(filtered, cols)

    if len(model_df) > 10:
        model_df['log_view'] = np.log1p(model_df['view_count'])

        X = model_df[['duration_sec', 'tags_count', 'is_vietnamese_language_flag']]
        X = sm.add_constant(X)
        y = model_df['log_view']

        model = sm.OLS(y, X).fit()

        coef = pd.DataFrame({
            "feature": X.columns,
            "coef": model.params
        })

        fig = px.bar(coef, x='feature', y='coef', title="Feature Importance")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Not enough clean data for modeling")


# ================= TAB 6: AI CHAT =================
with tab6:
    st.subheader("AI Insight Chat")

    q = st.text_input("Ask something...")

    if q:
        q = q.lower()

        if "vietnamese" in q:
            avg_vn = filtered[filtered['language_group']=="Vietnamese"]['view_count'].mean()
            avg_non = filtered[filtered['language_group']=="Non-Vietnamese"]['view_count'].mean()
            st.write(f"VN avg views: {avg_vn:,.0f} | Non-VN: {avg_non:,.0f}")

        elif "tags" in q:
            corr = filtered[['tags_count','view_count']].corr().iloc[0,1]
            st.write(f"Correlation tags vs views: {corr:.2f}")

        elif "best" in q:
            top = filtered.sort_values("view_count", ascending=False).iloc[0]
            st.write(f"Top video: {top['title']} ({top['view_count']:,} views)")

        else:
            st.write("Try asking about Vietnamese, tags, or best video.")