import streamlit as st
import plotly.express as px


def render_tab(filtered_df):
    st.subheader("Tổng quan")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric("Số video", len(filtered_df))
    col2.metric("Lượt xem trung bình", f"{filtered_df['video_view_count'].mean():,.0f}")
    col3.metric("Lượt xem trung vị", f"{filtered_df['video_view_count'].median():,.0f}")
    col4.metric("Lượt xem cao nhất", f"{filtered_df['video_view_count'].max():,.0f}")
    col5.metric("Số thẻ trung bình", f"{filtered_df['video_tags_count'].mean():.1f}")
    col6.metric("Số người đăng ký trung bình", f"{filtered_df['channel_subscriber_count'].mean():,.0f}")

    st.markdown("### Phân bố lượt xem")

    fig = px.histogram(filtered_df, x='video_view_count', nbins=60)
    st.plotly_chart(fig, use_container_width=True)
