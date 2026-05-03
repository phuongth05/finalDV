import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


def render_tab(filtered_df, base_filtered_df, active_cross_filters, apply_cross_filters, sync_chart_selection):
    st.header("Câu 3: Khám phá 'Khung giờ vàng' đăng nhạc")

    df_q3 = filtered_df.dropna(subset=['video_view_count', 'hour', 'day']).copy()

    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_q3['day'] = pd.Categorical(df_q3['day'], categories=days_order, ordered=True)

    # --- BIỂU ĐỒ 3.1: HEATMAP THỜI GIAN ĐĂNG BÀI ---
    st.subheader("1. Bản đồ nhiệt (Heatmap): Đăng lúc nào dễ có view cao nhất?")

    heat_data = df_q3.groupby(['day', 'hour'])['video_view_count'].mean().reset_index()
    heat_pivot = heat_data.pivot(index='day', columns='hour', values='video_view_count')
    heat_pivot = heat_pivot.reindex(index=days_order)
    heat_pivot = heat_pivot.reindex(columns=sorted(heat_pivot.columns))

    # Drop NaN values to avoid issues with imshow
    heat_pivot = heat_pivot.fillna(0)

    fig7 = px.imshow(
        heat_pivot,
        labels=dict(x="Giờ trong ngày (0-23h)", y="Ngày trong tuần", color="Trung bình View"),
        color_continuous_scale='Viridis',
        aspect="auto",
        title="Heatmap: Tương quan giữa Giờ đăng, Ngày đăng và Lượt xem"
    )
    st.plotly_chart(fig7, use_container_width=True)
    st.caption("💡 **Insight:** Những ô vuông đỏ/cam rực rỡ nhất chính là 'Khung giờ vàng'. Đây là thời điểm phát hành nhạc đạt hiệu suất lượt xem cao nhất trong tuần.")

    # --- BIỂU ĐỒ 3.2: LINE/RADAR CHART - CUNG VS CẦU ---
    st.subheader("2. Độ lệch pha: Youtuber thích đăng giờ nào vs Khán giả xem giờ nào?")
    df_line_radar = apply_cross_filters(base_filtered_df, active_cross_filters, exclude_key="cross_trend")
    df_q3_line_radar = df_line_radar.dropna(subset=['video_view_count', 'hour']).copy()

    supply = df_q3_line_radar.groupby('hour').size().reset_index(name='video_count')
    demand = df_q3_line_radar.groupby('hour')['video_view_count'].sum().reset_index(name='total_views')

    df_trend = pd.merge(supply, demand, on='hour')

    fig8 = make_subplots(specs=[[{"secondary_y": True}]])

    fig8.add_trace(
        go.Bar(x=df_trend['hour'], y=df_trend['video_count'], name="Số lượng Video đăng (Cung)", marker_color='rgb(158,202,225)'),
        secondary_y=False,
    )
    fig8.add_trace(
        go.Scatter(x=df_trend['hour'], y=df_trend['total_views'], name="Tổng lượt xem (Cầu)", marker_color='rgb(227,26,28)', mode='lines+markers', line=dict(width=3)),
        secondary_y=True,
    )

    fig8.update_layout(
        title_text="Đối chiếu Số lượng Video phát hành và Tổng View theo Giờ",
        xaxis_title="Giờ trong ngày"
    )
    fig8.update_yaxes(title_text="Số lượng Video", secondary_y=False)
    fig8.update_yaxes(title_text="Tổng lượt xem", secondary_y=True)

    trend_event = st.plotly_chart(
        fig8,
        use_container_width=True,
        key="cross_trend",
        on_select="rerun",
    )
    sync_chart_selection("cross_trend", trend_event)
    st.caption("💡 Đường màu đỏ chênh lệch cao tại những cột màu xanh thấp chính là 'Đại dương xanh' - ít người cạnh tranh đăng bài nhưng lượng người xem lại cực lớn.")

    # --- BIỂU ĐỒ 3.3: 2D DENSITY CONTOUR PLOT ---
    st.subheader("3. Biểu đồ Đường đồng mức: Sự kết hợp hoàn hảo giữa Thời lượng và Lượt xem")

    df_contour = df_q3[df_q3['video_duration'] > 0].copy()

    df_contour['log_duration'] = np.log10(df_contour['video_duration'])
    df_contour['log_views'] = np.log10(df_contour['video_view_count'])

    fig9 = px.density_contour(
        df_contour,
        x='log_duration',
        y='log_views',
        color_discrete_sequence=['#FF1493'],
        labels={
            'log_duration': 'Thời lượng video (giây - Log)',
            'log_views': 'Lượt xem (Log)'
        },
        title="2D Density Contour: Vùng 'Đỉnh núi' tập trung nhiều view nhất"
    )
    fig9.update_traces(contours_coloring="fill", contours_showlabels=True)
    st.plotly_chart(fig9, use_container_width=True)
    st.caption("💡 Các vòng tròn đồng mức cho thấy 'đỉnh núi' (nơi tập trung đậm màu nhất). Trục X sẽ cho bạn biết thời lượng bao nhiêu giây là tối ưu để đạt được lượt xem cao nhất ở trục Y.")
