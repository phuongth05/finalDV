import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_tab(base_filtered_df, active_cross_filters, apply_cross_filters, sync_chart_selection):
    st.header("Câu 1: Định nghĩa 'Thành công' của một sản phẩm âm nhạc")

    # Tiền xử lý dữ liệu cơ bản cho Câu 1
    df_bubble = apply_cross_filters(base_filtered_df, active_cross_filters, exclude_key="cross_bubble")
    df_q1 = df_bubble.dropna(subset=['video_view_count', 'video_like_count', 'video_comment_count']).copy()

    # --- BIỂU ĐỒ 1.1: BUBBLE CHART ---
    st.subheader("1. View cao có đi kèm lượng fan tương tác mạnh?")
    fig1 = px.scatter(
        df_q1,
        x='video_view_count',
        y='video_like_count',
        size='video_comment_count',
        color='video_licensed_content',
        hover_name='video_title',
        custom_data=['_row_id'],
        log_x=True,
        log_y=True,
        size_max=60,
        color_discrete_map={True: '#1DB954', False: '#E50914'},
        labels={
            'video_view_count': 'Lượt xem',
            'video_like_count': 'Lượt thích',
            'video_comment_count': 'Lượt bình luận',
            'video_licensed_content': 'Có bản quyền'
        },
        title="Bubble Chart: Tương quan View, Like, Comment & Bản quyền"
    )
    fig1.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "Lượt xem: %{x:,}<br>"
            "Lượt thích: %{y:,}<br>"
            "Lượt bình luận: %{marker.size:,}<extra></extra>"
        )
    )
    bubble_event = st.plotly_chart(
        fig1,
        use_container_width=True,
        key="cross_bubble",
        on_select="rerun",
    )
    sync_chart_selection("cross_bubble", bubble_event)
    st.caption("💡 **Insight:** Những bong bóng to nhất biểu thị video gây bão bình luận. So sánh màu sắc để xem nhạc có bản quyền hay nhạc tự do đem lại nhiều tương tác hơn.")

    # --- BIỂU ĐỒ 1.2: DENSITY HEATMAP ---
    st.subheader("2. Ma trận Thành công: Mật độ phân bổ Video")

    df_q1['engagement_rate'] = ((df_q1['video_like_count'] + df_q1['video_comment_count']) / df_q1['video_view_count']) * 100
    df_q1_clean = df_q1[(df_q1['engagement_rate'] > 0) & (df_q1['video_view_count'] > 0) & (df_q1['engagement_rate'] < 100)]

    df_q1_clean['log_view'] = np.log10(df_q1_clean['video_view_count'])
    df_q1_clean['log_er'] = np.log10(df_q1_clean['engagement_rate'])

    fig2 = px.density_heatmap(
        df_q1_clean,
        x='log_view',
        y='log_er',
        nbinsx=30,
        nbinsy=30,
        color_continuous_scale="Viridis",
        text_auto=True,
        labels={
            'log_view': 'Lượt xem (Log)',
            'log_er': 'Tỷ lệ Tương tác (%) (Log)',
        },
        title="2D Density Heatmap: Đa số video rơi vào nhóm 'Viral' hay 'Mì ăn liền'?"
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("💡 **Insight:** Các ô vuông sáng màu (vàng/xanh lá) thể hiện nơi tập trung đông đúc nhất. Nếu đỉnh tập trung nằm ở góc phải phía trên, thị trường đang có nhiều video 'Siêu phẩm' (View cao + Tương tác mạnh).")

    # --- BIỂU ĐỒ 1.3: LOLLIPOP CHART ---
    st.subheader("3. Top 10 Kênh có Tỷ lệ Tương tác (Engagement Rate) cao nhất")
    df_lollipop = apply_cross_filters(base_filtered_df, active_cross_filters, exclude_key="cross_lollipop")
    df_q1_lollipop = df_lollipop.dropna(subset=['video_view_count', 'video_like_count', 'video_comment_count']).copy()

    top_10_channels = df_q1_lollipop.groupby('channel_title')['video_view_count'].sum().nlargest(10).index
    df_top10 = df_q1_lollipop[df_q1_lollipop['channel_title'].isin(top_10_channels)]

    df_grouped = df_top10.groupby('channel_title')[['video_view_count', 'video_like_count', 'video_comment_count']].sum().reset_index()
    df_grouped['engagement_rate'] = (df_grouped['video_like_count'] + df_grouped['video_comment_count']) / df_grouped['video_view_count']
    df_grouped = df_grouped.sort_values(by='engagement_rate', ascending=True)

    fig3 = go.Figure()
    for _, row in df_grouped.iterrows():
        fig3.add_shape(
            type="line",
            x0=0, x1=row['engagement_rate'],
            y0=row['channel_title'], y1=row['channel_title'],
            line=dict(color="#888888", width=2)
        )
    fig3.add_trace(go.Scatter(
        x=df_grouped['engagement_rate'],
        y=df_grouped['channel_title'],
        mode='markers',
        customdata=df_grouped['channel_title'],
        marker=dict(color='#FF5722', size=12),
        name='Engagement Rate',
        hovertemplate="Kênh: %{y}<br>Tỷ lệ tương tác: %{x:.4f}<extra></extra>"
    ))

    fig3.update_layout(
        title="Lollipop Chart: Tỷ lệ tương tác (Like+Comment / View) của Top 10 Kênh",
        xaxis_title="Tỷ lệ tương tác",
        yaxis_title="Kênh",
        showlegend=False,
        height=500
    )
    lollipop_event = st.plotly_chart(
        fig3,
        use_container_width=True,
        key="cross_lollipop",
        on_select="rerun",
    )
    sync_chart_selection("cross_lollipop", lollipop_event)
    st.caption("💡 **Insight:** Kênh có thanh kẹo dài nhất là kênh tuy có thể ít View, nhưng sở hữu lượng fan 'cày' like và comment nhiệt tình và trung thành nhất.")
