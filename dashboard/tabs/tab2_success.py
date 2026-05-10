import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_tab(base_filtered_df, active_cross_filters, apply_cross_filters, sync_chart_selection):
    st.info(
            "Một video âm nhạc thành công thường được hình thành từ những đặc điểm nào, và điều gì tạo ra sự khác biệt giữa các video có mức độ thành công khác nhau?"
        )

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
        color_discrete_map={True: '#1f77b4', False: '#E50914'},
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
    st.caption(
        "Mỗi bong bóng là một video; trục X/Y dùng thang log để dễ so sánh các mức view/like. "
        "Kích thước bong bóng biểu thị số bình luận và màu sắc phân biệt video có bản quyền hay không. "
        "Quan sát cụm điểm giúp đánh giá mối quan hệ giữa lượt xem và mức tương tác." 
    )

    # --- BIỂU ĐỒ 1.2: DENSITY HEATMAP ---
    st.subheader("2. Ma trận mật độ phân bổ tương tác dựa trên lượt xem")

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
        color_continuous_scale="Blues",
        text_auto=True,
        labels={
            'log_view': 'Lượt xem (Log)',
            'log_er': 'Tỷ lệ Tương tác (%) (Log)',
            'count': "Số lượng"
        },

    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Mỗi ô thể hiện mật độ video theo lượt xem và tỷ lệ tương tác (đã log). "
        "Ô càng sáng nghĩa là càng nhiều video rơi vào vùng đó. "
        "Tâm của vùng sáng cho biết mặt bằng chung của thị trường đang nghiêng về nhóm nào." 
    )

    # --- BIỂU ĐỒ 1.3: LOLLIPOP CHART ---
    st.subheader("3. Top 10 Kênh có Tỷ lệ Tương tác cao nhất")
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
        marker=dict(color='#1f77b4', size=12),
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
    st.caption(
        "Mỗi đường ngang biểu thị một kênh trong top 10 theo tổng view; điểm càng xa 0 thì tỷ lệ tương tác càng cao. "
        "Biểu đồ này nhấn mạnh chất lượng tương tác thay vì chỉ nhìn vào tổng lượt xem." 
    )

    st.subheader("4. So sánh nhóm view cao và view thấp theo các đặc trưng")
    df_compare_source = apply_cross_filters(base_filtered_df, active_cross_filters)
    if df_compare_source.empty or 'video_view_count' not in df_compare_source.columns:
        st.info("Không đủ dữ liệu để so sánh nhóm view cao và view thấp.")
    else:
        q_low = df_compare_source['video_view_count'].quantile(0.2)
        q_high = df_compare_source['video_view_count'].quantile(0.8)
        bottom_df = df_compare_source[df_compare_source['video_view_count'] <= q_low]
        top_df = df_compare_source[df_compare_source['video_view_count'] >= q_high]

        feature_specs = [
            ('video_like_count', 'Lượt thích'),
            ('video_comment_count', 'Bình luận'),
            ('video_tags_count', 'Số thẻ'),
            ('channel_subscriber_count', 'Subscriber kênh'),
            ('channel_view_count', 'Tổng view kênh'),
            ('channel_video_count', 'Số video kênh'),
            ('title_length', 'Độ dài tiêu đề'),
            ('video_duration', 'Thời lượng (giây)')
        ]

        rows = []
        for col, label in feature_specs:
            if col not in df_compare_source.columns:
                continue
            top_val = pd.to_numeric(top_df[col], errors='coerce').mean()
            bottom_val = pd.to_numeric(bottom_df[col], errors='coerce').mean()
            if pd.isna(top_val) and pd.isna(bottom_val):
                continue
            rows.append({'Đặc trưng': label, 'Nhóm': f'Top 20% (n={len(top_df)})', 'Giá trị': top_val})
            rows.append({'Đặc trưng': label, 'Nhóm': f'Bottom 20% (n={len(bottom_df)})', 'Giá trị': bottom_val})

        if rows:
            compare_df = pd.DataFrame(rows)
            compare_df['Giá trị'] = compare_df['Giá trị'].clip(lower=1)
            fig_compare = px.bar(
                compare_df,
                x='Đặc trưng',
                y='Giá trị',
                color='Nhóm',
                barmode='group',
                labels={'Giá trị': 'Giá trị (log)'},
                title="So sánh trung bình đặc trưng giữa nhóm view cao và view thấp"
            )
            fig_compare.update_yaxes(type='log', title_text='Giá trị (log)')
            st.plotly_chart(fig_compare, use_container_width=True)
            st.caption(
                "Biểu đồ so sánh giá trị trung bình của các đặc trưng giữa nhóm view cao (top 20%) "
                "và nhóm view thấp (bottom 20%) trên thang log để dễ nhìn các chênh lệch lớn. "
                "Đặc trưng có khoảng cách lớn giữa hai nhóm là ứng viên có tác động mạnh đến hiệu quả." 
            )
        else:
            st.info("Không đủ đặc trưng số để tạo biểu đồ so sánh.")
