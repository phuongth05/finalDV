import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer


def render_tab(filtered_df, base_filtered_df, active_cross_filters, apply_cross_filters, sync_chart_selection):
    st.header("Câu 3: Yếu tố nền tảng và tối ưu thuật toán")

    df_q3 = filtered_df.dropna(subset=['video_view_count', 'hour', 'day']).copy()

    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_q3['day'] = pd.Categorical(df_q3['day'], categories=days_order, ordered=True)

    # --- BIỂU ĐỒ 3.1: HEATMAP THỜI GIAN ĐĂNG BÀI ---
    st.subheader("1. Bản đồ nhiệt (Heatmap): Ảnh hưởng của thời điểm đăng (giờ, ngày) đến lượt xem")

    heat_data = df_q3.groupby(['day', 'hour'])['video_view_count'].mean().reset_index()
    heat_pivot = heat_data.pivot(index='day', columns='hour', values='video_view_count')
    heat_pivot = heat_pivot.reindex(index=days_order)
    heat_pivot = heat_pivot.reindex(columns=sorted(heat_pivot.columns))

    # Drop NaN values to avoid issues with imshow
    heat_pivot = heat_pivot.fillna(0)

    fig7 = px.imshow(
        heat_pivot,
        labels=dict(x="Giờ trong ngày (0-23h)", y="Ngày trong tuần", color="Trung bình View"),
        color_continuous_scale='Blues',
        aspect="auto",
        title="Heatmap: Tương quan giữa Giờ đăng, Ngày đăng và Lượt xem"
    )
    st.plotly_chart(fig7, use_container_width=True)
    st.caption(
        "Mỗi ô là lượt xem trung bình tại một giờ và một ngày trong tuần. "
        "Màu càng sáng nghĩa là hiệu suất càng cao. "
        "Biểu đồ này giúp xác định khung giờ/khung ngày nên ưu tiên đăng bài." 
    )

    st.subheader("2. Mức độ hiệu quả theo ngày trong tuần và theo giờ")
    col_day, col_hour = st.columns(2)

    with col_day:
        day_views = (
            df_q3.groupby('day')['video_view_count']
            .mean()
            .reindex(days_order)
            .reset_index()
        )
        fig_day = px.bar(
            day_views,
            x='day',
            y='video_view_count',
            labels={'day': 'Ngày trong tuần', 'video_view_count': 'Lượt xem trung bình'},
            title="Lượt xem trung bình theo ngày"
        )
        st.plotly_chart(fig_day, use_container_width=True)

    with col_hour:
        hour_views = df_q3.groupby('hour')['video_view_count'].mean().reset_index()
        fig_hour = px.bar(
            hour_views,
            x='hour',
            y='video_view_count',
            labels={'hour': 'Giờ trong ngày', 'video_view_count': 'Lượt xem trung bình'},
            title="Lượt xem trung bình theo giờ"
        )
        st.plotly_chart(fig_hour, use_container_width=True)

    st.caption(
        "Hai biểu đồ này tóm tắt hiệu quả theo từng ngày và từng giờ. "
        "So sánh các cột để nhận biết thời điểm có mức view trung bình cao hơn." 
    )

    # --- BIỂU ĐỒ 3.2: LINE/RADAR CHART - CUNG VS CẦU ---
    st.subheader("3. Độ lệch pha(Bar & Line Chart): Số lượng Video đăng (Cung) và Tổng View (Cầu) theo Giờ đăng")
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
    st.caption(
        "Cột xanh thể hiện số video được đăng (cung), đường đỏ là tổng lượt xem (cầu). "
        "Khoảng cách giữa hai đường cho biết nơi cung thấp nhưng cầu cao hoặc ngược lại, "
        "từ đó gợi ý thời điểm đăng bài hiệu quả hơn." 
    )

    st.subheader("4. Độ dài video và hiệu suất (scatter)")
    df_duration = filtered_df.dropna(subset=['video_duration', 'video_view_count']).copy()
    df_duration = df_duration[df_duration['video_duration'] > 0]
    if df_duration.empty:
        st.info("Không đủ dữ liệu thời lượng để phân tích hiệu suất.")
    else:
        df_duration['duration_min'] = df_duration['video_duration'] / 60
        fig_duration = px.scatter(
            df_duration,
            x='duration_min',
            y='video_view_count',
            log_y=True,
            labels={'duration_min': 'Thời lượng (phút)', 'video_view_count': 'Lượt xem (log)'},
            title="Quan hệ giữa thời lượng video và lượt xem"
        )
        st.plotly_chart(fig_duration, use_container_width=True)
        st.caption(
            "Mỗi điểm là một video; trục Y dùng log để dễ thấy sự khác biệt. "
            "Biểu đồ giúp kiểm tra xem thời lượng dài/ngắn có liên quan tới hiệu suất không." 
        )

    st.subheader("5. So sánh nhóm độ dài video")
    if df_duration.empty:
        st.info("Không đủ dữ liệu để so sánh nhóm thời lượng.")
    else:
        bins = [0, 5, 10, np.inf]
        labels = ['< 5 phút', '5-10 phút', '> 10 phút']
        df_duration['duration_group'] = pd.cut(df_duration['duration_min'], bins=bins, labels=labels, right=False)
        group_stats = (
            df_duration.groupby('duration_group')['video_view_count']
            .agg(['mean', 'count'])
            .reset_index()
        )
        fig_group = px.bar(
            group_stats,
            x='duration_group',
            y='mean',
            text='count',
            labels={'duration_group': 'Nhóm thời lượng', 'mean': 'Lượt xem trung bình'},
            title="So sánh hiệu suất theo nhóm thời lượng"
        )
        fig_group.update_traces(texttemplate='n=%{text}', textposition='outside')
        st.plotly_chart(fig_group, use_container_width=True)
        st.caption(
            "Cột biểu thị lượt xem trung bình của từng nhóm thời lượng, số lượng video hiển thị trên nhãn. "
            "Nhìn vào độ cao cột để chọn khung thời lượng hiệu quả hơn." 
        )

    # --- BIỂU ĐỒ 3.3: 2D DENSITY CONTOUR PLOT ---
    st.subheader("6. Biểu đồ Đường đồng mức: Sự kết hợp hoàn hảo giữa Thời lượng và Lượt xem")

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
    st.caption(
        "Các vòng đồng mức biểu thị mật độ video theo thời lượng và lượt xem (đã log). "
        "Vùng đậm nhất là nơi tập trung nhiều video, thường đại diện cho độ dài phổ biến và hiệu quả. "
        "Biểu đồ này giúp ước lượng khoảng thời lượng phù hợp để tối ưu lượt xem." 
    )

    # --- BIỂU ĐỒ 3.7: BAR CHART CHO NLP (TỪ KHÓA TRONG TITLE) ---
    st.subheader("7. Sức mạnh Từ khóa trong Tiêu đề (Định dạng nhạc)")
    df_bar = apply_cross_filters(base_filtered_df, active_cross_filters, exclude_key="cross_keywords")
    df_q3_bar = df_bar.dropna(subset=['video_view_count', 'video_title']).copy()

    titles = df_q3_bar['video_title'].dropna().astype(str).tolist()

    custom_stopwords = ['và', 'của', 'là', 'những', 'các', 'trong', 'với', 'cho', 'để', 'có', 'không', 'bài', 'hát', 'tập', 'phần', 'the', 'of', 'in', 'and', 'to', 'a', 'is', 'that', 'it', 'on', 'for', 'as', 'was', 'but', 'are']

    try:
        vectorizer = TfidfVectorizer(max_features=15, ngram_range=(1, 2), stop_words=custom_stopwords)
        vectorizer.fit(titles)
        top_keywords = vectorizer.get_feature_names_out()

        keyword_stats = []
        for kw in top_keywords:
            mask = df_q3_bar['video_title'].astype(str).str.contains(kw, case=False, na=False)
            count = mask.sum()
            if count > 0:
                avg_views = df_q3_bar[mask]['video_view_count'].mean()
                keyword_stats.append({
                    'Keyword': kw.upper(),
                    'Avg_Views': avg_views,
                    'Video_Count': count
                })

        if keyword_stats:
            df_kw = pd.DataFrame(keyword_stats).sort_values(by='Avg_Views', ascending=True)

            fig6 = px.bar(
                df_kw,
                x='Avg_Views',
                y='Keyword',
                orientation='h',
                color='Video_Count',
                text_auto=True,
                custom_data=['Keyword'],
                color_continuous_scale='Blues',
                labels={'Avg_Views': 'Trung bình Lượt xem', 'Keyword': 'Từ khóa (TF-IDF)', 'Video_Count': 'Số Video'},
                title="Horizontal Bar Chart: Top 15 Cụm từ quan trọng nhất tự động trích xuất bởi TF-IDF"
            )
            fig6.update_traces(texttemplate='%{x:,.0f}')
            keyword_event = st.plotly_chart(
                fig6,
                use_container_width=True,
                key="cross_keywords",
                on_select="rerun",
            )
            sync_chart_selection("cross_keywords", keyword_event)
            st.caption(
                "Biểu đồ tổng hợp các từ khóa xuất hiện nhiều và gắn với lượt xem trung bình cao. "
                "Màu thể hiện số video chứa từ khóa đó; cột càng dài nghĩa là nhóm video có từ khóa này "
                "đang đạt mức view trung bình tốt hơn." 
            )
        else:
            st.warning("Không trích xuất được từ khóa nào thỏa mãn.")

    except Exception as e:
        st.error(f"Lỗi khi chạy TF-IDF: {e}. Vui lòng kiểm tra lại dữ liệu chữ.")

    st.subheader("8. Word cloud từ title/description")
    image_path = Path("dashboard/images/wordcloud.png")
    st.image(str(image_path), width=1000)
    st.caption(
        "Word cloud trực quan hóa tần suất xuất hiện của các từ trong tiêu đề và mô tả video. "
        "Từ nào càng lớn nghĩa là xuất hiện càng nhiều, giúp nhận diện xu hướng chủ đề phổ biến." 
    )