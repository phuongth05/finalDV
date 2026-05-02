import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer


def render_tab(filtered_df, base_filtered_df, active_cross_filters, apply_cross_filters, sync_chart_selection):
    st.header("Câu 2: Khẩu vị khán giả & Sức mạnh của Từ khóa (NLP)")

    # Tiền xử lý cho Q2
    df_q2 = filtered_df.dropna(subset=['video_view_count']).copy()
    df_q2['video_made_for_kids'] = df_q2['video_made_for_kids'].fillna(False).astype(bool)
    df_q2['video_licensed_content'] = df_q2['video_licensed_content'].fillna(False).astype(bool)

    # --- BIỂU ĐỒ 2.1: TREEMAP TRỰC QUAN HÓA THỊ PHẦN ---
    st.subheader("1. Cơ cấu lượt xem: Đối tượng, Bản quyền và Chất lượng")

    df_q2['Doi_tuong'] = df_q2['video_made_for_kids'].map({True: 'Nhạc Trẻ Em', False: 'Nhạc Đại Chúng'})
    df_q2['Ban_quyen'] = df_q2['video_licensed_content'].map({True: 'Chính thức (Official)', False: 'Tự do (Cover/Remix)'})
    df_q2['Chat_luong'] = df_q2['video_definition'].astype(str).str.upper()

    df_q2['path_kids_license'] = df_q2['Doi_tuong'] + " - " + df_q2['Ban_quyen']

    fig4 = px.treemap(
        df_q2,
        path=['Doi_tuong', 'Ban_quyen', 'Chat_luong'],
        values='video_view_count',
        color='Doi_tuong',
        color_discrete_map={'Nhạc Trẻ Em': '#FF9900', 'Nhạc Đại Chúng': '#3366CC'},
        title="Treemap: Miếng bánh thị phần View trên YouTube Music VN"
    )
    fig4.update_traces(root_color="lightgrey", textinfo="label+percent parent")
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("💡 Nhìn vào diện tích các ô chữ nhật, ta thấy ngay nhóm nội dung nào đang chiếm trọn lượt xem.")

    # --- BIỂU ĐỒ 2.2: BOXPLOT CHO PHỤ ĐỀ ---
    st.subheader("2. Vai trò của Phụ đề (Lyrics/CC) đối với lượt xem")
    df_q2['has_caption'] = df_q2['video_caption_status'].astype(str).str.lower() == 'true'

    fig5 = px.box(
        df_q2,
        x='has_caption',
        y='video_view_count',
        color='has_caption',
        log_y=True,
        color_discrete_sequence=['#9C27B0', '#00BCD4'],
        labels={'has_caption': 'Có phụ đề (CC)', 'video_view_count': 'Lượt xem (Log)'},
        title="Box Plot: Phân phối lượt xem của video Có vs Không có phụ đề"
    )
    fig5.update_traces(
        hovertemplate=(
            "Có phụ đề: %{x}<br>"
            "Lượt xem gốc: %{y:,}<extra></extra>"
        )
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("💡 **Insight:** So sánh phần hộp (box) của hai màu. Nếu hộp của video Có phụ đề nằm ở mức view cao hơn đáng kể, chứng tỏ khán giả nhạc Việt rất quan tâm đến Lyrics/Vietsub.")

    # --- BIỂU ĐỒ 2.3: BAR CHART CHO NLP (TỪ KHÓA TRONG TITLE) ---
    st.subheader("3. Sức mạnh Từ khóa trong Tiêu đề (Định dạng nhạc)")
    df_bar = apply_cross_filters(base_filtered_df, active_cross_filters, exclude_key="cross_keywords")
    df_q2_bar = df_bar.dropna(subset=['video_view_count', 'video_title']).copy()

    titles = df_q2_bar['video_title'].dropna().astype(str).tolist()

    custom_stopwords = ['và', 'của', 'là', 'những', 'các', 'trong', 'với', 'cho', 'để', 'có', 'không', 'bài', 'hát', 'tập', 'phần', 'the', 'of', 'in', 'and', 'to', 'a', 'is', 'that', 'it', 'on', 'for', 'as', 'was', 'but', 'are']

    try:
        vectorizer = TfidfVectorizer(max_features=15, ngram_range=(1, 2), stop_words=custom_stopwords)
        vectorizer.fit(titles)
        top_keywords = vectorizer.get_feature_names_out()

        keyword_stats = []
        for kw in top_keywords:
            mask = df_q2_bar['video_title'].astype(str).str.contains(kw, case=False, na=False)
            count = mask.sum()
            if count > 0:
                avg_views = df_q2_bar[mask]['video_view_count'].mean()
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
                color_continuous_scale='Plasma',
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
            st.caption("💡 **Insight:** Biểu đồ này không dùng danh sách cho trước mà dùng AI (TF-IDF) tự động dò quét toàn bộ text. Những cụm từ có cột mọc dài nhất chính là 'công thức đặt tên' thu hút view khủng nhất trên thị trường.")
        else:
            st.warning("Không trích xuất được từ khóa nào thỏa mãn.")

    except Exception as e:
        st.error(f"Lỗi khi chạy TF-IDF: {e}. Vui lòng kiểm tra lại dữ liệu chữ.")
