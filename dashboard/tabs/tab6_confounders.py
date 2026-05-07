
# tabs/tab6_confounders.py

import pandas as pd
import plotly.express as px
import streamlit as st
from groq import Groq


def create_data_summary(_df):
    """Tóm tắt dữ liệu để gửi cho AI"""
    summary = f"""
    BẠN LÀ DATA ANALYST PHÂN TÍCH DỮ LIỆU YOUTUBE ÂM NHẠC VIỆT NAM.
    Trả lời BẰNG TIẾNG VIỆT, ngắn gọn, có số liệu cụ thể.

    THÔNG TIN DATASET:
    - Tổng số video: {len(_df):,}
    - Tổng số kênh: {_df['channel_id'].nunique():,}
    - Khoảng thời gian: {_df['video_publish_date'].min().strftime('%Y-%m-%d')} đến {_df['video_publish_date'].max().strftime('%Y-%m-%d')}

    THỐNG KÊ VIEWS:
    - Trung bình: {_df['video_view_count'].mean():,.0f}
    - Trung vị: {_df['video_view_count'].median():,.0f}
    - Cao nhất: {_df['video_view_count'].max():,.0f}
    - Thấp nhất: {_df['video_view_count'].min():,.0f}

    THỐNG KÊ LIKES:
    - Trung bình: {_df['video_like_count'].mean():,.0f}
    - Cao nhất: {_df['video_like_count'].max():,.0f}

    THỐNG KÊ COMMENTS:
    - Trung bình: {_df['video_comment_count'].mean():,.0f}

    TOP 10 KÊNH (theo tổng views):
    {_df.groupby('channel_title')['video_view_count'].sum().nlargest(10).to_string()}

    TOP 5 THỂ LOẠI (theo số video):
    {_df['genre'].value_counts().head(5).to_string()}

    TOP 5 THỂ LOẠI (theo views trung bình):
    {_df.groupby('genre')['video_view_count'].mean().nlargest(5).round(0).to_string()}

    PHÂN BỐ VIDEO TYPE:
    {_df['video_type'].value_counts().to_string()}

    GIỜ ĐĂNG PHỔ BIẾN NHẤT:
    {_df['hour'].value_counts().head(5).to_string()}

    ENGAGEMENT RATE TRUNG BÌNH THEO THỂ LOẠI: {_df.groupby('genre')['engagement_rate'].mean().to_string()}

    TOP 10 VIDEO NHIỀU VIEW NHẤT:
    {_df.nlargest(10, 'video_view_count')[['video_title', 'channel_title', 'video_view_count']].to_string(index=False)}

    THÔNG TIN CHANNEL SIZE:
    {_df.groupby('channel_size', observed=True).agg(
        count=('video_id', 'count'),
        avg_views=('video_view_count', 'mean')
    ).round(0).to_string()}
    """
    return summary


def ask_groq(question, data_context):
    """Gọi Groq API với câu hỏi + context dữ liệu"""
    try:
        # Lấy API key từ secrets hoặc hardcode (không khuyến nghị)
        api_key = st.secrets["GROQ_API_KEY"]
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": data_context},
                {"role": "user", "content": question}
            ],
            temperature=0.3,
            max_tokens=1024
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Lỗi khi gọi AI: {str(e)}"


def auto_chart(question, data):
    """Tự động tạo chart phù hợp với câu hỏi"""
    q = question.lower()

    # ── Top video / viral ──────────────────────
    if any(kw in q for kw in [
        'top video', 'viral', 'video nào',
        'nhiều view nhất', 'hot nhất','nổi bật',"lượt xem nhất"
    ]):
        top_df = data.nlargest(
            10, 'video_view_count'
        )[['video_title', 'channel_title',
           'video_view_count']].copy()
        top_df['video_title'] = (
            top_df['video_title'].astype(str).str[:40] + '...'
        )
        fig = px.bar(
            top_df,
            x='video_view_count',
            y='video_title',
            orientation='h',
            color='channel_title',
            title='🔥 Top 10 Video nhiều view nhất',
            labels={
                'video_view_count': 'Lượt xem',
                'video_title': 'Video',
                'channel_title': 'Kênh'
            },
            height=450
        )
        fig.update_layout(yaxis={'autorange': 'reversed'})
        return fig

    # ── Thể loại / genre ──────────────────────
    elif any(kw in q for kw in [
        'view trung bình', 'genre', 'loại nhạc',
        'bolero', 'rap', 'indie', 'remix',
        'nhạc nào'
    ]):
        genre_stats = data.groupby('genre').agg(
            count=('video_id', 'count'),
            avg_views=('video_view_count', 'mean')
        ).reset_index().sort_values(
            'avg_views', ascending=False
        )
        fig = px.bar(
            genre_stats,
            x='genre', y='avg_views',
            color='count',
            title='🎵 Views trung bình theo thể loại',
            labels={
                'genre': 'Thể loại',
                'avg_views': 'Views trung bình',
                'count': 'Số video'
            },
            text=genre_stats['avg_views'].apply(
                lambda x: f'{x:,.0f}'
            ),
            height=450
        )
        fig.update_traces(textposition='outside')
        return fig

    # ── Giờ đăng / thời gian ─────────────────
    elif any(kw in q for kw in [
        'giờ', 'thời gian', 'time', 'khi nào',
        'lúc nào', 'đăng khi', 'best time'
    ]):
        hour_stats = data.groupby('hour').agg(
            avg_views=('video_view_count', 'mean'),
            count=('video_id', 'count')
        ).reset_index()
        fig = px.bar(
            hour_stats,
            x='hour', y='avg_views',
            title='⏰ Views trung bình theo giờ đăng',
            labels={
                'hour': 'Giờ đăng',
                'avg_views': 'Views trung bình'
            },
            color='avg_views',
            color_continuous_scale='Blues',
            height=400
        )
        return fig

    # ── Kênh / channel ────────────────────────
    elif any(kw in q for kw in [
        'kênh', 'channel', 'ca sĩ', 'nghệ sĩ',
        'ai có', 'top kênh'
    ]):
        ch_stats = data.groupby(
            'channel_title'
        ).agg(
            total_views=(
                'video_view_count', 'sum'
            ),
            videos=('video_id', 'count'),
            avg_views=(
                'video_view_count', 'mean'
            )
        ).nlargest(
            10, 'total_views'
        ).reset_index()

        fig = px.bar(
            ch_stats,
            x='total_views',
            y='channel_title',
            orientation='h',
            color='videos',
            title='📺 Top 10 Kênh theo tổng views',
            labels={
                'total_views': 'Tổng views',
                'channel_title': 'Kênh',
                'videos': 'Số video'
            },
            height=450
        )
        fig.update_layout(
            yaxis={'autorange': 'reversed'}
        )
        return fig

    # ── Engagement ────────────────────────────
    elif any(kw in q for kw in [
        'engagement', 'tương tác', 'like',
        'comment', 'bình luận', 'thích'
    ]):
        eng_by_genre = data.groupby(
            'genre'
        )['engagement_rate'].mean().reset_index()
        eng_by_genre = eng_by_genre.sort_values(
            'engagement_rate', ascending=False
        )
        fig = px.bar(
            eng_by_genre,
            x='genre',
            y='engagement_rate',
            title=(
                '💬 Engagement Rate '
                'theo thể loại'
            ),
            labels={
                'genre': 'Thể loại',
                'engagement_rate':
                    'Engagement Rate'
            },
            color='engagement_rate',
            color_continuous_scale='Greens',
            height=400
        )
        return fig

    # ── So sánh ──────────────────────────────
    elif any(kw in q for kw in [
        'so sánh', 'khác nhau', 'vs',
        'hơn', 'thua', 'compare'
    ]):
        fig = px.box(
            data,
            x='genre',
            y='video_view_count',
            title='📊 So sánh Views theo thể loại',
            labels={
                'genre': 'Thể loại',
                'video_view_count': 'Lượt xem'
            },
            height=450
        )
        return fig

    # ── Phân bố / distribution ────────────────
    elif any(kw in q for kw in [
        'phân bố', 'distribution',
        'histogram', 'biểu đồ'
    ]):
        fig = px.histogram(
            data,
            x='video_view_count',
            nbins=50,
            title='📊 Phân bố lượt xem',
            labels={
                'video_view_count': 'Lượt xem'
            },
            height=400
        )
        return fig

    # ── Duration / thời lượng ─────────────────
    elif any(kw in q for kw in [
        'thời lượng', 'duration', 'dài',
        'ngắn', 'phút', 'giây'
    ]):
        fig = px.scatter(
            data.sample(
                min(1000, len(data))
            ),
            x='video_duration',
            y='video_view_count',
            color='genre',
            opacity=0.4,
            title=(
                '⏱️ Thời lượng vs '
                'Lượt xem'
            ),
            labels={
                'video_duration':
                    'Thời lượng (giây)',
                'video_view_count':
                    'Lượt xem'
            },
            height=450
        )
        return fig

    # ── Confounder ────────────────────────────
    elif any(kw in q for kw in [
        'confounder', 'nhiễu', 'ẩn',
        'stepwise'
    ]):
        fig = px.scatter(
            data.sample(
                min(1000, len(data))
            ),
            x='video_duration',
            y='video_view_count',
            color='channel_size',
            trendline='ols',
            opacity=0.3,
            title=(
                '🔍 Confounder: '
                'Duration vs Views '
                '(theo Channel Size)'
            ),
            labels={
                'video_duration':
                    'Thời lượng (giây)',
                'video_view_count':
                    'Lượt xem',
                'channel_size':
                    'Quy mô kênh'
            },
            height=450
        )
        return fig

    # ── Không match → không vẽ chart ──────────
    return None


def render_tab(filtered_df: pd.DataFrame):
    """Hàm chính được gọi từ app.py"""

    st.subheader("🤖 AI Data Assistant")

    # Tạo data summary
    data_summary = create_data_summary(filtered_df)

    # Khởi tạo session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_chart" not in st.session_state:
        st.session_state.current_chart = None

    # Layout: Chat + Chart
    chat_col, chart_col = st.columns([1, 1])

    # ── Cột trái: Chat ───────────────────────────────
    with chat_col:
        st.markdown("#### 💬 Chat")

        chat_container = st.container(height=500)
        with chat_container:
            if not st.session_state.messages:
                st.markdown(
                    "🤖 Xin chào! Tôi là trợ lý "
                    "phân tích dữ liệu YouTube "
                    "âm nhạc Việt Nam. Hãy hỏi "
                    "tôi bất cứ điều gì về dữ liệu!"
                )

            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Input câu hỏi
        if prompt := st.chat_input(
            "Hỏi về dữ liệu YouTube âm nhạc VN..."
        ):
            st.session_state.messages.append(
                {"role": "user", "content": prompt}
            )

            with st.spinner("🤖 Đang phân tích..."):
                ai_response = ask_groq(
                    prompt, data_summary
                )

            st.session_state.messages.append(
                {"role": "assistant", "content": ai_response}
            )

            chart = auto_chart(prompt, filtered_df)
            if chart is not None:
                st.session_state.current_chart = chart

            st.rerun()

        # Gợi ý câu hỏi
        st.markdown("**💡 Gợi ý:**")
        sug_col1, sug_col2 = st.columns(2)

        suggestions = [
            ("🔥 Top video viral", "Top 10 video có nhiều lượt xem nhất?"),
            ("🎵 Thể loại hot", "Thể loại nhạc nào có view cao nhất?"),
            ("⏰ Giờ đăng phổ biến", "Giờ nào là video được đăng nhiều nhất?"),
            ("📺 Top kênh", "Kênh nào có tổng lượt xem nhiều nhất?"),
            ("💬 Tỷ lệ tương tác", "Thể loại nào có tỷ lệ tương tác cao nhất?"),
        ]

        for i, (label, question) in enumerate(suggestions):
            col = sug_col1 if i % 2 == 0 else sug_col2
            if col.button(
                label,
                key=f"sug_ai_{i}",
                use_container_width=True
            ):
                st.session_state.messages.append(
                    {"role": "user", "content": question}
                )

                with st.spinner("🤖 Đang phân tích..."):
                    ai_resp = ask_groq(
                        question, data_summary
                    )

                st.session_state.messages.append(
                    {"role": "assistant", "content": ai_resp}
                )

                chart = auto_chart(question, filtered_df)
                if chart is not None:
                    st.session_state.current_chart = chart

                st.rerun()

    # ── Cột phải: Chart panel ─────────────────────────
    with chart_col:
        st.markdown("#### 📊 Biểu đồ")

        if st.session_state.current_chart is not None:
            st.plotly_chart(
                st.session_state.current_chart,
                use_container_width=True
            )

            if st.button(
                "🗑️ Xóa biểu đồ",
                key="clear_chart"
            ):
                st.session_state.current_chart = None
                st.rerun()
        else:
            st.info(
                "📌 Biểu đồ sẽ tự động hiển thị "
                "khi bạn hỏi câu hỏi liên quan.\n\n"
                "**Thử hỏi:**\n"
                "- \"Top video viral nhất?\"\n"
                "- \"Thể loại nào hot nhất?\"\n"
                "- \"Giờ đăng tốt nhất?\"\n"
                "- \"So sánh views theo genre\""
            )

    # ── Nút xóa chat ─────────────────────────────────
    st.markdown("---")
    if st.button(
        "🗑️ Xóa toàn bộ cuộc hội thoại",
        key="clear_all"
    ):
        st.session_state.messages = []
        st.session_state.current_chart = None
        st.rerun()
