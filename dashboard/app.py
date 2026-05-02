import json
import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from sklearn.feature_extraction.text import TfidfVectorizer

# ================= CONFIG =================
st.set_page_config(
    page_title="Phân tích nhạc YouTube",
    layout="wide",
    page_icon="🎧"
)

# ================= STYLE =================
st.markdown("""
<style>
.main-title {
    font-size: 42px !important;
    font-weight: 800;
    line-height: 1.08;
    letter-spacing: -0.03em;
    margin-bottom: 0.35rem;
}
.big-title {
    font-size:28px !important;
    font-weight:700;
}
.sidebar-card {
    background: linear-gradient(180deg, rgba(29,185,84,0.12), rgba(17,17,17,0.96));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 16px 16px 12px 16px;
    margin-bottom: 14px;
}
.sidebar-card h3 {
    margin: 0 0 6px 0;
    font-size: 18px;
}
.sidebar-card p {
    margin: 0;
    font-size: 13px;
    line-height: 1.5;
    color: rgba(255,255,255,0.78);
}
.sidebar-filter-title {
    font-size: 16px !important;
    font-weight: 700;
    margin-top: 0.7rem;
}
.metric-card {
    background-color:#111;
    padding:15px;
    border-radius:12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🎧 Phân tích nhạc YouTube Việt Nam</p>', unsafe_allow_html=True)

# ================= LOAD =================
df = pd.read_csv("data/youtube_vn_music_cleaned.csv")
df["_row_id"] = df.index.astype(str)

# ================= FEATURE ENGINEERING =================
df['video_publish_date'] = pd.to_datetime(df['video_publish_date'], errors='coerce')
df['hour'] = df['video_publish_date'].dt.hour
df['day'] = df['video_publish_date'].dt.day_name()

# basic features
df['title_length'] = df['video_title'].astype(str).apply(len)

# ================= SIDEBAR =================
        # <h1>Điều khiển</h1>
st.sidebar.markdown(
    """
    <div class="sidebar-card">
        <p>Chọn khoảng thời gian, khoảng lượt xem, rồi dùng cross-filter trên các biểu đồ để khoanh vùng dữ liệu nhanh hơn.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div class="sidebar-filter-title">Bộ lọc chính</div>', unsafe_allow_html=True)

date_range = st.sidebar.date_input(
    "Khoảng ngày",
    [df['video_publish_date'].min().date(), df['video_publish_date'].max().date()]
)

view_range = st.sidebar.slider(
    "Khoảng lượt xem",
    int(df['video_view_count'].min()),
    int(df['video_view_count'].max()),
    (int(df['video_view_count'].min()), int(df['video_view_count'].max()))
)

# convert to date
df['publish_date'] = df['video_publish_date'].dt.date

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

base_filtered_df = df[
    (df['publish_date'] >= start_date) &
    (df['publish_date'] <= end_date) &
    (df['video_view_count'].between(*view_range))
].copy()

if "cross_filter_disabled" not in st.session_state:
    st.session_state.cross_filter_disabled = {}
if "cross_filter_cache" not in st.session_state:
    st.session_state.cross_filter_cache = {}

# dọn key cross-filter cũ đã ngưng sử dụng
# for stale_key in ["cross_heatmap"]:
#     st.session_state.cross_filter_disabled.pop(stale_key, None)
#     st.session_state.cross_filter_cache.pop(stale_key, None)
#     st.session_state.cross_filter_cache.pop(f"{stale_key}__signature", None)


def _selection_state(widget_key):
    cached_selection = st.session_state.cross_filter_cache.get(widget_key)
    if cached_selection:
        return {"points": cached_selection}

    state = st.session_state.get(widget_key)
    if state is None:
        return None
    if isinstance(state, dict):
        return state.get("selection", state)
    return getattr(state, "selection", state)


def _selection_signature(points):
    try:
        return json.dumps(points, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return repr(points)


def _current_selection_signature(widget_key):
    selection = _selection_state(widget_key)
    if selection is None:
        return ""
    points = selection.get("points", []) if isinstance(selection, dict) else getattr(selection, "points", [])
    return _selection_signature(points or [])


def _selection_points(selection_state):
    if selection_state is None:
        return []
    if isinstance(selection_state, dict):
        return selection_state.get("points", []) or []
    return getattr(selection_state, "points", []) or []


def _sync_chart_selection(widget_key, event):
    selection = _selection_points(event)
    signature = _selection_signature(selection)
    previous_signature = st.session_state.cross_filter_cache.get(f"{widget_key}__signature", "")

    if selection:
        st.session_state.cross_filter_cache[widget_key] = selection
        st.session_state.cross_filter_cache[f"{widget_key}__signature"] = signature
    else:
        if widget_key not in st.session_state.cross_filter_cache:
            st.session_state.cross_filter_cache[f"{widget_key}__signature"] = ""

    return selection and signature != previous_signature


def _point_value(point, field):
    if isinstance(point, dict):
        return point.get(field)
    return getattr(point, field, None)


def _custom_value(point, index=0):
    customdata = _point_value(point, "customdata")
    if customdata is None:
        return None
    if isinstance(customdata, (list, tuple, np.ndarray)):
        if len(customdata) > index:
            return customdata[index]
        return None
    return customdata


def _points_to_row_ids(points):
    row_ids = []
    for point in points:
        row_id = _custom_value(point, 0)
        if row_id is None:
            row_id = _point_value(point, "point_index")
        if row_id is not None:
            row_ids.append(str(row_id))
    return sorted(set(row_ids))


def _points_to_keywords(points):
    keywords = []
    for point in points:
        keyword = _custom_value(point, 0)
        if keyword is None:
            keyword = _point_value(point, "y")
        if keyword is not None:
            keywords.append(str(keyword))
    return sorted(set(keywords))


def _points_to_channels(points):
    channels = []
    for point in points:
        channel = _custom_value(point, 0)
        if channel is None:
            channel = _point_value(point, "y")
        if channel is not None:
            channels.append(str(channel))
    return sorted(set(channels))


def _points_to_hour_day_pairs(points):
    pairs = []
    for point in points:
        hour = _point_value(point, "x")
        day = _point_value(point, "y")
        if hour is None or day is None:
            continue
        try:
            hour = int(hour)
        except (TypeError, ValueError):
            continue
        pairs.append((str(day), hour))
    return sorted(set(pairs))


def _points_to_hours(points):
    hours = []
    for point in points:
        hour = _point_value(point, "x")
        if hour is None:
            continue
        try:
            hours.append(int(hour))
        except (TypeError, ValueError):
            continue
    return sorted(set(hours))


def _collect_cross_filters(source_df):
    specs = [
        {"key": "cross_bubble", "label": "Bubble chart: video được chọn", "kind": "row_ids", "extract": _points_to_row_ids},
        {"key": "cross_lollipop", "label": "Lollipop chart: kênh được chọn", "kind": "channels", "extract": _points_to_channels},
        {"key": "cross_keywords", "label": "TF-IDF: từ khóa được chọn", "kind": "keywords", "extract": _points_to_keywords},
        {"key": "cross_trend", "label": "Biểu đồ theo giờ: giờ được chọn", "kind": "hours", "extract": _points_to_hours},
    ]

    active_filters = []
    disabled = st.session_state.cross_filter_disabled

    for spec in specs:
        selection = _selection_state(spec["key"])
        points = _selection_points(selection)
        if not points:
            continue

        signature = _selection_signature(points)
        if disabled.get(spec["key"]) == signature:
            continue

        values = spec["extract"](points)
        if not values:
            continue

        if spec["kind"] == "row_ids":
            summary = f"{len(values)} video"
        elif spec["kind"] == "channels":
            summary = f"{len(values)} kênh"
        elif spec["kind"] == "keywords":
            summary = f"{len(values)} từ khóa"
        elif spec["kind"] == "hours":
            summary = f"{', '.join(map(str, values))}"
        else:
            summary = f"{len(values)} ô"

        active_filters.append(
            {
                "key": spec["key"],
                "label": f"{spec['label']} ({summary})",
                "kind": spec["kind"],
                "values": values,
                "signature": signature,
            }
        )

    return active_filters


def _apply_cross_filters(frame, active_filters, exclude_key=None):
    result = frame.copy()

    for item in active_filters:
        # bỏ qua filter nếu đã bị vô hiệu hóa trong session state
        if item["key"] == exclude_key:
            continue


        kind = item["kind"]
        values = item["values"]

        if kind == "row_ids":
            result = result[result["_row_id"].astype(str).isin(values)]
        elif kind == "channels":
            result = result[result["channel_title"].astype(str).isin(values)]
        elif kind == "keywords":
            title_series = result["video_title"].astype(str)
            mask = pd.Series(False, index=result.index)
            for keyword in values:
                mask |= title_series.str.contains(re.escape(keyword), case=False, na=False)
            result = result[mask]
        elif kind == "hours":
            result = result[result["hour"].isin(values)]
        elif kind == "day_hour_pairs":
            pairs = set(values)
            mask = result.apply(
                lambda row: (str(row["day"]), int(row["hour"])) in pairs if pd.notna(row["day"]) and pd.notna(row["hour"]) else False,
                axis=1,
            )
            result = result[mask]

    return result


active_cross_filters = _collect_cross_filters(base_filtered_df)

for filter_key, disabled_signature in list(st.session_state.cross_filter_disabled.items()):
    if _current_selection_signature(filter_key) != disabled_signature:
        st.session_state.cross_filter_disabled.pop(filter_key, None)

# Tạo filtered_df chứa TẤT CẢ các filter để hiển thị cho phần text, tính toán metric (Tab 1)
filtered_df = _apply_cross_filters(base_filtered_df, active_cross_filters)

st.sidebar.success(f"Đang hiển thị {len(filtered_df)} video sau khi lọc.")

with st.sidebar.expander("Cross-filters đang áp dụng", expanded=True):
    if active_cross_filters:
        for item in active_cross_filters:
            cols = st.columns([0.75, 0.25])
            cols[0].markdown(f"**{item['label']}**")
            if cols[1].button("Bỏ", key=f"remove_{item['key']}"):
                st.session_state.cross_filter_disabled[item["key"]] = item["signature"]
                st.rerun()

        if st.button("Bỏ tất cả cross-filters", key="remove_all_cross_filters"):
            for item in active_cross_filters:
                st.session_state.cross_filter_disabled[item["key"]] = item["signature"]
            st.rerun()
    else:
        st.caption("Chưa có cross-filter nào. Hãy chọn điểm/vùng trên các biểu đồ hỗ trợ tương tác.")

if filtered_df.empty:
    st.warning("Không còn dữ liệu sau khi áp dụng các bộ lọc hiện tại. Hãy bỏ bớt cross-filter ở sidebar.")
    st.stop()

# ================= TABS =================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Tổng quan",
    "Định nghĩa thành công",
    "Khẩu vị khán giả & Từ khóa",
    "Điểm ngọt thời gian phát hành",
    "Mô hình hóa",
    "Yếu tố gây nhiễu",
])

# ================= TAB 1 =================
with tab1:
    st.subheader("Tổng quan")

    col1,col2,col3,col4,col5,col6 = st.columns(6)

    col1.metric("Số video", len(filtered_df))
    col2.metric("Lượt xem trung bình", f"{filtered_df['video_view_count'].mean():,.0f}")
    col3.metric("Lượt xem trung vị", f"{filtered_df['video_view_count'].median():,.0f}")
    col4.metric("Lượt xem cao nhất", f"{filtered_df['video_view_count'].max():,.0f}")
    col5.metric("Số thẻ trung bình", f"{filtered_df['video_tags_count'].mean():.1f}")
    col6.metric("Số người đăng ký trung bình", f"{filtered_df['channel_subscriber_count'].mean():,.0f}")

    st.markdown("### Phân bố lượt xem")

    fig = px.histogram(filtered_df, x='video_view_count', nbins=60)
    st.plotly_chart(fig, use_container_width=True)

# ================= TAB 2 =================
with tab2:
    st.header("Câu 1: Định nghĩa 'Thành công' của một sản phẩm âm nhạc")

    # Tiền xử lý dữ liệu cơ bản cho Câu 1
    df_bubble = _apply_cross_filters(base_filtered_df, active_cross_filters, exclude_key="cross_bubble")
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
        log_x=True, # Dùng log scale vì view chênh lệch lớn
        log_y=True,
        size_max=60,
        color_discrete_map={True: '#1DB954', False: '#E50914'}, # Xanh Spotify, Đỏ Netflix
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
    _sync_chart_selection("cross_bubble", bubble_event)
    st.caption("💡 **Insight:** Những bong bóng to nhất biểu thị video gây bão bình luận. So sánh màu sắc để xem nhạc có bản quyền hay nhạc tự do đem lại nhiều tương tác hơn.")


    # --- BIỂU ĐỒ 1.2: DENSITY HEATMAP ---
    st.subheader("2. Ma trận Thành công: Mật độ phân bổ Video")
    
    # Tính tỷ lệ và LỌC BỎ các giá trị <= 0 để tránh lỗi Logarit
    df_q1['engagement_rate'] = ((df_q1['video_like_count'] + df_q1['video_comment_count']) / df_q1['video_view_count']) * 100
    df_q1_clean = df_q1[(df_q1['engagement_rate'] > 0) & (df_q1['video_view_count'] > 0) & (df_q1['engagement_rate'] < 100)]
    
    df_q1_clean['log_view'] = np.log10(df_q1_clean['video_view_count'])
    df_q1_clean['log_er'] = np.log10(df_q1_clean['engagement_rate'])
    
    fig2 = px.density_heatmap(
        df_q1_clean,
        x='log_view',
        y='log_er',
        # log_x=True,
        # log_y=True,
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
    df_lollipop = _apply_cross_filters(base_filtered_df, active_cross_filters, exclude_key="cross_lollipop")
    df_q1_lollipop = df_lollipop.dropna(subset=['video_view_count', 'video_like_count', 'video_comment_count']).copy()
    
    # Tính toán top 10 kênh có tổng view cao nhất
    top_10_channels = df_q1_lollipop.groupby('channel_title')['video_view_count'].sum().nlargest(10).index
    df_top10 = df_q1_lollipop[df_q1_lollipop['channel_title'].isin(top_10_channels)]

    # Group by tính tổng view, like, comment cho top 10
    df_grouped = df_top10.groupby('channel_title')[['video_view_count', 'video_like_count', 'video_comment_count']].sum().reset_index()
    # Tính tỷ lệ tương tác
    df_grouped['engagement_rate'] = (df_grouped['video_like_count'] + df_grouped['video_comment_count']) / df_grouped['video_view_count']
    df_grouped = df_grouped.sort_values(by='engagement_rate', ascending=True)

    fig3 = go.Figure()
    # Vẽ các đường thẳng (que kẹo)
    for i, row in df_grouped.iterrows():
        fig3.add_shape(
            type="line",
            x0=0, x1=row['engagement_rate'],
            y0=row['channel_title'], y1=row['channel_title'],
            line=dict(color="#888888", width=2)
        )
    # Vẽ các điểm tròn (viên kẹo)
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
    _sync_chart_selection("cross_lollipop", lollipop_event)
    st.caption("💡 **Insight:** Kênh có thanh kẹo dài nhất là kênh tuy có thể ít View, nhưng sở hữu lượng fan 'cày' like và comment nhiệt tình và trung thành nhất.")

# ================= TAB 3 =================
with tab3:
    st.header("Câu 2: Khẩu vị khán giả & Sức mạnh của Từ khóa (NLP)")

    # Tiền xử lý cho Q2
    df_q2 = filtered_df.dropna(subset=['video_view_count']).copy()
    df_q2['video_made_for_kids'] = df_q2['video_made_for_kids'].fillna(False).astype(bool)
    df_q2['video_licensed_content'] = df_q2['video_licensed_content'].fillna(False).astype(bool)

    # --- BIỂU ĐỒ 2.1: TREEMAP TRỰC QUAN HÓA THỊ PHẦN ---
    st.subheader("1. Cơ cấu lượt xem: Đối tượng, Bản quyền và Chất lượng")

    # Đổi True/False thành nhãn Tiếng Việt dễ hiểu ngay lập tức
    df_q2['Doi_tuong'] = df_q2['video_made_for_kids'].map({True: 'Nhạc Trẻ Em', False: 'Nhạc Đại Chúng'})
    df_q2['Ban_quyen'] = df_q2['video_licensed_content'].map({True: 'Chính thức (Official)', False: 'Tự do (Cover/Remix)'})
    df_q2['Chat_luong'] = df_q2['video_definition'].astype(str).str.upper()

    # Tạo cột tổng hợp đường dẫn cho Treemap
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
    # Đưa cột caption về dạng boolean
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
    df_bar = _apply_cross_filters(base_filtered_df, active_cross_filters, exclude_key="cross_keywords")
    df_q2_bar = df_bar.dropna(subset=['video_view_count', 'video_title']).copy()
    keyword_stats = []

    # Chuẩn bị dữ liệu text
    titles = df_q2_bar['video_title'].dropna().astype(str).tolist()

    # Định nghĩa các từ dừng (stopwords) tiếng Việt và Anh để thuật toán bỏ qua
    custom_stopwords = ['và', 'của', 'là', 'những', 'các', 'trong', 'với', 'cho', 'để', 'có', 'không', 'bài', 'hát', 'tập', 'phần', 'the', 'of', 'in', 'and', 'to', 'a', 'is', 'that', 'it', 'on', 'for', 'as', 'was', 'but', 'are']

    # Chạy thuật toán TF-IDF (lấy top 15 cụm từ quan trọng nhất, có thể là 1 hoặc 2 từ ghép lại)
    try:
        vectorizer = TfidfVectorizer(max_features=15, ngram_range=(1, 2), stop_words=custom_stopwords)
        vectorizer.fit(titles)
        top_keywords = vectorizer.get_feature_names_out()

        keyword_stats = []
        for kw in top_keywords:
            # Lọc các video chứa từ khóa này
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
            _sync_chart_selection("cross_keywords", keyword_event)
            st.caption("💡 **Insight:** Biểu đồ này không dùng danh sách cho trước mà dùng AI (TF-IDF) tự động dò quét toàn bộ text. Những cụm từ có cột mọc dài nhất chính là 'công thức đặt tên' thu hút view khủng nhất trên thị trường.")
        else:
            st.warning("Không trích xuất được từ khóa nào thỏa mãn.")

    except Exception as e:
        st.error(f"Lỗi khi chạy TF-IDF: {e}. Vui lòng kiểm tra lại dữ liệu chữ.")


# ================= TAB 4 =================
with tab4:
    st.header("Câu 3: Khám phá 'Khung giờ vàng' đăng nhạc")

    df_q3 = filtered_df.dropna(subset=['video_view_count', 'hour', 'day']).copy()

    # Sắp xếp thứ tự các ngày trong tuần cho chuẩn
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_q3['day'] = pd.Categorical(df_q3['day'], categories=days_order, ordered=True)

    # --- BIỂU ĐỒ 3.1: HEATMAP THỜI GIAN ĐĂNG BÀI ---
    st.subheader("1. Bản đồ nhiệt (Heatmap): Đăng lúc nào dễ có view cao nhất?")

    # Tính trung bình view theo ngày và giờ
    heat_data = df_q3.groupby(['day', 'hour'])['video_view_count'].mean().reset_index()
    heat_pivot = heat_data.pivot(index='day', columns='hour', values='video_view_count')
    heat_pivot = heat_pivot.reindex(index=days_order)
    heat_pivot = heat_pivot.reindex(columns=sorted(heat_pivot.columns))

    fig7 = px.imshow(
        heat_pivot,
        labels=dict(x="Giờ trong ngày (0-23h)", y="Ngày trong tuần", color="Trung bình View"),
        color_continuous_scale='Inferno',
        aspect="auto",
        title="Heatmap: Tương quan giữa Giờ đăng, Ngày đăng và Lượt xem"
    )
    st.plotly_chart(fig7, use_container_width=True)
    st.caption("💡 **Insight:** Những ô vuông đỏ/cam rực rỡ nhất chính là 'Khung giờ vàng'. Đây là thời điểm phát hành nhạc đạt hiệu suất lượt xem cao nhất trong tuần.")

    # --- BIỂU ĐỒ 3.2: LINE/RADAR CHART - CUNG VS CẦU ---
    st.subheader("2. Độ lệch pha: Youtuber thích đăng giờ nào vs Khán giả xem giờ nào?")
    df_line_radar = _apply_cross_filters(base_filtered_df, active_cross_filters, exclude_key="cross_trend")
    df_q3_line_radar = df_line_radar.dropna(subset=['video_view_count', 'hour']).copy()

    # Đếm số lượng video phát hành theo giờ (Nguồn Cung)
    supply = df_q3_line_radar.groupby('hour').size().reset_index(name='video_count')
    # Tổng view thu về theo giờ (Nguồn Cầu)
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
    _sync_chart_selection("cross_trend", trend_event)
    st.caption("💡 Đường màu đỏ chênh lệch cao tại những cột màu xanh thấp chính là 'Đại dương xanh' - ít người cạnh tranh đăng bài nhưng lượng người xem lại cực lớn.")

    # --- BIỂU ĐỒ 3.3: 2D DENSITY CONTOUR PLOT ---
    st.subheader("3. Biểu đồ Đường đồng mức: Sự kết hợp hoàn hảo giữa Thời lượng và Lượt xem")

    # Lọc bỏ các video có thời lượng = 0 để tránh lỗi Log scale
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


# ================= TAB 5 =================
with tab5:
    st.subheader("Hồi quy từng bước")

    features = [
        'video_tags_count',
        'channel_subscriber_count',
        'title_length',
        'channel_view_count',
        'channel_video_count'
    ]

    df_model = filtered_df.copy()
    df_model = df_model.replace([np.inf, -np.inf], np.nan)

    for col in features:
        df_model[col] = df_model[col].fillna(df_model[col].median())

    df_model = df_model.dropna(subset=['video_view_count'])
    if len(df_model) < 10:
        st.warning("⚠️ Dữ liệu hiện tại quá ít (ít hơn 10 video) để chạy mô hình Hồi quy đáng tin cậy. Các Cross-filter bạn đang chọn đã thu hẹp dữ liệu. Vui lòng nhấn nút 'Bỏ' bớt bộ lọc ở Sidebar để xem Tab này.")
    else:
        def stepwise(data, target, feats):
            selected = []
            log = []

            data = data.copy()
            # ép các cột ứng viên về dạng số và thay vô cực bằng giá trị khuyết
            for c in list(feats) + [target]:
                if c in data.columns:
                    data[c] = pd.to_numeric(data[c], errors='coerce')

            data = data.replace([np.inf, -np.inf], np.nan)

            while feats:
                scores = []
                for f in feats:
                    cols = selected + [f]
                    # chuẩn bị biến độc lập và phụ thuộc, rồi bỏ dòng thiếu dữ liệu
                    X = pd.DataFrame(sm.add_constant(data[cols], has_constant='add'), index=data.index)
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

            # mô hình cuối cùng: dùng các biến đã chọn, hoặc chỉ hằng số nếu không có biến nào
            if selected:
                X_final = pd.DataFrame(sm.add_constant(data[selected], has_constant='add'), index=data.index)
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
            st.write("### Biến được chọn")
            for l in log:
                st.code(l)

        with col2:
            st.write("### Tóm tắt mô hình")
            if model is None:
                st.info("Không đủ dữ liệu để ước lượng mô hình.")
            else:
                st.text(model.summary().tables[1])

# ================= TAB 6 =================
with tab6:
    st.subheader("Phân tích yếu tố gây nhiễu")

    st.info("Mục này được giữ ở dạng mô tả vì phần cross-filter của dashboard đang tập trung vào các plot có ý nghĩa rõ ràng hơn để tránh gây nhiễu cho luồng lọc chung.")

    st.markdown("""
    ### Nhận định:
    - Quy mô kênh có thể làm lệch kết luận về khung giờ đăng  
    - Thời điểm đăng chưa chắc là nguyên nhân thực sự  
    """)

# ================= CHATBOT =================
st.markdown("---")
st.subheader("💬 Hỏi đáp nhanh")

q = st.text_input("Hỏi điều bạn muốn biết...")

if q:
    if "viral" in q:
        pct = (filtered_df['video_view_count'] >= q).mean()*100
        st.write(f"🔥 Tỷ lệ viral: {pct:.2f}%")

    elif "time" in q:
        best = filtered_df.groupby('hour')['video_view_count'].mean().idxmax()
        st.write(f"Khung giờ đăng tốt nhất: {best}")

    else:
        st.write("Thử nhập: viral, time")