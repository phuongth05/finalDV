import json
import re

import numpy as np
import pandas as pd
import streamlit as st

from tabs.tab1_overview import render_tab as render_tab1
from tabs.tab2_success import render_tab as render_tab2
from tabs.tab3_audience import render_tab as render_tab3
from tabs.tab4_timing import render_tab as render_tab4
from tabs.tab5_modeling import render_tab as render_tab5
from tabs.tab6_confounders import render_tab as render_tab6

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
# Thêm vào app.py, ngay sau khi load df:
# Kiểm tra và tạo cột genre nếu chưa có
def extract_genre(query):
    genre_map = {
        'bolero': 'Bolero', 'rap': 'Rap',
        'indie': 'Indie', 'lofi': 'Lofi',
        'remix': 'Remix', 'trữ tình': 'Trữ tình',
        'trẻ': 'Nhạc trẻ', 'buồn': 'Nhạc buồn',
        'chill': 'Chill', 'thiếu nhi': 'Thiếu nhi',
        'vàng': 'Nhạc vàng', 'đỏ': 'Nhạc đỏ',
        'quê hương': 'Quê hương', 'dân ca': 'Dân ca',
        'acoustic': 'Acoustic', 'không lời': 'Không lời',
        'cải lương': 'Cải lương', 'chế': 'Nhạc chế',
        'sàn': 'Nhạc sàn', 'tết': 'Nhạc Tết',
        'đám cưới': 'Nhạc đám cưới',
        'vui tươi': 'Vui tươi', 'live': 'Live',
        'nhạc cụ': 'Nhạc cụ',
    }
    query_lower = str(query).lower()
    for key, value in genre_map.items():
        if key in query_lower:
            return value
    return 'Khác'

df['genre'] = df['search_query'].apply(extract_genre)
# basic features
df['title_length'] = df['video_title'].astype(str).apply(len)

def detect_video_type(title):
    title_lower = str(title).lower()
    if 'official' in title_lower or 'mv' in title_lower:
        return 'MV Chính thức'
    elif 'lyric' in title_lower:
        return 'Lyric Video'
    elif 'cover' in title_lower:
        return 'Cover'
    elif 'live' in title_lower:
        return 'Live'
    elif 'remix' in title_lower:
        return 'Remix'
    elif 'karaoke' in title_lower:
        return 'Karaoke'
    else:
        return 'Khác'

df['video_type'] = df['video_title'].apply(detect_video_type)

df['engagement_rate'] = (
    (pd.to_numeric(df['video_like_count'], errors='coerce')
     + pd.to_numeric(df['video_comment_count'], errors='coerce'))
    / pd.to_numeric(df['video_view_count'], errors='coerce')
).replace([np.inf, -np.inf], np.nan)

df['channel_size'] = pd.qcut(
    df['channel_subscriber_count'].rank(method='first'),
    q=3,
    labels=['Small', 'Medium', 'Large']
)

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
    "Chatbot AI",
])

# ================= TAB 1 =================
with tab1:
    render_tab1(filtered_df)

# ================= TAB 2 =================
with tab2:
    render_tab2(base_filtered_df, active_cross_filters, _apply_cross_filters, _sync_chart_selection)

# ================= TAB 3 =================
with tab3:
    render_tab3(filtered_df, base_filtered_df, active_cross_filters, _apply_cross_filters, _sync_chart_selection)


# ================= TAB 4 =================
with tab4:
    render_tab4(filtered_df, base_filtered_df, active_cross_filters, _apply_cross_filters, _sync_chart_selection)


# ================= TAB 5 =================
with tab5:
    render_tab5(filtered_df)

# ================= TAB 6 =================
with tab6:
    render_tab6(filtered_df)

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