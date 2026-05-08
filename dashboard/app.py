import json
import os
import re

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from tabs.tab1_overview import render_tab as render_tab1
from tabs.tab2_success import render_tab as render_tab2
from tabs.tab3_audience import render_tab as render_tab3
from tabs.tab4_timing import render_tab as render_tab4
from tabs.tab5_modeling import render_tab as render_tab5
from tabs.tab6_confounders import render_tab as render_tab6
from tabs.tab7_ai import render_tab as render_tab7
from tabs.tab5_modeling import _build_df_model

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

TAB_CHAT_PROMPT = """
Bạn là trợ lý phân tích dữ liệu cho dashboard nhạc YouTube Việt Nam.
Chỉ được dựa trên context JSON của tab hiện tại.
Nếu dữ liệu chưa đủ để trả lời, hãy nói rõ là chưa đủ dữ liệu thay vì đoán.
Trả lời ngắn gọn, đúng trọng tâm, có thể nêu insight hoặc gợi ý hành động.

Context JSON:
{context_json}
"""

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


def _numeric_series(frame, column):
    if column not in frame.columns:
        return pd.Series(dtype=float, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def _get_groq_client():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def _jsonify_context(payload):
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _persist_tab_context(tab_key, tab_context):
    context_dir = os.path.join("data", "ai_for_only_tab")
    os.makedirs(context_dir, exist_ok=True)
    file_path = os.path.join(context_dir, f"{tab_key}_context.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(tab_context, f, ensure_ascii=False, indent=2, default=str)
    return file_path


def _safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _shared_selection_context(frame):
    cache = st.session_state.get("cross_filter_cache", {})
    selection_context = {}

    bubble_points = cache.get("cross_bubble", []) or []
    if bubble_points and "_row_id" in frame.columns:
        row_ids = _points_to_row_ids(bubble_points)
        selected_frame = frame[frame["_row_id"].astype(str).isin(row_ids)]
        selection_context["bubble"] = {
            "selected_count": len(row_ids),
            "video_titles": selected_frame["video_title"].dropna().astype(str).head(10).tolist() if "video_title" in selected_frame.columns else [],
        }

    lollipop_points = cache.get("cross_lollipop", []) or []
    if lollipop_points:
        selection_context["lollipop_channels"] = _points_to_channels(lollipop_points)

    keyword_points = cache.get("cross_keywords", []) or []
    if keyword_points:
        selection_context["keywords"] = _points_to_keywords(keyword_points)

    trend_points = cache.get("cross_trend", []) or []
    if trend_points:
        selection_context["hours"] = _points_to_hours(trend_points)

    return selection_context


def _bubble_regression_hint(frame):
    bubble_df = frame.dropna(subset=["video_view_count", "video_like_count", "video_comment_count"]).copy()
    bubble_df = bubble_df[(bubble_df["video_view_count"] > 0) & (bubble_df["video_like_count"] > 0)]
    if len(bubble_df) < 3:
        return {"status": "insufficient_data"}

    x = np.log10(bubble_df["video_view_count"].astype(float))
    y_like = np.log10(bubble_df["video_like_count"].astype(float))
    y_comment = np.log10((bubble_df["video_comment_count"].astype(float) + 1).clip(lower=1))

    like_slope, _ = np.polyfit(x, y_like, 1)
    comment_slope, _ = np.polyfit(x, y_comment, 1)
    like_corr = np.corrcoef(x, y_like)[0, 1]
    comment_corr = np.corrcoef(x, y_comment)[0, 1]

    return {
        "status": "ok",
        "log_view_like_slope": round(float(like_slope), 4),
        "log_view_comment_slope": round(float(comment_slope), 4),
        "log_view_like_corr": round(float(like_corr), 4) if pd.notna(like_corr) else None,
        "log_view_comment_corr": round(float(comment_corr), 4) if pd.notna(comment_corr) else None,
        "direction_like": "tăng" if like_slope > 0 else "giảm",
        "direction_comment": "tăng" if comment_slope > 0 else "giảm",
        "selected_points": len(bubble_df),
    }


def _top_table_payload(frame, value_col, label_col, n=5):
    if value_col not in frame.columns or label_col not in frame.columns:
        return []
    top_df = (
        frame[[label_col, value_col]]
        .dropna()
        .groupby(label_col, as_index=False)[value_col]
        .sum()
        .sort_values(value_col, ascending=False)
        .head(n)
    )
    return [
        {
            label_col: str(row[label_col]),
            value_col: _safe_float(row[value_col]),
        }
        for _, row in top_df.iterrows()
    ]


def _best_time_context(frame):
    time_frame = frame.dropna(subset=["video_publish_date", "hour", "day", "video_view_count"]).copy()
    if time_frame.empty:
        return {"status": "insufficient_data"}

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat_data = time_frame.groupby(["day", "hour"], as_index=False)["video_view_count"].mean()
    if heat_data.empty:
        return {"status": "insufficient_data"}

    heat_data["day"] = pd.Categorical(heat_data["day"], categories=day_order, ordered=True)
    best_heat = heat_data.sort_values("video_view_count", ascending=False).head(5)
    best_day = (
        time_frame.groupby("day", as_index=False)["video_view_count"].mean()
        .sort_values("video_view_count", ascending=False)
        .head(3)
    )
    best_hour = (
        time_frame.groupby("hour", as_index=False)["video_view_count"].mean()
        .sort_values("video_view_count", ascending=False)
        .head(3)
    )

    return {
        "status": "ok",
        "best_day_heatmap_cells": best_heat.to_dict(orient="records"),
        "best_days": best_day.to_dict(orient="records"),
        "best_hours": best_hour.to_dict(orient="records"),
    }


def _build_tab1_context(frame):
    views = _numeric_series(frame, "video_view_count")
    likes = _numeric_series(frame, "video_like_count")
    comments = _numeric_series(frame, "video_comment_count")
    tags = _numeric_series(frame, "video_tags_count")
    subscribers = _numeric_series(frame, "channel_subscriber_count")

    total_views = _safe_float(views.sum())
    total_likes = _safe_float(likes.sum())
    total_comments = _safe_float(comments.sum())
    engagement_rate = None
    if views.fillna(0).sum() > 0:
        engagement_rate = _safe_float(((likes.fillna(0) + comments.fillna(0)).sum() / views.fillna(0).sum()))

    top_channel = None
    if {"channel_title", "video_view_count"}.issubset(frame.columns):
        channel_stats = (
            frame.groupby("channel_title", as_index=False)["video_view_count"]
            .sum()
            .sort_values("video_view_count", ascending=False)
            .head(1)
        )
        if not channel_stats.empty:
            top_channel = channel_stats.to_dict(orient="records")[0]

    payload = {
        "tab": 1,
        "name": "Tổng quan",
        "rows": int(len(frame)),
        "active_filters": [item["label"] for item in active_cross_filters],
        "metrics": {
            "total_views": total_views,
            "total_likes": total_likes,
            "total_comments": total_comments,
            "mean_views": _safe_float(views.mean()),
            "median_views": _safe_float(views.median()),
            "max_views": _safe_float(views.max()),
            "mean_tags": _safe_float(tags.mean()),
            "mean_subscribers": _safe_float(subscribers.mean()),
            "engagement_rate": engagement_rate,
            "licensed_rate": _safe_float(frame["video_licensed_content"].fillna(False).astype(bool).mean()) if "video_licensed_content" in frame.columns else None,
            "caption_rate": _safe_float(frame["video_caption_status"].astype(str).str.lower().eq("true").mean()) if "video_caption_status" in frame.columns else None,
        },
        "top_channel_by_views": top_channel,
        "top_videos_by_views": _top_table_payload(frame, "video_view_count", "video_title", 5),
        "top_videos_by_likes": _top_table_payload(frame, "video_like_count", "video_title", 5),
        "bubble_relation_hint": _bubble_regression_hint(frame),
        "selected_chart_state": _shared_selection_context(frame),
    }
    return payload


def _build_tab2_context(frame):
    payload = {
        "tab": 2,
        "name": "Định nghĩa thành công",
        "rows": int(len(frame)),
        "active_filters": [item["label"] for item in active_cross_filters],
        "selected_chart_state": _shared_selection_context(frame),
    }

    if {"video_caption_status", "video_view_count"}.issubset(frame.columns):
        caption_group = (
            frame.assign(has_caption=frame["video_caption_status"].astype(str).str.lower().eq("true"))
            .groupby("has_caption", as_index=False)["video_view_count"].mean()
        )
        payload["caption_vs_no_caption"] = caption_group.to_dict(orient="records")

    if {"video_licensed_content", "video_view_count"}.issubset(frame.columns):
        license_group = (
            frame.assign(is_licensed=frame["video_licensed_content"].fillna(False).astype(bool))
            .groupby("is_licensed", as_index=False)["video_view_count"].mean()
        )
        payload["official_vs_free_avg_views"] = license_group.to_dict(orient="records")

    if "genre" in frame.columns and "video_view_count" in frame.columns:
        genre_group = (
            frame.groupby("genre", as_index=False)["video_view_count"]
            .sum()
            .sort_values("video_view_count", ascending=False)
            .head(6)
        )
        payload["top_genres_by_views"] = genre_group.to_dict(orient="records")

    payload["top_videos_by_views"] = _top_table_payload(frame, "video_view_count", "video_title", 5)
    payload["top_videos_by_likes"] = _top_table_payload(frame, "video_like_count", "video_title", 5)
    return payload


def _build_tab3_context(frame):
    payload = {
        "tab": 3,
        "name": "Hành vi người dùng",
        "rows": int(len(frame)),
        "time_context": _best_time_context(frame),
        "selected_chart_state": _shared_selection_context(frame),
    }

    if "video_duration" in frame.columns and "video_view_count" in frame.columns:
        duration_df = frame.dropna(subset=["video_duration", "video_view_count"]).copy()
        duration_df = duration_df[duration_df["video_duration"] > 0]
        if not duration_df.empty:
            duration_df["duration_min"] = duration_df["video_duration"] / 60
            duration_df["duration_group"] = pd.cut(
                duration_df["duration_min"],
                bins=[0, 5, 10, np.inf],
                labels=["< 5 phút", "5-10 phút", "> 10 phút"],
                right=False,
            )
            duration_group = (
                duration_df.groupby("duration_group", as_index=False)["video_view_count"]
                .agg(avg_views="mean", video_count="count")
            )
            payload["duration_summary"] = duration_group.to_dict(orient="records")

    payload["top_videos_by_views"] = _top_table_payload(frame, "video_view_count", "video_title", 5)
    return payload


def _build_tab4_context(frame):
    payload = {
        "tab": 4,
        "name": "Nền tảng & Thuật toán",
        "rows": int(len(frame)),
        "time_context": _best_time_context(frame),
        "selected_chart_state": _shared_selection_context(frame),
    }

    if {"video_duration", "video_view_count"}.issubset(frame.columns):
        duration_df = frame.dropna(subset=["video_duration", "video_view_count"]).copy()
        duration_df = duration_df[duration_df["video_duration"] > 0]
        if not duration_df.empty:
            duration_df["duration_min"] = duration_df["video_duration"] / 60
            payload["duration_vs_views_corr"] = _safe_float(duration_df[["duration_min", "video_view_count"]].corr().iloc[0, 1])
            payload["duration_top_groups"] = (
                duration_df.assign(duration_group=pd.cut(duration_df["duration_min"], bins=[0, 5, 10, np.inf], labels=["< 5 phút", "5-10 phút", "> 10 phút"], right=False))
                .groupby("duration_group", as_index=False)["video_view_count"].mean()
                .to_dict(orient="records")
            )

    if {"hour", "video_view_count"}.issubset(frame.columns):
        time_frame = frame.dropna(subset=["hour", "video_view_count"]).copy()
        supply = time_frame.groupby("hour").size().reset_index(name="video_count")
        demand = time_frame.groupby("hour", as_index=False)["video_view_count"].sum().rename(columns={"video_view_count": "total_views"})
        merged = pd.merge(supply, demand, on="hour", how="inner")
        if not merged.empty:
            merged["gap"] = merged["total_views"] - merged["video_count"]
            payload["supply_demand_hours"] = merged.to_dict(orient="records")
            payload["largest_gap_hour"] = merged.sort_values("gap", ascending=False).head(3).to_dict(orient="records")

    return payload


def _build_tab5_context(frame):
    payload = {
        "tab": 5,
        "name": "Mô hình hóa",
        "rows": int(len(frame)),
        "high_vif_vars": st.session_state.get("high_vif_vars", []),
        "ok_vif_vars": st.session_state.get("ok_vif_vars", []),
        "selected_chart_state": _shared_selection_context(frame),
    }

    try:
        df_model, genre_dummy_cols = _build_df_model(frame)
        corr_cols = [c for c in ["video_view_count", "video_like_count", "video_comment_count", "video_duration", "title_length", "description_length", "channel_subscriber_count", "channel_view_count", "channel_video_count", "hour", "engagement_rate"] if c in df_model.columns]
        if len(corr_cols) >= 2 and "video_view_count" in df_model.columns:
            corr_matrix = df_model[corr_cols].corr(numeric_only=True)
            corr_series = corr_matrix["video_view_count"].drop(labels=["video_view_count"], errors="ignore").dropna().sort_values(key=lambda s: s.abs(), ascending=False)
            payload["top_view_correlations"] = corr_series.head(8).round(4).to_dict()
        payload["genre_dummy_count"] = len(genre_dummy_cols)
        payload["numeric_feature_count"] = len([c for c in ["video_duration", "video_tags_count", "title_length", "description_length", "channel_subscriber_count", "channel_view_count", "channel_video_count", "hour"] if c in df_model.columns])
    except Exception as exc:
        payload["build_model_error"] = str(exc)

    return payload


def _ask_groq(user_question, tab_context, tab_label):
    client = _get_groq_client()
    if client is None:
        return None, "Chưa cấu hình GROQ_API_KEY trong môi trường."

    system_prompt = TAB_CHAT_PROMPT.format(context_json=_jsonify_context(tab_context))
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Tab hiện tại: {tab_label}. Câu hỏi: {user_question}"},
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.2,
    )
    return response.choices[0].message.content, None


def _render_tab_chatbot(tab_key, tab_label, context_builder, frame):
    st.markdown("---")
    with st.expander(f"🤖 Chatbot AI cho {tab_label}", expanded=False):
        st.caption("Chatbot này không giữ lịch sử hội thoại. Nó chỉ dùng context JSON của đúng tab hiện tại.")
        tab_context = context_builder(frame)
        st.session_state[f"{tab_key}_context"] = tab_context
        context_file = _persist_tab_context(tab_key, tab_context)
        st.caption(f"Context JSON: {context_file}")

        user_question = st.chat_input(f"Hỏi về {tab_label}...", key=f"{tab_key}_chat_input")
        if user_question:
            with st.spinner("AI đang đọc context của tab..."):
                answer, error = _ask_groq(user_question, tab_context, tab_label)
            if error:
                st.error(error)
            else:
                st.session_state[f"{tab_key}_last_question"] = user_question
                st.session_state[f"{tab_key}_last_answer"] = answer

        last_question = st.session_state.get(f"{tab_key}_last_question")
        last_answer = st.session_state.get(f"{tab_key}_last_answer")
        if last_question and last_answer:
            st.chat_message("user").write(last_question)
            st.chat_message("assistant").write(last_answer)
        else:
            st.info("Hãy nhập câu hỏi. Ví dụ: 'Vì sao bubble chart này tăng mạnh ở nhóm nào?' hoặc 'Khung giờ nào tốt nhất?'")

# ================= TABS =================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Tổng quan",
    "Định nghĩa thành công",
    "Hành vi người dùng",
    "Nền tảng & Thuật toán",
    "Mô hình hóa",
    "Yếu tố gây nhiễu",
    "AI Phân tích",
    "Chatbot AI"
    ])

# ================= TAB 1 =================
with tab1:
    render_tab1(filtered_df)
    _render_tab_chatbot("tab1", "Tab 1 - Tổng quan", _build_tab1_context, filtered_df)

# ================= TAB 2 =================
with tab2:
    render_tab2(base_filtered_df, active_cross_filters, _apply_cross_filters, _sync_chart_selection)
    _render_tab_chatbot("tab2", "Tab 2 - Định nghĩa thành công", _build_tab2_context, filtered_df)

# ================= TAB 3 =================
with tab3:
    render_tab3(filtered_df, base_filtered_df, active_cross_filters, _apply_cross_filters, _sync_chart_selection)
    _render_tab_chatbot("tab3", "Tab 3 - Hành vi người dùng", _build_tab3_context, filtered_df)


# ================= TAB 4 =================
with tab4:
    render_tab4(filtered_df, base_filtered_df, active_cross_filters, _apply_cross_filters, _sync_chart_selection)
    _render_tab_chatbot("tab4", "Tab 4 - Nền tảng & Thuật toán", _build_tab4_context, filtered_df)


# ================= TAB 5 =================
with tab5:
    render_tab5(filtered_df)
    _render_tab_chatbot("tab5", "Tab 5 - Mô hình hóa", _build_tab5_context, filtered_df)

# ================= TAB 6 =================
with tab6:
    render_tab6(filtered_df)

# ================= CHATBOT =================
with tab7:
    render_tab7(filtered_df)

