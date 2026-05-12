import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from tabs.tab5_modeling import BINARY_COLS, NUMERIC_COLS, _build_df_model, _stepwise_selection


TAB_CHAT_PROMPT = """
Bạn là trợ lý phân tích dữ liệu cho dashboard nhạc YouTube Việt Nam.
Chỉ được dựa trên context JSON của tab hiện tại.
Nếu dữ liệu chưa đủ để trả lời, hãy nói rõ là chưa đủ dữ liệu thay vì đoán.
Trả lời ngắn gọn, đúng trọng tâm, nhưng phải đủ 3 phần sau:
1. Nhận xét biểu đồ: nói rõ đang nói về biểu đồ nào, đọc đúng xu hướng/phân phối/tương quan/outlier/tỷ trọng nếu có.
2. Dẫn dắt câu chuyện: nếu có nhiều biểu đồ liên quan, hãy liên kết chúng thành một logic phân tích ngắn, nêu biểu đồ nào bổ sung hoặc xác nhận biểu đồ nào.
3. Kết luận cuối cùng: chốt lại một câu trả lời trực tiếp cho câu hỏi của người dùng, ưu tiên insight hoặc hành động cụ thể.

Không được chỉ liệt kê số liệu rời rạc. Nếu context có cả visualization và bảng dữ liệu, phải kết hợp cả hai khi reasoning.
Nếu một chart hoặc biến không có trong context, ghi rõ là không đủ dữ liệu cho phần đó.

**ĐẶC BIỆT CHO TAB 5 (MODELING):**
- Nếu context có "glossary" và "how_to_read", PHẢI dùng chúng để giải thích các thuật ngữ kỹ thuật.
- Khi nhắc đến VIF, removed_variables, confounding, p-value, R², hoặc Simpson's Paradox:
  → Dùng định nghĩa từ glossary để giải thích ý nghĩa
  → Trích dẫn how_to_read để hướng dẫn người dùng đọc mô hình từng bước
- Nếu phát hiện "flip_detected": True hoặc "flip_explanation", PHẢI nêu ra cảnh báo rõ ràng và giải thích lý do.
- Mục đích: giúp người dùng hiểu mô hình hóa từng bước, không chỉ đọc số liệu.

Context JSON:
{context_json}
"""


def _safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _numeric_series(frame, column):
    if column not in frame.columns:
        return pd.Series(dtype=float, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def _boolean_rate(frame, column):
    if column not in frame.columns:
        return None
    series = frame[column].astype(str).str.lower().str.strip()
    return _safe_float(series.isin(["true", "1", "yes"]).mean())


def _jsonify_context(payload):
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _persist_tab_context(tab_key, tab_context):
    context_dir = os.path.join("data", "ai_for_only_tab")
    os.makedirs(context_dir, exist_ok=True)
    file_path = os.path.join(context_dir, f"{tab_key}_context.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(tab_context, f, ensure_ascii=False, indent=2, default=str)
    return file_path


def _append_tab_history_log(tab_key, tab_label, user_question, answer, context_file, tab_context):
    log_dir = os.path.join("data", "ai_logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().isoformat(timespec="seconds")
    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d')}_{tab_key}.jsonl")

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "role": "user",
            "content": user_question,
            "timestamp": timestamp,
            "tab_key": tab_key,
            "tab_label": tab_label,
        }, ensure_ascii=False, default=str) + "\n")
        f.write(json.dumps({
            "role": "assistant",
            "content": answer,
            "timestamp": timestamp,
            "tab_key": tab_key,
            "tab_label": tab_label,
            "context_file": context_file,
            "context_rows": tab_context.get("rows"),
            "active_filters": tab_context.get("active_filters", []),
            "selected_chart_state": tab_context.get("selected_chart_state", {}),
        }, ensure_ascii=False, default=str) + "\n")

    return log_file


def _selection_state(widget_key):
    cached_selection = st.session_state.get("cross_filter_cache", {}).get(widget_key)
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


def _selection_points(selection_state):
    if selection_state is None:
        return []
    if isinstance(selection_state, dict):
        return selection_state.get("points", []) or []
    return getattr(selection_state, "points", []) or []


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


def _points_to_day_hour_pairs(points):
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


def _series_profile(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"status": "insufficient_data"}

    return {
        "status": "ok",
        "count": int(len(values)),
        "missing_count": int(series.isna().sum()) if hasattr(series, "isna") else None,
        "min": _safe_float(values.min()),
        "p25": _safe_float(values.quantile(0.25)),
        "median": _safe_float(values.median()),
        "p75": _safe_float(values.quantile(0.75)),
        "max": _safe_float(values.max()),
        "mean": _safe_float(values.mean()),
        "std": _safe_float(values.std()),
        "skewness": _safe_float(values.skew()),
        "kurtosis": _safe_float(values.kurtosis()),
        "zero_share": _safe_float((values == 0).mean()),
        "shape": _distribution_shape(values).get("shape"),
    }


def _monthly_aggregate(frame, date_col, value_col, agg="sum"):
    if date_col not in frame.columns or value_col not in frame.columns:
        return {"status": "insufficient_data"}

    time_frame = frame.dropna(subset=[date_col, value_col]).copy()
    if time_frame.empty:
        return {"status": "insufficient_data"}

    time_frame[date_col] = pd.to_datetime(time_frame[date_col], errors="coerce")
    time_frame = time_frame.dropna(subset=[date_col])
    if time_frame.empty:
        return {"status": "insufficient_data"}

    monthly = (
        time_frame.assign(period=time_frame[date_col].dt.to_period("M").dt.to_timestamp())
        .groupby("period", as_index=False)[value_col]
        .agg(agg)
        .sort_values("period")
    )
    if monthly.empty:
        return {"status": "insufficient_data"}

    monthly["mom_change_pct"] = monthly[value_col].pct_change() * 100
    monthly["mom_change_abs"] = monthly[value_col].diff().abs()
    return {
        "status": "ok",
        "rows": int(len(monthly)),
        "series": monthly.to_dict(orient="records"),
    }


def _top_group_table(frame, group_col, value_col, n=10, agg="sum", sort_desc=True):
    if group_col not in frame.columns or value_col not in frame.columns:
        return []

    grouped = (
        frame[[group_col, value_col]]
        .dropna()
        .groupby(group_col, as_index=False)[value_col]
        .agg(agg)
        .sort_values(value_col, ascending=not sort_desc)
        .head(n)
    )
    return grouped.to_dict(orient="records")


def _channel_engagement_table(frame, n=10):
    required = {"channel_title", "video_view_count", "video_like_count", "video_comment_count"}
    if not required.issubset(frame.columns):
        return {"status": "insufficient_data"}

    df = frame.dropna(subset=["channel_title", "video_view_count"]).copy()
    if df.empty:
        return {"status": "insufficient_data"}

    grouped = (
        df.groupby("channel_title", as_index=False)
        .agg(
            total_views=("video_view_count", "sum"),
            total_likes=("video_like_count", "sum"),
            total_comments=("video_comment_count", "sum"),
            video_count=("video_view_count", "size"),
        )
    )
    grouped["engagement_rate"] = (grouped["total_likes"].fillna(0) + grouped["total_comments"].fillna(0)) / grouped["total_views"].replace(0, np.nan)
    grouped = grouped.replace([np.inf, -np.inf], np.nan).dropna(subset=["engagement_rate"])
    grouped = grouped.sort_values(["engagement_rate", "total_views"], ascending=[False, False]).head(n)
    return {
        "status": "ok",
        "rows": int(len(grouped)),
        "table": grouped.to_dict(orient="records"),
    }


def _genre_distribution_table(frame, genre_col="genre", value_col="video_view_count"):
    if genre_col not in frame.columns or value_col not in frame.columns:
        return {"status": "insufficient_data"}

    df = frame.dropna(subset=[genre_col, value_col]).copy()
    if df.empty:
        return {"status": "insufficient_data"}

    grouped = (
        df.groupby(genre_col, as_index=False)
        .agg(
            video_count=(value_col, "size"),
            total_views=(value_col, "sum"),
            avg_views=(value_col, "mean"),
            median_views=(value_col, "median"),
            p25_views=(value_col, lambda x: x.quantile(0.25)),
            p75_views=(value_col, lambda x: x.quantile(0.75)),
        )
        .sort_values("total_views", ascending=False)
    )
    grouped["count_share"] = grouped["video_count"] / grouped["video_count"].sum()
    grouped["view_share"] = grouped["total_views"] / grouped["total_views"].sum()
    grouped["demand_gap"] = grouped["view_share"] - grouped["count_share"]
    return {
        "status": "ok",
        "rows": int(len(grouped)),
        "table": grouped.to_dict(orient="records"),
    }


def _time_heatmap_payload(frame):
    if {"day", "hour", "video_view_count"}.difference(frame.columns):
        return {"status": "insufficient_data"}

    df = frame.dropna(subset=["day", "hour", "video_view_count"]).copy()
    if df.empty:
        return {"status": "insufficient_data"}

    heat_data = df.groupby(["day", "hour"], as_index=False)["video_view_count"].mean()
    heat_data["video_count"] = df.groupby(["day", "hour"]).size().values
    heat_data = heat_data.sort_values("video_view_count", ascending=False)
    top_cells = heat_data.head(8)
    return {
        "status": "ok",
        "rows": int(len(top_cells)),
        "table": top_cells.to_dict(orient="records"),
        "top_cells": top_cells.to_dict(orient="records"),
    }


def _duration_group_payload(frame, bins, labels):
    if "video_duration" not in frame.columns or "video_view_count" not in frame.columns:
        return {"status": "insufficient_data"}

    df = frame.dropna(subset=["video_duration", "video_view_count"]).copy()
    df = df[df["video_duration"] > 0]
    if df.empty:
        return {"status": "insufficient_data"}

    df["duration_min"] = df["video_duration"] / 60
    df["duration_group"] = pd.cut(df["duration_min"], bins=bins, labels=labels, right=False)
    grouped = (
        df.groupby("duration_group", as_index=False)
        .agg(
            video_count=("video_view_count", "size"),
            avg_views=("video_view_count", "mean"),
            median_views=("video_view_count", "median"),
        )
        .sort_values("avg_views", ascending=False)
    )
    return {
        "status": "ok",
        "rows": int(len(grouped)),
        "table": grouped.to_dict(orient="records"),
    }


def _top_duration_videos(frame, n=20):
    if "video_duration" not in frame.columns:
        return {"status": "insufficient_data"}

    df = frame.dropna(subset=["video_duration"]).copy()
    df = df[df["video_duration"] > 0]
    if df.empty:
        return {"status": "insufficient_data"}

    columns = [c for c in ["video_title", "channel_title", "video_duration", "video_view_count", "genre", "video_publish_date"] if c in df.columns]
    outliers = df.sort_values("video_duration", ascending=False).head(n)[columns].copy()
    if "video_duration" in outliers.columns:
        outliers["duration_min"] = outliers["video_duration"] / 60
    return {
        "status": "ok",
        "rows": int(len(outliers)),
        "table": outliers.to_dict(orient="records"),
    }


def _correlation_payload(frame, cols):
    existing = [c for c in cols if c in frame.columns]
    if len(existing) < 2:
        return {"status": "insufficient_data"}

    corr_df = frame[existing].apply(pd.to_numeric, errors="coerce").corr(numeric_only=True)
    pairs = []
    for i in range(len(existing)):
        for j in range(i + 1, len(existing)):
            corr_val = corr_df.iloc[i, j]
            if pd.notna(corr_val):
                pairs.append({"pair": [existing[i], existing[j]], "corr": _safe_float(corr_val)})

    sorted_pairs = sorted(pairs, key=lambda item: abs(item["corr"]), reverse=True)
    return {
        "status": "ok",
        "matrix": corr_df.round(4).fillna(None).to_dict(),
        "pairs": sorted_pairs,
        "strongest_positive": next((item for item in sorted_pairs if item["corr"] is not None and item["corr"] > 0), None),
        "strongest_negative": next((item for item in sorted_pairs if item["corr"] is not None and item["corr"] < 0), None),
    }


def _heatmap_bins_payload(frame, x_col, y_col, value_col, bins=30):
    if {x_col, y_col, value_col}.difference(frame.columns):
        return {"status": "insufficient_data"}

    df = frame.dropna(subset=[x_col, y_col, value_col]).copy()
    if df.empty:
        return {"status": "insufficient_data"}

    df = df[(pd.to_numeric(df[x_col], errors="coerce") > 0) & (pd.to_numeric(df[y_col], errors="coerce") > 0)]
    if df.empty:
        return {"status": "insufficient_data"}

    x_log = np.log10(pd.to_numeric(df[x_col], errors="coerce"))
    y_log = np.log10(pd.to_numeric(df[y_col], errors="coerce"))
    df = df.assign(x_log=x_log, y_log=y_log).dropna(subset=["x_log", "y_log"])
    if df.empty:
        return {"status": "insufficient_data"}

    x_bins = pd.cut(df["x_log"], bins=bins, include_lowest=True)
    y_bins = pd.cut(df["y_log"], bins=bins, include_lowest=True)
    heat = df.groupby([x_bins, y_bins]).size().reset_index(name="count")
    heat = heat.sort_values("count", ascending=False)
    return {
        "status": "ok",
        "rows": int(len(heat)),
        "top_cells": heat.head(10).to_dict(orient="records"),
        "peak_cell": heat.head(1).to_dict(orient="records")[0] if not heat.empty else None,
    }


def _top_keyword_payload(frame, n=10):
    if "search_query" not in frame.columns or "video_view_count" not in frame.columns:
        return {"status": "insufficient_data"}

    keyword_df = frame.dropna(subset=["search_query", "video_view_count"]).copy()
    if keyword_df.empty:
        return {"status": "insufficient_data"}

    token_pattern = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
    stop_tokens = {"nhac", "music", "official", "video", "mv", "song", "cover", "remix", "lyrics", "lyric", "playlist", "audio"}
    rows = []
    for _, row in keyword_df.iterrows():
        text = f"{row.get('search_query', '')} {row.get('video_title', '')}"
        tokens = [token.lower() for token in token_pattern.findall(str(text)) if len(token) > 2]
        tokens = [token for token in tokens if token not in stop_tokens]
        for token in tokens[:12]:
            rows.append({"token": token, "views": row["video_view_count"]})

    if not rows:
        return {"status": "insufficient_data"}

    token_df = pd.DataFrame(rows)
    keyword_stats = token_df.groupby("token", as_index=False).agg(occurrences=("views", "size"), avg_views=("views", "mean"), total_views=("views", "sum")).sort_values(["avg_views", "occurrences"], ascending=[False, False])
    return {
        "status": "ok",
        "table": keyword_stats.head(n).to_dict(orient="records"),
    }


def _active_filter_labels(active_cross_filters):
    return [item["label"] for item in active_cross_filters] if active_cross_filters else []


def _distribution_shape(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"status": "insufficient_data"}

    skewness = _safe_float(values.skew())
    kurtosis = _safe_float(values.kurtosis())
    if skewness is None:
        label = "unknown"
    elif skewness > 1:
        label = "right_skewed"
    elif skewness < -1:
        label = "left_skewed"
    else:
        label = "balanced"

    return {
        "status": "ok",
        "shape": label,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "p25": _safe_float(values.quantile(0.25)),
        "median": _safe_float(values.median()),
        "p75": _safe_float(values.quantile(0.75)),
    }


def _outlier_summary(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"status": "insufficient_data"}

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = values[(values < lower) | (values > upper)]

    return {
        "status": "ok",
        "exists": bool(len(outliers) > 0),
        "outlier_count": int(len(outliers)),
        "outlier_share": _safe_float(len(outliers) / len(values)),
        "lower_bound": _safe_float(lower),
        "upper_bound": _safe_float(upper),
    }


def _trend_summary(frame, date_col="video_publish_date", value_col="video_view_count"):
    if date_col not in frame.columns or value_col not in frame.columns:
        return {"status": "insufficient_data"}

    time_frame = frame.dropna(subset=[date_col, value_col]).copy()
    if time_frame.empty:
        return {"status": "insufficient_data"}

    time_frame[date_col] = pd.to_datetime(time_frame[date_col], errors="coerce")
    time_frame = time_frame.dropna(subset=[date_col])
    if time_frame.empty:
        return {"status": "insufficient_data"}

    monthly = (
        time_frame.assign(period=time_frame[date_col].dt.to_period("M").dt.to_timestamp())
        .groupby("period", as_index=False)[value_col]
        .sum()
        .sort_values("period")
    )
    if monthly.empty:
        return {"status": "insufficient_data"}

    monthly["mom_change_pct"] = monthly[value_col].pct_change() * 100
    monthly["mom_change_abs"] = monthly[value_col].diff().abs()

    peak_periods = monthly.sort_values(value_col, ascending=False).head(3)
    most_volatile = monthly.sort_values("mom_change_abs", ascending=False).head(3)

    return {
        "status": "ok",
        "peak_periods": peak_periods.to_dict(orient="records"),
        "fluctuation": {
            "avg_monthly_change_pct": _safe_float(monthly["mom_change_pct"].abs().mean()),
            "max_monthly_change_pct": _safe_float(monthly["mom_change_pct"].abs().max()),
            "most_volatile_months": most_volatile.to_dict(orient="records"),
        },
        "series_length": int(len(monthly)),
    }


def _top_table_payload(frame, value_col, label_col, n=5, ascending=False):
    if value_col not in frame.columns or label_col not in frame.columns or n < 1:
        return []
    top_df = (
        frame[[label_col, value_col]]
        .dropna()
        .groupby(label_col, as_index=False)[value_col]
        .sum()
        .sort_values(value_col, ascending=ascending)
        .head(n)
    )
    return [
        {label_col: str(row[label_col]), value_col: _safe_float(row[value_col])}
        for _, row in top_df.iterrows()
    ]


def _pairwise_correlations(frame, cols):
    existing = [c for c in cols if c in frame.columns]
    if len(existing) < 2:
        return []
    corr_df = frame[existing].apply(pd.to_numeric, errors="coerce").corr(numeric_only=True)
    rows = []
    for i in range(len(existing)):
        for j in range(i + 1, len(existing)):
            corr_val = corr_df.iloc[i, j]
            if pd.notna(corr_val):
                rows.append({"pair": [existing[i], existing[j]], "corr": _safe_float(corr_val)})
    return sorted(rows, key=lambda item: abs(item["corr"]), reverse=True)


def _density_hotspot(frame):
    if {"video_view_count", "video_like_count", "video_comment_count"}.difference(frame.columns):
        return {"status": "insufficient_data"}

    df_heat = frame.dropna(subset=["video_view_count", "video_like_count", "video_comment_count"]).copy()
    df_heat = df_heat[(df_heat["video_view_count"] > 0) & (df_heat["video_like_count"] >= 0) & (df_heat["video_comment_count"] >= 0)]
    if len(df_heat) < 5:
        return {"status": "insufficient_data"}

    df_heat["engagement_rate"] = (df_heat["video_like_count"] + df_heat["video_comment_count"]) / df_heat["video_view_count"].replace(0, np.nan)
    df_heat = df_heat.replace([np.inf, -np.inf], np.nan).dropna(subset=["engagement_rate"])
    if df_heat.empty:
        return {"status": "insufficient_data"}

    df_heat["log_views"] = np.log10(df_heat["video_view_count"])
    df_heat["log_engagement"] = np.log10(df_heat["engagement_rate"].clip(lower=1e-9))

    x_bins = pd.cut(df_heat["log_views"], bins=30, include_lowest=True)
    y_bins = pd.cut(df_heat["log_engagement"], bins=30, include_lowest=True)
    heat = df_heat.groupby([x_bins, y_bins]).size().reset_index(name="count")
    if heat.empty:
        return {"status": "insufficient_data"}

    hotspot = heat.sort_values("count", ascending=False).head(1).iloc[0]
    success_zone = df_heat[
        (df_heat["video_view_count"] >= df_heat["video_view_count"].quantile(0.75))
        & (df_heat["engagement_rate"] >= df_heat["engagement_rate"].quantile(0.75))
    ]

    return {
        "status": "ok",
        "hotspot_region": {
            "log_views_bin": str(hotspot["log_views"]),
            "log_engagement_bin": str(hotspot["log_engagement"]),
            "count": int(hotspot["count"]),
        },
        "success_zone": {
            "count": int(len(success_zone)),
            "share": _safe_float(len(success_zone) / len(df_heat)),
            "views_threshold": _safe_float(df_heat["video_view_count"].quantile(0.75)),
            "engagement_threshold": _safe_float(df_heat["engagement_rate"].quantile(0.75)),
        },
    }


def _compare_groups(frame):
    if "video_view_count" not in frame.columns:
        return {"status": "insufficient_data"}

    compare_source = frame.dropna(subset=["video_view_count"]).copy()
    if compare_source.empty:
        return {"status": "insufficient_data"}

    q_low = compare_source["video_view_count"].quantile(0.2)
    q_high = compare_source["video_view_count"].quantile(0.8)
    bottom_df = compare_source[compare_source["video_view_count"] <= q_low]
    top_df = compare_source[compare_source["video_view_count"] >= q_high]

    feature_specs = [
        ("video_like_count", "likes"),
        ("video_comment_count", "comments"),
        ("video_tags_count", "tags"),
        ("channel_subscriber_count", "channel_subscribers"),
        ("channel_view_count", "channel_views"),
        ("channel_video_count", "channel_videos"),
        ("title_length", "title_length"),
        ("video_duration", "video_duration"),
        ("engagement_rate", "engagement_rate"),
    ]

    rows = []
    for col, label in feature_specs:
        if col not in compare_source.columns:
            continue
        top_val = pd.to_numeric(top_df[col], errors="coerce").median()
        bottom_val = pd.to_numeric(bottom_df[col], errors="coerce").median()
        if pd.isna(top_val) and pd.isna(bottom_val):
            continue
        rows.append({"feature": label, "top": _safe_float(top_val), "bottom": _safe_float(bottom_val), "diff": _safe_float(top_val - bottom_val if pd.notna(top_val) and pd.notna(bottom_val) else np.nan)})

    if not rows:
        return {"status": "insufficient_data"}

    compare_df = pd.DataFrame(rows)
    compare_df["abs_diff"] = compare_df["diff"].abs()
    strongest = compare_df.sort_values("abs_diff", ascending=False).head(5)
    weakest = compare_df.sort_values("abs_diff", ascending=True).head(5)

    return {
        "status": "ok",
        "top_20_vs_bottom_20": {
            "top_count": int(len(top_df)),
            "bottom_count": int(len(bottom_df)),
            "comparisons": compare_df.sort_values("abs_diff", ascending=False).to_dict(orient="records"),
        },
        "viral_vs_flop": {
            "viral_median_views": _safe_float(top_df["video_view_count"].median()),
            "flop_median_views": _safe_float(bottom_df["video_view_count"].median()),
            "viral_median_engagement": _safe_float(top_df["engagement_rate"].median()) if "engagement_rate" in top_df.columns else None,
            "flop_median_engagement": _safe_float(bottom_df["engagement_rate"].median()) if "engagement_rate" in bottom_df.columns else None,
        },
        "key_findings": {
            "strongest_differentiators": strongest.to_dict(orient="records"),
            "weak_differentiators": weakest.to_dict(orient="records"),
        },
    }


def _genre_supply_demand(frame):
    if "genre" not in frame.columns or "video_view_count" not in frame.columns:
        return {"status": "insufficient_data"}

    genre_df = frame.dropna(subset=["genre", "video_view_count"]).copy()
    if genre_df.empty:
        return {"status": "insufficient_data"}

    genre_stats = (
        genre_df.groupby("genre", as_index=False)
        .agg(video_count=("video_view_count", "size"), total_views=("video_view_count", "sum"), avg_views=("video_view_count", "mean"))
        .sort_values("avg_views", ascending=False)
    )

    genre_stats["count_share"] = genre_stats["video_count"] / genre_stats["video_count"].sum()
    genre_stats["view_share"] = genre_stats["total_views"] / genre_stats["total_views"].sum()
    genre_stats["demand_gap"] = genre_stats["view_share"] - genre_stats["count_share"]

    oversupplied = genre_stats.sort_values(["demand_gap", "video_count"], ascending=[True, False]).head(5)
    undersupplied = genre_stats.sort_values(["demand_gap", "avg_views"], ascending=[False, False]).head(5)
    opportunities = genre_stats[(genre_stats["count_share"] <= genre_stats["count_share"].median()) & (genre_stats["avg_views"] >= genre_stats["avg_views"].median())].sort_values("avg_views", ascending=False).head(5)

    return {
        "status": "ok",
        "oversupplied_genres": oversupplied.to_dict(orient="records"),
        "undersupplied_genres": undersupplied.to_dict(orient="records"),
        "market_opportunities": opportunities.to_dict(orient="records"),
    }


def _caption_impact(frame):
    if "video_view_count" not in frame.columns or "video_caption_status" not in frame.columns:
        return {"status": "insufficient_data"}

    caption_df = frame.dropna(subset=["video_view_count"]).copy()
    caption_df["has_caption"] = caption_df["video_caption_status"].astype(str).str.lower().str.strip().isin(["true", "1", "yes"])
    if caption_df.empty:
        return {"status": "insufficient_data"}

    grouped = caption_df.groupby("has_caption").agg(
        median_views=("video_view_count", "median"),
        median_likes=("video_like_count", "median") if "video_like_count" in caption_df.columns else ("video_view_count", "median"),
        median_comments=("video_comment_count", "median") if "video_comment_count" in caption_df.columns else ("video_view_count", "median"),
    ).reset_index()

    if "video_like_count" in caption_df.columns and "video_comment_count" in caption_df.columns:
        caption_df["engagement_rate"] = (pd.to_numeric(caption_df["video_like_count"], errors="coerce").fillna(0) + pd.to_numeric(caption_df["video_comment_count"], errors="coerce").fillna(0)) / pd.to_numeric(caption_df["video_view_count"], errors="coerce").replace(0, np.nan)
        grouped["median_engagement"] = caption_df.groupby("has_caption")["engagement_rate"].median().values

    return {"status": "ok", "median_performance_comparison": grouped.to_dict(orient="records")}


def _audience_segments(frame):
    if "video_view_count" not in frame.columns:
        return {"status": "insufficient_data"}

    segment_cols = [c for c in ["channel_size", "genre", "video_type"] if c in frame.columns]
    if not segment_cols:
        return {"status": "insufficient_data"}

    seg_df = frame.dropna(subset=[c for c in ["channel_size", "genre"] if c in frame.columns] + ["video_view_count"]).copy()
    if seg_df.empty:
        return {"status": "insufficient_data"}

    group_cols = [c for c in ["channel_size", "genre"] if c in seg_df.columns]
    if not group_cols:
        group_cols = [segment_cols[0]]

    grouped = (
        seg_df.groupby(group_cols, as_index=False)
        .agg(video_count=("video_view_count", "size"), avg_views=("video_view_count", "mean"), total_views=("video_view_count", "sum"))
        .sort_values("total_views", ascending=False)
    )
    dominant = grouped.head(5)
    niche = grouped.sort_values(["video_count", "avg_views"], ascending=[True, False]).head(5)

    return {
        "status": "ok",
        "dominant_segments": dominant.to_dict(orient="records"),
        "niche_segments": niche.to_dict(orient="records"),
    }


def _upload_trends(frame):
    if "video_publish_date" not in frame.columns or "genre" not in frame.columns:
        return {"status": "insufficient_data"}

    trend_df = frame.dropna(subset=["video_publish_date", "genre"]).copy()
    trend_df["video_publish_date"] = pd.to_datetime(trend_df["video_publish_date"], errors="coerce")
    trend_df = trend_df.dropna(subset=["video_publish_date"])
    if trend_df.empty:
        return {"status": "insufficient_data"}

    trend_df["month"] = trend_df["video_publish_date"].dt.to_period("M").dt.to_timestamp()
    monthly = trend_df.groupby("month").size().reset_index(name="video_count")
    monthly["growth"] = monthly["video_count"].diff()

    if monthly.empty:
        return {"status": "insufficient_data"}

    growth_period = monthly.sort_values("growth", ascending=False).head(3)

    genre_month = trend_df.groupby(["genre", "month"]).size().reset_index(name="video_count")
    stable_genres = genre_month.groupby("genre")["video_count"].agg(["mean", "std", "count"]).reset_index()
    stable_genres = stable_genres[stable_genres["count"] >= 3]
    if not stable_genres.empty:
        stable_genres["cv"] = stable_genres["std"] / stable_genres["mean"].replace(0, np.nan)
        stable_genres = stable_genres.sort_values(["cv", "mean"], ascending=[True, False]).head(5)

    return {
        "status": "ok",
        "growth_period": growth_period.to_dict(orient="records"),
        "consistent_genres": stable_genres.to_dict(orient="records") if not stable_genres.empty else [],
    }


def _timing_optimization(frame):
    if "hour" not in frame.columns or "day" not in frame.columns or "video_view_count" not in frame.columns:
        return {"status": "insufficient_data"}

    time_df = frame.dropna(subset=["hour", "day", "video_view_count"]).copy()
    if time_df.empty:
        return {"status": "insufficient_data"}

    best_day = time_df.groupby("day", as_index=False)["video_view_count"].mean().sort_values("video_view_count", ascending=False).head(3)
    best_hour = time_df.groupby("hour", as_index=False)["video_view_count"].mean().sort_values("video_view_count", ascending=False).head(3)
    peak_slots = time_df.groupby(["day", "hour"], as_index=False)["video_view_count"].mean().sort_values("video_view_count", ascending=False).head(5)

    return {"status": "ok", "best_day": best_day.to_dict(orient="records"), "best_hour": best_hour.to_dict(orient="records"), "peak_slots": peak_slots.to_dict(orient="records")}


def _duration_optimization(frame):
    if "video_duration" not in frame.columns or "video_view_count" not in frame.columns:
        return {"status": "insufficient_data"}

    duration_df = frame.dropna(subset=["video_duration", "video_view_count"]).copy()
    duration_df = duration_df[duration_df["video_duration"] > 0]
    if duration_df.empty:
        return {"status": "insufficient_data"}

    duration_df["duration_min"] = duration_df["video_duration"] / 60
    bins = [0, 5, 10, 15, np.inf]
    labels = ["< 5 phút", "5-10 phút", "10-15 phút", "> 15 phút"]
    duration_df["duration_group"] = pd.cut(duration_df["duration_min"], bins=bins, labels=labels, right=False)
    grouped = duration_df.groupby("duration_group", as_index=False)["video_view_count"].mean().sort_values("video_view_count", ascending=False)

    sweet_spot = grouped.head(1).to_dict(orient="records")
    low_perf = grouped.sort_values("video_view_count", ascending=True).head(1).to_dict(orient="records")

    return {"status": "ok", "sweet_spot_duration": sweet_spot, "low_performance_duration": low_perf, "all_groups": grouped.to_dict(orient="records")}


def _keyword_impacts(frame):
    if "search_query" not in frame.columns or "video_view_count" not in frame.columns:
        return {"status": "insufficient_data"}

    keyword_df = frame.dropna(subset=["search_query", "video_view_count"]).copy()
    if keyword_df.empty:
        return {"status": "insufficient_data"}

    token_pattern = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
    stop_tokens = {"nhac", "music", "official", "video", "mv", "song", "cover", "remix", "lyrics", "lyric", "playlist", "audio"}
    rows = []
    for _, row in keyword_df.iterrows():
        text = f"{row.get('search_query', '')} {row.get('video_title', '')}"
        tokens = [token.lower() for token in token_pattern.findall(str(text)) if len(token) > 2]
        tokens = [token for token in tokens if token not in stop_tokens]
        for token in tokens[:12]:
            rows.append({"token": token, "views": row["video_view_count"]})

    if not rows:
        return {"status": "insufficient_data"}

    token_df = pd.DataFrame(rows)
    keyword_stats = token_df.groupby("token", as_index=False).agg(occurrences=("views", "size"), avg_views=("views", "mean"), total_views=("views", "sum")).sort_values(["avg_views", "occurrences"], ascending=[False, False])

    emotional_words = ["buồn", "chill", "vui", "tết", "live", "love", "sad", "heart", "nỗi", "đám", "cưới", "quê", "tâm", "đau"]
    emotional_keywords = keyword_stats[keyword_stats["token"].isin(emotional_words)].head(10)

    return {"status": "ok", "high_impact_keywords": keyword_stats.head(10).to_dict(orient="records"), "emotional_keywords": emotional_keywords.to_dict(orient="records")}


def _metadata_effectiveness(frame):
    metrics = []
    if "video_view_count" not in frame.columns:
        return metrics

    if "video_caption_status" in frame.columns:
        tmp = frame.copy()
        tmp["has_caption"] = tmp["video_caption_status"].astype(str).str.lower().str.strip().isin(["true", "1", "yes"])
        metrics.append({"feature": "caption", "with_true": _safe_float(tmp[tmp["has_caption"]]["video_view_count"].median()), "with_false": _safe_float(tmp[~tmp["has_caption"]]["video_view_count"].median())})

    if "video_licensed_content" in frame.columns:
        tmp = frame.copy()
        tmp["is_licensed"] = tmp["video_licensed_content"].fillna(False).astype(bool)
        metrics.append({"feature": "licensed", "with_true": _safe_float(tmp[tmp["is_licensed"]]["video_view_count"].median()), "with_false": _safe_float(tmp[~tmp["is_licensed"]]["video_view_count"].median())})

    if "video_title" in frame.columns:
        title_df = frame.copy()
        title_df["is_official"] = title_df["video_title"].astype(str).str.lower().str.contains("official|chính thức", regex=True)
        title_df["is_remix"] = title_df["video_title"].astype(str).str.lower().str.contains("remix", regex=False)
        metrics.append({"feature": "official_title", "with_true": _safe_float(title_df[title_df["is_official"]]["video_view_count"].median()), "with_false": _safe_float(title_df[~title_df["is_official"]]["video_view_count"].median())})
        metrics.append({"feature": "remix_title", "with_true": _safe_float(title_df[title_df["is_remix"]]["video_view_count"].median()), "with_false": _safe_float(title_df[~title_df["is_remix"]]["video_view_count"].median())})

    return metrics


def _platform_behavior(frame):
    if "hour" not in frame.columns or "video_view_count" not in frame.columns:
        return {"status": "insufficient_data"}

    hour_df = frame.dropna(subset=["hour", "video_view_count"]).copy()
    if hour_df.empty:
        return {"status": "insufficient_data"}

    upload_by_hour = hour_df.groupby("hour").size().rename("upload_count")
    views_by_hour = hour_df.groupby("hour")["video_view_count"].sum().rename("total_views")
    merged = pd.concat([upload_by_hour, views_by_hour], axis=1).reset_index()
    merged["total_views"] = merged["total_views"].fillna(0)

    sync_corr = _safe_float(merged[["upload_count", "total_views"]].corr().iloc[0, 1]) if len(merged) >= 2 else None
    preferred_hours = merged.sort_values("total_views", ascending=False).head(3)
    synchronized_hours = merged.sort_values("upload_count", ascending=False).head(3)

    return {"status": "ok", "upload_view_synchronization": {"correlation": sync_corr, "preferred_hours": preferred_hours.to_dict(orient="records"), "most_active_upload_hours": synchronized_hours.to_dict(orient="records")}, "algorithmic_preference": {"best_hours": preferred_hours["hour"].tolist()}}


def _build_tab1_context(frame, active_cross_filters):
    views = _numeric_series(frame, "video_view_count")
    likes = _numeric_series(frame, "video_like_count")
    comments = _numeric_series(frame, "video_comment_count")
    durations = _numeric_series(frame, "video_duration") / 60

    total_views = _safe_float(views.sum())
    total_likes = _safe_float(likes.sum())
    total_comments = _safe_float(comments.sum())
    engagement_rate = None
    if views.fillna(0).sum() > 0:
        engagement_rate = _safe_float(((likes.fillna(0) + comments.fillna(0)).sum() / views.fillna(0).sum()))

    top_channel = None
    if {"channel_title", "video_view_count"}.issubset(frame.columns):
        channel_stats = frame.groupby("channel_title", as_index=False)["video_view_count"].sum().sort_values("video_view_count", ascending=False).head(1)
        if not channel_stats.empty:
            top_channel = channel_stats.to_dict(orient="records")[0]

    trend = _trend_summary(frame, "video_publish_date", "video_view_count")
    duration_profile = _series_profile(durations)
    top_channels = _top_table_payload(frame, "video_view_count", "channel_title", 10)
    top_videos = _top_table_payload(frame, "video_view_count", "video_title", 10)
    license_counts = []
    if "video_licensed_content" in frame.columns:
        license_series = frame["video_licensed_content"].fillna(False).astype(bool).map({True: "Chính thức (Official)", False: "Tự do (Cover/Remix)"})
        license_counts = license_series.value_counts().rename_axis("Loai").reset_index(name="SoLuong").to_dict(orient="records")
    monthly_views = _monthly_aggregate(frame, "video_publish_date", "video_view_count", agg="sum")
    duration_outliers = _top_duration_videos(frame, n=20)

    payload = {
        "tab": 1,
        "name": "Tổng quan",
        "rows": int(len(frame)),
        "dashboard_state": {
            "columns": list(frame.columns),
            "numeric_profile": {
                "views": _series_profile(views),
                "likes": _series_profile(likes),
                "comments": _series_profile(comments),
                "duration_minutes": duration_profile,
            },
            "selected_chart_state": _shared_selection_context(frame),
        },
        "active_filters": _active_filter_labels(active_cross_filters),
        "kpi": {
            "total_views": total_views,
            "total_likes": total_likes,
            "total_comments": total_comments,
            "mean_views": _safe_float(views.mean()),
            "median_views": _safe_float(views.median()),
            "max_views": _safe_float(views.max()),
            "engagement_rate": engagement_rate,
            "licensed_ratio": _safe_float(frame["video_licensed_content"].fillna(False).astype(bool).mean()) if "video_licensed_content" in frame.columns else None,
            "caption_ratio": _boolean_rate(frame, "video_caption_status"),
        },
        "charts": {
            "view_distribution": {
                "chart_type": "histogram",
                "source_column": "video_view_count",
                "profile": _series_profile(views),
                "insight": "Phân bố lượt xem thường lệch phải nếu phần lớn video có view thấp và chỉ một số ít video bứt phá.",
            },
            "channel_overview": {
                "chart_type": "bar",
                "source_table": top_channels,
                "insight": "Top kênh cho biết nơi tập trung phần lớn traffic và mức độ tập trung của thị trường.",
            },
            "copyright_share": {
                "chart_type": "pie",
                "source_table": license_counts,
                "insight": "Tỷ trọng bản quyền cho biết phần nào của dữ liệu thuộc nội dung chính thức so với các biến thể cover/remix.",
            },
            "time_trend": {
                "chart_type": "line",
                "source": monthly_views,
                "insight": "Xu hướng theo tháng giúp nhận diện giai đoạn tăng trưởng, chững lại hoặc biến động bất thường.",
            },
            "duration_distribution": {
                "chart_type": "histogram",
                "source_column": "video_duration",
                "profile": duration_profile,
                "insight": "Độ dài video thường tập trung ở một vài khoảng nhất định; lệch phải hàm ý có một nhóm video rất dài.",
            },
            "duration_outliers": {
                "chart_type": "top_20_bubble",
                "source_table": duration_outliers,
                "insight": "Danh sách video dài nhất giúp phát hiện outlier và kiểm tra các nội dung hòa tấu, live, hoặc clip tổng hợp.",
            },
        },
        "distribution": {
            "view_distribution_shape": _distribution_shape(views),
            "duration_distribution": _distribution_shape(durations),
            "outlier_existence": {
                "views": _outlier_summary(views),
                "duration_minutes": _outlier_summary(durations),
            },
        },
        "top_entities": {
            "top_channels": top_channels,
            "top_videos": top_videos,
        },
        "trend": {
            "peak_periods": trend.get("peak_periods", []),
            "fluctuation": trend.get("fluctuation", {}),
        },
        "top_channel_by_views": top_channel,
        "tab_insights": [
            "Đọc đồng thời phân phối, xu hướng và top entities để tránh chỉ nhìn vào KPI tổng.",
            "Outlier thời lượng dài cần được diễn giải cùng lượt xem và thể loại, không xem như lỗi dữ liệu ngay lập tức.",
        ],
    }
    return payload


def _build_tab2_context(frame, active_cross_filters):
    corr_payload = _correlation_payload(frame, ["video_view_count", "video_like_count", "video_comment_count", "engagement_rate"])
    heat_payload = _density_hotspot(frame)
    compare_payload = _compare_groups(frame)
    engagement_payload = _channel_engagement_table(frame, n=10)

    payload = {
        "tab": 2,
        "name": "Định nghĩa thành công",
        "rows": int(len(frame)),
        "dashboard_state": {
            "columns": list(frame.columns),
            "selected_chart_state": _shared_selection_context(frame),
        },
        "active_filters": _active_filter_labels(active_cross_filters),
        "correlation": {
            "matrix": corr_payload.get("matrix") if isinstance(corr_payload, dict) else None,
            "pairs": corr_payload.get("pairs", []) if isinstance(corr_payload, dict) else [],
            "views_vs_likes": next((item["corr"] for item in corr_payload.get("pairs", []) if set(item["pair"]) == {"video_view_count", "video_like_count"}), None) if isinstance(corr_payload, dict) else None,
            "views_vs_comments": next((item["corr"] for item in corr_payload.get("pairs", []) if set(item["pair"]) == {"video_view_count", "video_comment_count"}), None) if isinstance(corr_payload, dict) else None,
            "engagement_correlation": next((item["corr"] for item in corr_payload.get("pairs", []) if "engagement_rate" in item["pair"]), None) if isinstance(corr_payload, dict) else None,
            "insight": "Tương quan cần đọc cùng heatmap và bảng so sánh top/bottom để tránh kết luận chỉ từ r.",
        },
        "density_heatmap": heat_payload,
        "channel_engagement": {
            "chart_type": "lollipop",
            "source_table": engagement_payload.get("table", []) if isinstance(engagement_payload, dict) else [],
            "insight": "Top kênh theo tỷ lệ tương tác cho thấy chất lượng tương tác, không chỉ tổng view.",
        },
        "comparative_analysis": compare_payload,
        "key_findings": compare_payload.get("key_findings", {}) if isinstance(compare_payload, dict) else {},
        "tab_insights": [
            "Nếu heatmap tập trung ở một vùng sáng, đó là cụm dữ liệu phổ biến chứ không tự động là cụm hiệu quả nhất.",
            "So sánh nhóm view cao và thấp nên ưu tiên các biến có chênh lệch ổn định và có ý nghĩa thực tế.",
        ],
    }
    return payload


def _build_tab3_context(frame, active_cross_filters):
    genre_distribution = _genre_distribution_table(frame)
    top_liked = _top_table_payload(frame, "video_like_count", "video_title", 5)
    low_liked = _top_table_payload(frame, "video_like_count", "video_title", 5, ascending=True)
    upload_share = genre_distribution.get("table", []) if isinstance(genre_distribution, dict) else []
    upload_vs_view = []
    if upload_share:
        share_df = pd.DataFrame(upload_share)
        if {"genre", "count_share", "view_share"}.issubset(share_df.columns):
            upload_vs_view = share_df[["genre", "count_share", "view_share", "demand_gap"]].to_dict(orient="records")

    genre_trend_table = []
    if {"video_publish_date", "genre"}.issubset(frame.columns):
        trend_df = frame.dropna(subset=["video_publish_date", "genre"]).copy()
        if not trend_df.empty:
            top_genres = []
            if upload_share:
                genre_rank_df = pd.DataFrame(upload_share)
                if "genre" in genre_rank_df.columns:
                    top_genres = genre_rank_df["genre"].astype(str).head(4).tolist()
            if top_genres:
                trend_df = trend_df[trend_df["genre"].astype(str).isin(top_genres)]
            trend_df["quarter"] = pd.to_datetime(trend_df["video_publish_date"], errors="coerce").dt.to_period("Q").dt.to_timestamp()
            trend_df = trend_df.dropna(subset=["quarter"])
            if not trend_df.empty:
                monthly = (
                    trend_df.groupby(["quarter", "genre"], as_index=False)
                    .size()
                    .rename(columns={"size": "video_count"})
                    .sort_values("quarter")
                )
                genre_trend_table = monthly.head(12).to_dict(orient="records")

    payload = {
        "tab": 3,
        "name": "Hành vi người dùng & Từ khóa",
        "rows": int(len(frame)),
        "dashboard_state": {
            "columns": list(frame.columns),
            "selected_chart_state": _shared_selection_context(frame),
            "dislike_data_available": "video_dislike_count" in frame.columns,
        },
        "active_filters": _active_filter_labels(active_cross_filters),
        "charts": {
            "genre_boxplot": {
                "chart_type": "box",
                "source_table": genre_distribution,
                "insight": "Box plot cần đọc median, IQR và outlier để hiểu phân phối view theo thể loại.",
            },
            "top_liked_videos": {
                "chart_type": "bar/bubble",
                "source_table": top_liked,
                "insight": "Top video theo lượt thích cho biết nội dung nào được cộng đồng phản hồi tích cực nhất.",
            },
            "least_liked_videos_proxy": {
                "chart_type": "bar/bubble",
                "source_table": low_liked,
                "insight": "Tập dữ liệu hiện không có cột dislike, nên nhóm này chỉ là proxy từ video có lượt thích thấp nhất.",
            },
            "genre_share": {
                "chart_type": "pie",
                "source_table": genre_distribution,
                "insight": "Tỷ trọng thể loại giúp nhìn lệch dữ liệu và mức độ tập trung nội dung.",
            },
            "upload_vs_view_share": {
                "chart_type": "combo_bar_line",
                "source_table": upload_vs_view,
                "insight": "So sánh upload share với view share để phát hiện thể loại bị đăng quá nhiều nhưng không thu hút tương ứng, hoặc ngược lại.",
            },
            "genre_trend": {
                "chart_type": "stacked_area",
                "source_table": genre_trend_table,
                "insight": "Chỉ giữ các mốc tháng thưa để nhìn xu hướng thể loại mà không làm context quá dày.",
            },
        },
        "supply_vs_demand": _genre_supply_demand(frame),
        "tab_insights": [
            "Đọc box plot cùng pie chart và combo chart sẽ cho thấy cả phân phối, tỷ trọng và độ lệch giữa cung - cầu nội dung.",
        ],
    }
    return payload


def _build_tab4_context(frame, active_cross_filters):
    day_table = _top_group_table(frame, "day", "video_view_count", n=7, agg="mean", sort_desc=True)
    hour_table = _top_group_table(frame, "hour", "video_view_count", n=8, agg="mean", sort_desc=True)
    supply_demand = frame.dropna(subset=["hour", "video_view_count"]).copy() if {"hour", "video_view_count"}.issubset(frame.columns) else pd.DataFrame()
    if not supply_demand.empty:
        hourly_supply = supply_demand.groupby("hour").size().reset_index(name="video_count")
        hourly_demand = supply_demand.groupby("hour", as_index=False)["video_view_count"].sum().rename(columns={"video_view_count": "total_views"})
        hourly_combo = pd.merge(hourly_supply, hourly_demand, on="hour", how="outer").fillna(0)
        hourly_combo["imbalance"] = hourly_combo["total_views"] - hourly_combo["video_count"]
        hourly_combo_table = hourly_combo.sort_values("total_views", ascending=False).head(8).to_dict(orient="records")
    else:
        hourly_combo_table = []

    payload = {
        "tab": 4,
        "name": "Thuật toán & Tối ưu nền tảng",
        "rows": int(len(frame)),
        "dashboard_state": {
            "columns": list(frame.columns),
            "selected_chart_state": _shared_selection_context(frame),
        },
        "active_filters": _active_filter_labels(active_cross_filters),
        "charts": {
            "time_heatmap": {
                "chart_type": "heatmap",
                "source_table": _time_heatmap_payload(frame),
                "insight": "Heatmap giờ-ngày cho thấy vùng dày dữ liệu và các ô hiệu suất cao để tối ưu lịch đăng.",
            },
            "day_hour_performance": {
                "chart_type": "bar",
                "day_table": day_table,
                "hour_table": hour_table,
                "insight": "Đọc riêng theo ngày và theo giờ giúp tách hiệu quả thời điểm đăng theo hai chiều khác nhau.",
            },
            "upload_vs_views_by_hour": {
                "chart_type": "combo_bar_line",
                "source_table": hourly_combo_table,
                "insight": "So sánh số video đăng và tổng lượt xem theo giờ giúp nhìn độ dồn dữ liệu và hiệu quả từng khung giờ.",
            },
            "duration_groups": {
                "chart_type": "column",
                "source_table": _duration_group_payload(frame, bins=[0, 5, 10, 15, np.inf], labels=["< 5 phút", "5-10 phút", "10-15 phút", "> 15 phút"]),
                "insight": "Nhóm thời lượng nào có view trung bình cao nhất là ứng viên cho chiến lược sản xuất nội dung.",
            },
            "duration_views_map": {
                "chart_type": "density_contour",
                "source_table": _heatmap_bins_payload(frame, "video_duration", "video_view_count", "video_view_count"),
                "insight": "Độ dồn dữ liệu giữa thời lượng và lượt xem cho biết vùng 'điểm rơi' phổ biến và outlier khác thường.",
            },
        },
        "timing_optimization": _timing_optimization(frame),
        "duration_optimization": _duration_optimization(frame),
        "tab_insights": [
            "Heatmap, combo chart và contour map cần được đọc cùng nhau để hiểu cả hành vi upload lẫn hiệu suất.",
        ],
    }
    return payload


def _vif_summary(df_model):
    num_exist = [c for c in NUMERIC_COLS if c in df_model.columns]
    if len(num_exist) < 2:
        return []

    x_vif = df_model[num_exist].dropna()
    non_const = [c for c in num_exist if c in x_vif.columns and x_vif[c].std() > 1e-10]
    x_vif = x_vif[non_const]
    if len(non_const) < 2 or len(x_vif) < 3:
        return []

    rows = []
    for i, col in enumerate(non_const):
        try:
            vif_val = variance_inflation_factor(x_vif.values, i)
            rows.append({"feature": col, "vif": _safe_float(vif_val)})
        except Exception:
            rows.append({"feature": col, "vif": None})
    return rows


def _stepwise_context(df_model, genre_dummy_cols):
    target_var = "log_views"
    high_vif_list = []
    ok_vif_list = []

    vif_rows = _vif_summary(df_model)
    if vif_rows:
        high_vif_list = [row["feature"] for row in vif_rows if row["vif"] is not None and row["vif"] >= 10]
        ok_vif_list = [row["feature"] for row in vif_rows if row["vif"] is not None and row["vif"] < 5]

    candidate_cols = [c for c in NUMERIC_COLS if c in df_model.columns and c != target_var]
    candidate_cols += [c for c in BINARY_COLS if c in df_model.columns]
    candidate_cols += [c for c in genre_dummy_cols if c in df_model.columns]
    final_pool = [c for c in candidate_cols if c not in high_vif_list and c in df_model.columns]

    if len(final_pool) < 1:
        return {"status": "insufficient_data", "high_vif_vars": high_vif_list, "ok_vif_vars": ok_vif_list, "model_context": {"correlation": _correlation_payload(df_model, ["video_view_count", "video_like_count", "video_comment_count", "engagement_rate"]), "vif": vif_rows}}

    try:
        selected, model, hist_df = _stepwise_selection(df_model, target_var, final_pool, direction="both", sl_enter=0.05, sl_remove=0.10)
    except Exception as exc:
        return {"status": "error", "error": str(exc), "high_vif_vars": high_vif_list, "ok_vif_vars": ok_vif_list, "model_context": {"correlation": _correlation_payload(df_model, ["video_view_count", "video_like_count", "video_comment_count", "engagement_rate"]), "vif": vif_rows}}

    rejected = [c for c in final_pool if c not in selected]
    removed_variables = sorted(set(high_vif_list + rejected))

    factor_importance = []
    model_info = {}
    statistical_interpretation = {}

    if model is not None and selected:
        try:
            X_s = df_model[selected].copy()
            X_s = X_s.loc[:, ~X_s.columns.duplicated()]
            selected_unique = X_s.columns.tolist()

            for c in selected_unique:
                X_s[c] = pd.to_numeric(X_s[c], errors="coerce")

            y_s = pd.to_numeric(df_model[target_var], errors="coerce")
            valid_mask = X_s.notna().all(axis=1) & y_s.notna()
            X_s = X_s[valid_mask]
            y_s = y_s[valid_mask]

            if len(X_s) >= len(selected_unique) + 2:
                scaler = StandardScaler()
                X_scaled = pd.DataFrame(scaler.fit_transform(X_s), columns=selected_unique, index=X_s.index)
                m_std = sm.OLS(y_s, sm.add_constant(X_scaled, has_constant="add")).fit()
                params = m_std.params.iloc[1:]
                pvals = m_std.pvalues.iloc[1:]
                coef_df = pd.DataFrame({"feature": params.index.tolist(), "std_coef": params.values, "p_value": pvals.values})
                coef_df["abs_coef"] = coef_df["std_coef"].abs()
                coef_df = coef_df.sort_values("abs_coef", ascending=False)
                factor_importance = coef_df.head(8).to_dict(orient="records")
                positive_effects = coef_df[coef_df["std_coef"] > 0].head(5).to_dict(orient="records")
                negative_effects = coef_df[coef_df["std_coef"] < 0].head(5).to_dict(orient="records")
                model_info = {
                    "r2": _safe_float(m_std.rsquared),
                    "adjusted_r2": _safe_float(m_std.rsquared_adj),
                    "selected_features": selected,
                    "rejected_features": rejected,
                    "n_obs": int(len(X_s)),
                    "chart_data": coef_df.to_dict(orient="records"),
                }
                statistical_interpretation = {
                    "multicollinearity_explanation": "Một số biến numeric có VIF cao hoặc tương quan mạnh, nên hệ số dễ biến động khi thêm/bớt biến.",
                    "significance_explanation": f"Có {int((coef_df['p_value'] < 0.05).sum())} biến có p-value < 0.05 trong mô hình cuối.",
                    "positive_effects": positive_effects,
                    "negative_effects": negative_effects,
                }
        except Exception as exc:
            model_info = {"error": str(exc)}

    high_corr_pairs = []
    numeric_cols = [c for c in NUMERIC_COLS if c in df_model.columns]
    if len(numeric_cols) >= 2:
        corr_matrix = df_model[numeric_cols].corr(numeric_only=True)
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                r = corr_matrix.iloc[i, j]
                if pd.notna(r) and abs(r) > 0.7:
                    high_corr_pairs.append({"feature_1": numeric_cols[i], "feature_2": numeric_cols[j], "corr": _safe_float(r)})

    return {
        "status": "ok",
        "high_corr_pairs": high_corr_pairs,
        "vif_analysis": vif_rows,
        "removed_variables": removed_variables,
        "model_results": model_info,
        "factor_importance": factor_importance,
        "statistical_interpretation": statistical_interpretation,
        "stepwise_history": hist_df.to_dict(orient="records") if "hist_df" in locals() else [],
        "high_vif_vars": high_vif_list,
        "ok_vif_vars": ok_vif_list,
        "model_context": {
            "correlation": _correlation_payload(df_model, ["video_view_count", "video_like_count", "video_comment_count", "engagement_rate"]),
            "vif": vif_rows,
            "selected_pool": final_pool,
            "insight": "Đọc tương quan và VIF cùng kết quả stepwise để phân biệt biến ảnh hưởng thật với biến gây nhiễu hoặc đa cộng tuyến.",
        },
    }

def _simpson_context(df_model: pd.DataFrame):
    """
    Context cho Tab 5C: Simpson/Confounding
    Hard-code:
      Y = log_views
      X = channel_video_count
      Z = channel_subscriber_count
    Dùng standardized coef giống tab6a.
    """
    target = "log_views"
    var_main = "channel_video_count"
    var_confounder = "channel_subscriber_count"

    need = [target, var_main, var_confounder]
    missing = [c for c in need if c not in df_model.columns]
    if missing:
        return {"status": "missing_vars", "missing": missing}

    d = (
        df_model[need]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    n = int(len(d))
    if n < 30:
        return {"status": "insufficient_data", "n": n}

    # chuẩn hoá X và Z để lấy Std Coef
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(d[[var_main, var_confounder]]),
        columns=[var_main, var_confounder],
        index=d.index,
    )
    y = pd.to_numeric(d[target], errors="coerce")

    # Model đơn: Y ~ X
    m_single = sm.OLS(
        y, sm.add_constant(X_scaled[[var_main]], has_constant="add")
    ).fit()
    coef_single = _safe_float(m_single.params.get(var_main))
    p_single = _safe_float(m_single.pvalues.get(var_main))
    r2_single = _safe_float(m_single.rsquared)

    # Model đa: Y ~ X + Z
    m_multi = sm.OLS(
        y, sm.add_constant(X_scaled[[var_main, var_confounder]], has_constant="add")
    ).fit()
    coef_multi = _safe_float(m_multi.params.get(var_main))
    p_multi = _safe_float(m_multi.pvalues.get(var_main))
    r2_multi = _safe_float(m_multi.rsquared)

    flip_detected = None
    pct_change = None
    flip_explanation = ""
    if coef_single is not None and coef_multi is not None:
        flip_detected = (np.sign(coef_single) != np.sign(coef_multi))
        if abs(coef_single) > 1e-12:
            pct_change = _safe_float(abs((coef_multi - coef_single) / coef_single))
        
        if flip_detected:
            flip_explanation = (
                "⚠️ HỆ SỐ THAY ĐỔI DẤU (Simpson's Paradox): "
                f"Hệ số của {var_main} từ {coef_single:.4f} thành {coef_multi:.4f}. "
                "Điều này cho thấy mối quan hệ thực sự phụ thuộc rất lớn vào biến {var_confounder} "
                "('gây nhiễu'). Kênh lớn (subscriber nhiều) có xu hướng upload nhiều video, "
                "nhưng nếu kiểm soát kích thước kênh, mối quan hệ thay đổi hoàn toàn."
            )
        else:
            if pct_change and pct_change > 0.3:
                flip_explanation = (
                    f"⚠️ HỆ SỐ THAY ĐỔI MẠNH ({pct_change*100:.1f}%): "
                    "Mối quan hệ vẫn cùng hướng nhưng độ mạnh giảm đáng kể khi kiểm soát "
                    f"{var_confounder}. Điều này gợi ý confounding yếu hơn (kiểu 'bộ lộc' chứ không "
                    "'đảo ngược')."
                )
            else:
                flip_explanation = (
                    f"✓ HỆ SỐ ỔNĐỊNH: Khi kiểm soát {var_confounder}, hệ số không thay đổi nhiều. "
                    "Mối quan hệ giữa {var_main} và {target} khá độc lập với {var_confounder}."
                )

    corr_xz = _safe_float(d[[var_main, var_confounder]].corr().iloc[0, 1])

    return {
        "status": "ok",
        "n": n,
        "setup": {"y": target, "x": var_main, "z": var_confounder},
        "coef_single_std": coef_single,
        "coef_multi_std": coef_multi,
        "p_single": p_single,
        "p_multi": p_multi,
        "r2_single": r2_single,
        "r2_multi": r2_multi,
        "pct_change": pct_change,         
        "flip_detected": flip_detected,   
        "corr_x_z": corr_xz,              
        "flip_explanation": flip_explanation,
        "interpretation": (
            "So sánh Std Coef (hệ số chuẩn hóa) của {var_main} trước/sau khi kiểm soát {var_confounder}. "
            "Nếu đổi dấu hay đổi mạnh (>30%), có khả năng confounding (biến gây nhiễu)."
        ),
        "chart_context": {
            "single_model": {
                "coef": coef_single, 
                "p_value": p_single, 
                "r2": r2_single,
                "label": "Mô hình đơn (Y ~ X): Chỉ xét X, chưa kiểm soát Z"
            },
            "multi_model": {
                "coef": coef_multi, 
                "p_value": p_multi, 
                "r2": r2_multi,
                "label": "Mô hình đa (Y ~ X + Z): Kiểm soát Z, cô lập tác động thật của X"
            },
            "insight": (
                "So sánh hai mô hình để phát hiện Simpson's Paradox hoặc confounding. "
                "Nếu hệ số đổi dấu/mạnh, Z là biến gây nhiễu và cần kiểm soát để diễn giải đúng."
            ),
        },
    }


def _build_tab5_context(frame, active_cross_filters):
    df_model, genre_dummy_cols = _build_df_model(frame)
    payload = _stepwise_context(df_model, genre_dummy_cols)
    payload["simpson_confounding"] = _simpson_context(df_model)

    # Add comprehensive glossary/definitions
    glossary = {
        "VIF (Variance Inflation Factor)": {
            "definition": "Chỉ số đo mức độ của đa cộng tuyến (multicollinearity) giữa các biến.",
            "scale": "VIF = 1: độc lập hoàn toàn | VIF < 5: chấp nhận được | VIF ≥ 10: đa cộng tuyến nghiêm trọng",
            "impact": "VIF cao → hệ số không ổn định, dễ thay đổi khi thêm/bớt biến. Thường loại bớt biến VIF ≥ 10.",
        },
        "Removed Variables": {
            "definition": "Các biến bị loại khỏi mô hình cuối do: (1) VIF cao hoặc (2) p-value > ngưỡng sl_remove",
            "reason_1": "VIF cao: Biến này tương quan mạnh với biến khác → làm hệ số không ổn định → loại khỏi mô hình.",
            "reason_2": "p-value > sl_remove: Biến này không có ý nghĩa thống kê trong mô hình → loại để đơn giản hóa.",
        },
        "Stepwise Regression": {
            "definition": "Quy trình tự động thêm/bớt biến dựa trên tiêu chí thống kê (p-value).",
            "forward": "Bắt đầu từ mô hình rỗng, từng bước THÊM biến có p-value < p_enter.",
            "backward": "Bắt đầu từ tất cả biến, từng bước BỚT biến có p-value > sl_remove.",
            "both": "Kết hợp forward và backward: vừa thêm biến mới tốt vừa bớt biến không còn cần.",
        },
        "Standardized Coefficients (Std Coef)": {
            "definition": "Hệ số hồi quy sau khi chuẩn hóa X và Y về scale 0-1, giúp so sánh tác động tương đối giữa các biến.",
            "interpretation": "|Std Coef| lớn → biến này ảnh hưởng mạnh hơn đến Y. Dấu (+/-) chỉ hướng tác động.",
            "caveat": "Std Coef chỉ cho thấy mối quan hệ tuyến tính, KHÔNG chứng minh nhân quả.",
        },
        "p-value": {
            "definition": "Xác suất quan sát thấy dữ liệu này nếu biến KHÔNG có tác động (giả thuyết vô).",
            "threshold": "p-value < 0.05: thường được coi là 'có ý nghĩa thống kê' (đủ bằng chứng biến này có tác động).",
            "danger": "p-value nhỏ ≠ tác động lớn. Dữ liệu lớn → p-value nhỏ ngay cả với tác động nhỏ.",
        },
        "R² (Coefficient of Determination)": {
            "definition": "Tỷ lệ phương sai trong Y được mô hình giải thích.",
            "scale": "0 ≤ R² ≤ 1. R² = 0.7 → mô hình giải thích 70% biến động trong lượt xem.",
            "danger": "R² cao ≠ mô hình tốt. Có thể do overfitting hoặc confounding chưa loại bớt.",
        },
        "Confounding (Biến gây nhiễu)": {
            "definition": "Biến Z ảnh hưởng đến cả X và Y → làm cho mối quan hệ X→Y bị bóp méo (spurious).",
            "example": "Ví dụ: Kênh lớn (Z) → upload nhiều video (X) + lượt xem nhiều (Y). Nếu không kiểm soát Z, ta sẽ tưởng X→Y mạnh, thực chất là vì Z.",
            "detection": "So sánh mô hình có/không Z: nếu hệ số X thay đổi mạnh, Z là confounding → cần kiểm soát.",
        },
        "Simpson's Paradox": {
            "definition": "Hiện tượng hệ số X đảo dấu hoàn toàn khi kiểm soát Z (mối quan hệ đảo ngược).",
            "example": "X→Y dương (lợi) nếu bỏ qua Z, nhưng X→Y âm (hại) khi kiểm soát Z.",
            "action": "Nếu phát hiện flip_detected=True, PHẢI kiểm soát Z, không được diễn giải từ mô hình đơn.",
        },
    }

    payload.update(
        {
            "tab": 5,
            "name": "Mô hình hóa",
            "rows": int(len(frame)),
            "active_filters": _active_filter_labels(active_cross_filters),
            "dashboard_state": {
                "columns": list(frame.columns),
                "selected_chart_state": _shared_selection_context(frame),
            },
            "selected_chart_state": _shared_selection_context(frame),
            "glossary": glossary,
            "how_to_read": {
                "step_1_vif": (
                    "Kiểm tra VIF của các biến số. Nếu VIF ≥ 10, biến đó bị loại khỏi mô hình stepwise. "
                    "Điều này tránh đa cộng tuyến làm hệ số không ổn định."
                ),
                "step_2_stepwise": (
                    "Chạy hồi quy từng bước: các biến có p-value < p_enter được THÊM vào mô hình. "
                    "Nếu dùng 'both', các biến có p-value > sl_remove bị BỚT. Kết quả: mô hình tối ưu với các biến "
                    "có ý nghĩa thống kê."
                ),
                "step_3_interpret": (
                    "Đọc Std Coef của các biến được chọn: |Std Coef| lớn → tác động mạnh, dấu chỉ hướng (+/-). "
                    "NHƯNG: Std Coef ≠ nhân quả. Cần kiểm soát confounding (bước 4)."
                ),
                "step_4_confounding": (
                    "Kiểm tra Simpson/Confounding: so sánh hệ số trước/sau kiểm soát Z. Nếu flip_detected=True "
                    "hoặc pct_change > 30%, Z là biến gây nhiễu → phải xem xét trong diễn giải."
                ),
            },
            "tab_insights": [
                "Cần đọc đồng thời correlation, VIF, stepwise và confounding để tránh diễn giải hệ số hồi quy sai.",
                "Hệ số chuẩn hóa và p-value phải được xem như bảng ưu tiên biến, không phải kết luận nhân quả trực tiếp.",
                "Nếu phát hiện flip_detected=True trong Simpson's Paradox, PHẢI kiểm soát biến gây nhiễu khi giải thích.",
            ],
        }
    )
    return payload


def _simple_standardized_coef(df, target, main_var, control_var):
    df_valid = df[[target, main_var, control_var]].dropna().copy()
    if len(df_valid) < 3:
        return None, None

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(df_valid[[main_var, control_var]]), columns=[main_var, control_var], index=df_valid.index)
    y = df_valid[target]

    model_single = sm.OLS(y, sm.add_constant(X_scaled[[main_var]], has_constant="add")).fit()
    model_multi = sm.OLS(y, sm.add_constant(X_scaled[[main_var, control_var]], has_constant="add")).fit()
    return model_single.params.get(main_var, 0), model_multi.params.get(main_var, 0)


def _simple_r2_pair(df, target, var1, var2):
    df_valid = df[[target, var1, var2]].dropna().copy()
    if len(df_valid) < 3:
        return 0, 0, 0

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(df_valid[[var1, var2]]), columns=[var1, var2], index=df_valid.index)
    y = df_valid[target]

    m1 = sm.OLS(y, sm.add_constant(X_scaled[[var1]], has_constant="add")).fit()
    m2 = sm.OLS(y, sm.add_constant(X_scaled[[var2]], has_constant="add")).fit()
    m_both = sm.OLS(y, sm.add_constant(X_scaled[[var1, var2]], has_constant="add")).fit()
    return m1.rsquared, m2.rsquared, m_both.rsquared


def _build_tab6_context(frame, active_cross_filters):
    df_model = _build_df_model(frame)[0]
    payload = {
        "tab": 6,
        "name": "Confounding & Synergy",
        "rows": int(len(frame)),
        "active_filters": _active_filter_labels(active_cross_filters),
        "simpsons_paradox": {},
        "synergy_analysis": {},
        "causal_interpretation": {},
        "selected_chart_state": _shared_selection_context(frame),
    }

    if len(df_model) >= 3 and {"log_views", "channel_video_count", "channel_subscriber_count"}.issubset(df_model.columns):
        coef_single, coef_multi = _simple_standardized_coef(df_model, "log_views", "channel_video_count", "channel_subscriber_count")
        if coef_single is not None:
            payload["simpsons_paradox"] = {
                "target": "log_views",
                "main_variable": "channel_video_count",
                "confounder_variable": "channel_subscriber_count",
                "coefficient_before_control": _safe_float(coef_single),
                "coefficient_after_control": _safe_float(coef_multi),
                "sign_flipping": bool(np.sign(coef_single) != np.sign(coef_multi)),
            }

    pair_options = {
        "licensed_caption": ["is_licensed", "has_caption"],
        "licensed_age": ["is_licensed", "video_age_days"],
        "subscriber_caption": ["channel_subscriber_count", "has_caption"],
    }

    pair_rows = []
    for name, (var1, var2) in pair_options.items():
        if {"log_views", var1, var2}.issubset(df_model.columns):
            r2_1, r2_2, r2_both = _simple_r2_pair(df_model, "log_views", var1, var2)
            pair_rows.append({"pair": name, "var1": var1, "var2": var2, "individual_r2": _safe_float(max(r2_1, r2_2)), "combined_r2": _safe_float(r2_both), "synergy_jump": _safe_float(r2_both - max(r2_1, r2_2)), "complementary_variables": [var1, var2]})

    pair_rows = sorted(pair_rows, key=lambda item: item.get("synergy_jump") or 0, reverse=True)
    best_pair = pair_rows[0] if pair_rows else {}

    payload["synergy_analysis"] = {"pairs": pair_rows, "best_pair": best_pair}
    payload["causal_interpretation"] = {
        "hidden_relationship": "Một số mối tương quan có thể thay đổi khi kiểm soát quy mô kênh hoặc các biến nền tảng.",
        "misleading_correlation": "Nếu chỉ nhìn hệ số đơn biến, ta có thể hiểu sai tác động của một biến do confounder che khuất.",
        "true_effect_explanation": "Hãy ưu tiên diễn giải hệ số sau kiểm soát và các cặp có synergy jump dương khi muốn chọn biến cho mô hình.",
    }
    return payload


def _get_groq_client():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


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


def _render_tab_chatbot(tab_key, tab_label, context_builder, frame, active_cross_filters):
    st.markdown("---")
    with st.expander(f"🤖 Chatbot AI cho {tab_label}", expanded=False):
        tab_context = context_builder(frame, active_cross_filters)
        st.session_state[f"{tab_key}_context"] = tab_context
        context_file = _persist_tab_context(tab_key, tab_context)

        user_question = st.chat_input(f"Hỏi về {tab_label}...", key=f"{tab_key}_chat_input")
        if user_question:
            with st.spinner("AI đang đọc context của tab..."):
                answer, error = _ask_groq(user_question, tab_context, tab_label)
            if error:
                st.error(error)
            else:
                st.session_state[f"{tab_key}_last_question"] = user_question
                st.session_state[f"{tab_key}_last_answer"] = answer
                log_file = _append_tab_history_log(tab_key, tab_label, user_question, answer, context_file, tab_context)
                st.session_state[f"{tab_key}_last_log_file"] = log_file

        last_question = st.session_state.get(f"{tab_key}_last_question")
        last_answer = st.session_state.get(f"{tab_key}_last_answer")
        if last_question and last_answer:
            st.chat_message("user").write(last_question)
            st.chat_message("assistant").write(last_answer)
            last_log_file = st.session_state.get(f"{tab_key}_last_log_file")
            if last_log_file:
                st.caption(f"Đã lưu lịch sử vào: {last_log_file}")
        else:
            st.info("Hãy nhập câu hỏi. Ví dụ: 'Vì sao bubble chart này tăng mạnh ở nhóm nào?' hoặc 'Khung giờ nào tốt nhất?'")