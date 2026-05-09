import json
import os
import re

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
Trả lời ngắn gọn, đúng trọng tâm, có thể nêu insight hoặc gợi ý hành động.

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

    payload = {
        "tab": 1,
        "name": "Tổng quan",
        "rows": int(len(frame)),
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
        "distribution": {
            "view_distribution_shape": _distribution_shape(views),
            "duration_distribution": _distribution_shape(durations),
            "outlier_existence": {
                "views": _outlier_summary(views),
                "duration_minutes": _outlier_summary(durations),
            },
        },
        "top_entities": {
            "top_channels": _top_table_payload(frame, "video_view_count", "channel_title", 5),
            "top_videos": _top_table_payload(frame, "video_view_count", "video_title", 5),
        },
        "trend": {
            "peak_periods": trend.get("peak_periods", []),
            "fluctuation": trend.get("fluctuation", {}),
        },
        "top_channel_by_views": top_channel,
        "selected_chart_state": _shared_selection_context(frame),
    }
    return payload


def _build_tab2_context(frame, active_cross_filters):
    corr_payload = _pairwise_correlations(frame, ["video_view_count", "video_like_count", "video_comment_count", "engagement_rate"])
    heat_payload = _density_hotspot(frame)
    compare_payload = _compare_groups(frame)

    payload = {
        "tab": 2,
        "name": "Định nghĩa thành công",
        "rows": int(len(frame)),
        "active_filters": _active_filter_labels(active_cross_filters),
        "correlation": {
            "pairs": corr_payload,
            "views_vs_likes": next((item["corr"] for item in corr_payload if set(item["pair"]) == {"video_view_count", "video_like_count"}), None),
            "views_vs_comments": next((item["corr"] for item in corr_payload if set(item["pair"]) == {"video_view_count", "video_comment_count"}), None),
            "engagement_correlation": next((item["corr"] for item in corr_payload if "engagement_rate" in item["pair"]), None),
        },
        "density_heatmap": heat_payload,
        "comparative_analysis": compare_payload,
        "key_findings": compare_payload.get("key_findings", {}) if isinstance(compare_payload, dict) else {},
        "selected_chart_state": _shared_selection_context(frame),
    }
    return payload


def _build_tab3_context(frame, active_cross_filters):
    payload = {
        "tab": 3,
        "name": "Hành vi người dùng & Từ khóa",
        "rows": int(len(frame)),
        "active_filters": _active_filter_labels(active_cross_filters),
        "supply_vs_demand": _genre_supply_demand(frame),
        "caption_impact": _caption_impact(frame),
        "audience_segments": _audience_segments(frame),
        "upload_trends": _upload_trends(frame),
        "selected_chart_state": _shared_selection_context(frame),
    }
    return payload


def _build_tab4_context(frame, active_cross_filters):
    payload = {
        "tab": 4,
        "name": "Thuật toán & Tối ưu nền tảng",
        "rows": int(len(frame)),
        "active_filters": _active_filter_labels(active_cross_filters),
        "timing_optimization": _timing_optimization(frame),
        "duration_optimization": _duration_optimization(frame),
        "seo_nlp": {
            "keyword_impacts": _keyword_impacts(frame),
            "metadata_effectiveness": _metadata_effectiveness(frame),
        },
        "platform_behavior": _platform_behavior(frame),
        "selected_chart_state": _shared_selection_context(frame),
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
        return {"status": "insufficient_data", "high_vif_vars": high_vif_list, "ok_vif_vars": ok_vif_list}

    try:
        selected, model, hist_df = _stepwise_selection(df_model, target_var, final_pool, direction="both", sl_enter=0.05, sl_remove=0.10)
    except Exception as exc:
        return {"status": "error", "error": str(exc), "high_vif_vars": high_vif_list, "ok_vif_vars": ok_vif_list}

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
                model_info = {"r2": _safe_float(m_std.rsquared), "adjusted_r2": _safe_float(m_std.rsquared_adj), "selected_features": selected, "rejected_features": rejected}
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
    }


def _build_tab5_context(frame, active_cross_filters):
    df_model, genre_dummy_cols = _build_df_model(frame)
    payload = _stepwise_context(df_model, genre_dummy_cols)
    payload.update(
        {
            "tab": 5,
            "name": "Mô hình hóa",
            "rows": int(len(frame)),
            "active_filters": _active_filter_labels(active_cross_filters),
            "selected_chart_state": _shared_selection_context(frame),
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

        last_question = st.session_state.get(f"{tab_key}_last_question")
        last_answer = st.session_state.get(f"{tab_key}_last_answer")
        if last_question and last_answer:
            st.chat_message("user").write(last_question)
            st.chat_message("assistant").write(last_answer)
        else:
            st.info("Hãy nhập câu hỏi. Ví dụ: 'Vì sao bubble chart này tăng mạnh ở nhóm nào?' hoặc 'Khung giờ nào tốt nhất?'")