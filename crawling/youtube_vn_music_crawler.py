#!/usr/bin/env python3
"""YouTube Data API crawler for Vietnamese music videos."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Set, Tuple

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


DEFAULT_PUBLIC_VIDEO_PARTS = [
    "snippet",
    "contentDetails",
    "statistics",
    "status",
    "topicDetails",
    "recordingDetails",
    "player",
    "liveStreamingDetails",
    "localizations",
    "paidProductPlacementDetails",
]

VIETNAMESE_DIACRITIC_PATTERN = re.compile(
    r"[ăâđêôơưáàạảãấầậẩẫắằặẳẵéèẹẻẽếềệểễ"
    r"íìịỉĩóòọỏõốồộổỗớờợởỡúùụủũứừựửữýỳỵỷỹ]",
    flags=re.IGNORECASE,
)

ALLOWED_SEARCH_ORDERS = {"date", "rating", "relevance", "title", "videoCount", "viewCount"}
QUOTA_ERROR_REASONS = {
    "quotaexceeded",
    "dailylimitexceeded",
    "dailylimitexceededunreg",
    "ratelimitexceeded",
    "userratelimitexceeded",
}
CSV_COLUMNS = [
    "video_id",
    "published_at",
    "channel_id",
    "channel_title",
    "title",
    "description",
    "tags",
    "category_id",
    "default_language",
    "default_audio_language",
    "is_vietnamese_language_flag",
    "has_vietnamese_diacritics",
    "in_publish_range",
    "accepted",
    "filter_reason",
    "view_count",
    "like_count",
    "comment_count",
    "duration",
    "definition",
    "caption",
    "licensed_content",
    "privacy_status",
    "made_for_kids",
    "tags_count",
    "topic_categories",
    "raw_json",
]


@dataclass
class FilterResult:
    accepted: bool
    reason: str
    has_language_flag: bool
    has_vietnamese_diacritics: bool
    in_publish_range: bool


def load_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_config(config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    normalized = dict(config)
    normalized.setdefault("api_key_env_var", "YOUTUBE_API_KEY")
    normalized.setdefault("test_mode", True)
    normalized.setdefault("output_dir", "data/test_run")
    normalized.setdefault("region_code", "VN")
    normalized.setdefault("relevance_language", "")
    normalized.setdefault("music_category_id", "10")
    normalized.setdefault("search_query", "music")
    normalized.setdefault("search_order", "viewCount")
    normalized.setdefault("published_after", "2021-01-01T00:00:00Z")
    normalized.setdefault("published_before", "2025-12-31T23:59:59Z")
    normalized.setdefault("max_search_pages", 1)
    normalized.setdefault("max_videos_to_fetch", 500)
    normalized.setdefault("max_videos_after_filter", 300)
    normalized.setdefault("video_parts", DEFAULT_PUBLIC_VIDEO_PARTS)
    normalized.setdefault("require_music_category", True)
    normalized.setdefault("request_sleep_seconds", 0.15)
    normalized["__config_path__"] = str(config_path)

    if int(normalized["max_search_pages"]) <= 0:
        raise ValueError("max_search_pages must be > 0.")
    if int(normalized["max_videos_to_fetch"]) <= 0:
        raise ValueError("max_videos_to_fetch must be > 0.")
    if int(normalized["max_videos_after_filter"]) <= 0:
        raise ValueError("max_videos_after_filter must be > 0.")
    if str(normalized["search_order"]) not in ALLOWED_SEARCH_ORDERS:
        raise ValueError(f"search_order must be one of: {sorted(ALLOWED_SEARCH_ORDERS)}")

    published_after = parse_utc_timestamp(str(normalized["published_after"]))
    published_before = parse_utc_timestamp(str(normalized["published_before"]))
    if not published_after or not published_before:
        raise ValueError("published_after and published_before must be RFC3339 timestamps.")
    if published_after > published_before:
        raise ValueError("published_after must be <= published_before.")

    return normalized


def build_youtube_client(api_key: str):
    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)


def batched(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def parse_utc_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def collect_candidate_video_ids(youtube: Any, config: Dict[str, Any]) -> Tuple[List[str], Counter]:
    unique_ids: set[str] = set()
    ordered_ids: List[str] = []
    summary = Counter()
    max_candidates = int(config["max_videos_to_fetch"])
    max_pages = int(config["max_search_pages"])
    sleep_seconds = float(config["request_sleep_seconds"])
    page_token = None

    for _ in range(max_pages):
        if len(ordered_ids) >= max_candidates:
            break

        request_kwargs: Dict[str, Any] = {
            "part": "id",
            "q": str(config["search_query"]),
            "type": "video",
            "maxResults": 50,
            "order": str(config["search_order"]),
            "videoCategoryId": str(config["music_category_id"]),
            "pageToken": page_token,
            "publishedAfter": str(config["published_after"]),
            "publishedBefore": str(config["published_before"]),
        }
        if str(config["region_code"]).strip():
            request_kwargs["regionCode"] = str(config["region_code"]).strip()
        if str(config["relevance_language"]).strip():
            request_kwargs["relevanceLanguage"] = str(config["relevance_language"]).strip()

        request = youtube.search().list(**request_kwargs)
        response = request.execute()
        items = response.get("items", [])
        summary["search_api_calls"] += 1
        summary["search_items_seen"] += len(items)

        for item in items:
            video_id = item.get("id", {}).get("videoId")
            if not video_id or video_id in unique_ids:
                continue
            unique_ids.add(video_id)
            ordered_ids.append(video_id)
            if len(ordered_ids) >= max_candidates:
                break

        page_token = response.get("nextPageToken")
        if not page_token:
            break
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return ordered_ids, summary


def has_vietnamese_diacritics(text: str) -> bool:
    return bool(VIETNAMESE_DIACRITIC_PATTERN.search(text))


def is_published_in_range(published_at: str, config: Dict[str, Any]) -> bool:
    published_time = parse_utc_timestamp(published_at)
    if not published_time:
        return False
    start_time = parse_utc_timestamp(str(config["published_after"]))
    end_time = parse_utc_timestamp(str(config["published_before"]))
    if not start_time or not end_time:
        return False
    return start_time <= published_time <= end_time


def evaluate_video(item: Dict[str, Any], config: Dict[str, Any]) -> FilterResult:
    snippet = item.get("snippet", {})
    category_id = str(snippet.get("categoryId", ""))
    expected_music_category = str(config["music_category_id"])
    is_music_category = (not bool(config["require_music_category"])) or (
        category_id == expected_music_category
    )
    if not is_music_category:
        return FilterResult(False, "not-music-category", False, False, False)

    in_publish_range = is_published_in_range(str(snippet.get("publishedAt", "")), config)
    if not in_publish_range:
        return FilterResult(False, "published-out-of-range", False, False, False)

    default_language = str(snippet.get("defaultLanguage", "")).lower()
    default_audio_language = str(snippet.get("defaultAudioLanguage", "")).lower()
    has_language_flag = default_language.startswith("vi") or default_audio_language.startswith("vi")

    text_sources = [
        str(snippet.get("title", "")),
        str(snippet.get("description", "")),
        str(snippet.get("channelTitle", "")),
        " ".join(snippet.get("tags", [])),
    ]
    has_vi_diacritics = has_vietnamese_diacritics(" ".join(text_sources))

    accepted = has_language_flag and has_vi_diacritics
    if accepted and has_language_flag:
        reason = "accepted-language-flag"
    elif accepted:
        reason = "accepted-vietnamese-diacritics"
    else:
        reason = "no-vietnamese-signal"

    return FilterResult(accepted, reason, has_language_flag, has_vi_diacritics, True)


def build_csv_row(item: Dict[str, Any], filter_result: FilterResult) -> Dict[str, Any]:
    snippet = item.get("snippet", {})
    statistics = item.get("statistics", {})
    content_details = item.get("contentDetails", {})
    status = item.get("status", {})
    topic_details = item.get("topicDetails", {})
    tags = snippet.get("tags", [])
    topic_categories = topic_details.get("topicCategories", [])

    return {
        "video_id": str(item.get("id", "")),
        "published_at": str(snippet.get("publishedAt", "")),
        "channel_id": str(snippet.get("channelId", "")),
        "channel_title": str(snippet.get("channelTitle", "")),
        "title": str(snippet.get("title", "")),
        "category_id": str(snippet.get("categoryId", "")),
        "default_language": str(snippet.get("defaultLanguage", "")),
        "default_audio_language": str(snippet.get("defaultAudioLanguage", "")),
        "is_vietnamese_language_flag": filter_result.has_language_flag,
        "has_vietnamese_diacritics": filter_result.has_vietnamese_diacritics,
        "in_publish_range": filter_result.in_publish_range,
        "accepted": filter_result.accepted,
        "filter_reason": filter_result.reason,
        "view_count": str(statistics.get("viewCount", "")),
        "like_count": str(statistics.get("likeCount", "")),
        "comment_count": str(statistics.get("commentCount", "")),
        "duration": str(content_details.get("duration", "")),
        "definition": str(content_details.get("definition", "")),
        "caption": str(content_details.get("caption", "")),
        "licensed_content": content_details.get("licensedContent", ""),
        "privacy_status": str(status.get("privacyStatus", "")),
        "made_for_kids": status.get("madeForKids", ""),
        "tags_count": len(tags) if isinstance(tags, list) else 0,
        "topic_categories": "|".join(topic_categories) if isinstance(topic_categories, list) else "",
        "raw_json": json.dumps(item, ensure_ascii=False),
    }


def fetch_and_filter_videos(
    youtube: Any,
    video_ids: List[str],
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Counter]:
    accepted_videos: List[Dict[str, Any]] = []
    before_filter_csv_rows: List[Dict[str, Any]] = []
    after_filter_csv_rows: List[Dict[str, Any]] = []
    summary = Counter()
    max_after_filter = int(config["max_videos_after_filter"])
    sleep_seconds = float(config["request_sleep_seconds"])
    part_csv = ",".join(config["video_parts"])

    for batch in batched(video_ids, 50):
        if len(accepted_videos) >= max_after_filter:
            break

        request = youtube.videos().list(
            part=part_csv,
            id=",".join(batch),
            maxResults=50,
        )
        response = request.execute()
        items = response.get("items", [])
        summary["videos_api_calls"] += 1
        summary["videos_items_seen"] += len(items)

        for item in items:
            filter_result = evaluate_video(item, config)
            csv_row = build_csv_row(item, filter_result)
            before_filter_csv_rows.append(csv_row)

            if filter_result.accepted:
                accepted_videos.append(item)
                after_filter_csv_rows.append(csv_row)
                summary["accepted"] += 1
                if filter_result.has_language_flag:
                    summary["accepted_with_language_field"] += 1
                if filter_result.has_vietnamese_diacritics:
                    summary["accepted_with_vietnamese_diacritics"] += 1
            else:
                summary["rejected"] += 1
                summary[f"rejected_{filter_result.reason}"] += 1

            if len(accepted_videos) >= max_after_filter:
                break

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return accepted_videos, before_filter_csv_rows, after_filter_csv_rows, summary


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_crawl(config: Dict[str, Any], output_dir: Path, api_key: str) -> Dict[str, Any]:
    youtube = build_youtube_client(api_key)
    candidate_ids, search_summary = collect_candidate_video_ids(youtube, config)
    accepted_videos, before_rows, after_rows, filter_summary = fetch_and_filter_videos(
        youtube,
        candidate_ids,
        config,
    )

    full_json_path = output_dir / "vn_music_videos_full.json"
    full_jsonl_path = output_dir / "vn_music_videos_full.jsonl"
    before_filter_csv_path = output_dir / "vn_music_videos_before_filter.csv"
    after_filter_csv_path = output_dir / "vn_music_videos_after_filter.csv"
    report_path = output_dir / "crawl_report.json"

    write_json(full_json_path, accepted_videos)
    write_jsonl(full_jsonl_path, accepted_videos)
    write_csv(before_filter_csv_path, before_rows)
    write_csv(after_filter_csv_path, after_rows)

    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_file": config["__config_path__"],
        "output_dir": str(output_dir),
        "test_mode": bool(config["test_mode"]),
        "candidate_video_ids": len(candidate_ids),
        "accepted_videos": len(accepted_videos),
        "search_summary": dict(search_summary),
        "filter_summary": dict(filter_summary),
        "restriction_logic": {
            "search_stage": {
                "region_code": config["region_code"],
                "relevance_language": config["relevance_language"],
                "video_category_id": str(config["music_category_id"]),
                "search_query": str(config["search_query"]),
                "search_order": str(config["search_order"]),
                "published_after": str(config["published_after"]),
                "published_before": str(config["published_before"]),
                "keyword_filtering": "disabled",
            },
            "video_stage": {
                "require_music_category": bool(config["require_music_category"]),
                "language_field_check": "snippet.defaultLanguage OR snippet.defaultAudioLanguage startswith 'vi'",
                "diacritic_check": "Vietnamese diacritics in title/description/tags/channelTitle",
                "publish_time_check": (
                    f"{config['published_after']} <= snippet.publishedAt <= {config['published_before']}"
                ),
                "decision_rule": "accept if language_field_check OR diacritic_check",
            },
        },
        "files": {
            "full_json": str(full_json_path),
            "full_jsonl": str(full_jsonl_path),
            "before_filter_csv": str(before_filter_csv_path),
            "after_filter_csv": str(after_filter_csv_path),
            "report": str(report_path),
        },
    }
    write_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl Vietnamese music video metadata from YouTube API.")
    parser.add_argument(
        "--config",
        default="configs/test_config.json",
        help="Path to config JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory from config.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="YouTube Data API key. If omitted, the script reads it from env var in config.",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Only validate and print normalized config, then exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(dotenv_path=Path(".env"))
    config_path = Path(args.config).expanduser().resolve()
    config = normalize_config(load_json_file(config_path), config_path)

    if args.validate_config:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return 0

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else Path(config["output_dir"])
    output_dir = output_dir.resolve()

    api_key = args.api_key or os.getenv(config["api_key_env_var"])
    if not api_key:
        print(
            f"Missing API key. Set {config['api_key_env_var']} or pass --api-key.",
            file=sys.stderr,
        )
        return 2

    try:
        report = run_crawl(config, output_dir, api_key)
    except HttpError as error:
        print(f"YouTube API request failed: {error}", file=sys.stderr)
        return 3

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
