import argparse
import csv
import json
import os
import sys

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# The new columns we want to add to your CSV
# (Removed hidden_subscriber_count, added channel_country)
NEW_COLUMNS = [
    "channel_subscriber_count",
    "channel_view_count",
    "channel_video_count",
    "raw_channel_json"
]

def chunked(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

def main():
    parser = argparse.ArgumentParser(description="Enrich CSV with YouTube Channel data.")
    parser.add_argument("--input-csv", required=True, help="Path to input CSV (e.g., vn_music_simple.csv)")
    parser.add_argument("--output-csv", required=True, help="Path to output CSV")
    parser.add_argument("--api-key", required=True, help="Your active YouTube API Key")
    args = parser.parse_args(sys.argv[1:])

    input_file = args.input_csv
    output_file = args.output_csv

    # 1. Read existing data
    rows = []
    fieldnames = []
    
    if os.path.exists(output_file):
        print(f"Resuming from {output_file}...")
        file_to_read = output_file
    else:
        print(f"Starting fresh from {input_file}...")
        file_to_read = input_file

    with open(file_to_read, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows.append(row)

    # Add new columns to fieldnames if they don't exist
    for col in NEW_COLUMNS:
        if col not in fieldnames:
            fieldnames.append(col)

    # Cleanup: If the old deprecated column is in the CSV, remove it
    if "channel_hidden_subscriber_count" in fieldnames:
        fieldnames.remove("channel_hidden_subscriber_count")
        for row in rows:
            row.pop("channel_hidden_subscriber_count", None)

    # 2. Identify missing channels (Checkpointing Logic)
    # We only want to fetch channels for rows where raw_channel_json is empty or missing
    missing_channel_ids = set()
    for row in rows:
        if not row.get("raw_channel_json"):
            missing_channel_ids.add(row["channel_id"])

    missing_channel_ids = list(missing_channel_ids)
    print(f"Found {len(missing_channel_ids)} unique channels that need data.")

    if not missing_channel_ids:
        print("All rows already have channel data! Exiting.")
        return

    # 3. Setup YouTube API Client
    youtube = build("youtube", "v3", developerKey=args.api_key)

    # 4. Fetch in batches of 50
    channel_data_map = {}
    quota_exceeded = False

    try:
        for batch in chunked(missing_channel_ids, 50):
            print(f"Fetching batch of {len(batch)} channels...")
            
            response = youtube.channels().list(
                part="statistics,snippet",
                id=",".join(batch)
            ).execute()

            for item in response.get("items", []):
                cid = item["id"]
                stats = item.get("statistics", {})
                snippet = item.get("snippet", {})
                
                channel_data_map[cid] = {
                    "channel_subscriber_count": stats.get("subscriberCount", "0"),
                    "channel_view_count": stats.get("viewCount", "0"),
                    "channel_video_count": stats.get("videoCount", "0"),
                    "raw_channel_json": json.dumps(item, ensure_ascii=False)
                }

    except HttpError as e:
        # Better error handling for quota limits
        if e.resp.status in [403]:
            print(f"\n[WARNING] API Quota Exceeded or Forbidden! Saving progress before exiting...")
            quota_exceeded = True
        else:
            print(f"\n[ERROR] HTTP Error: {e}")
            quota_exceeded = True # Save progress on any crash just to be safe

    # 5. Merge fetched data back into rows
    updated_count = 0
    for row in rows:
        cid = row.get("channel_id")
        if cid in channel_data_map and not row.get("raw_channel_json"):
            data = channel_data_map[cid]
            row["channel_subscriber_count"] = data["channel_subscriber_count"]
            row["channel_view_count"] = data["channel_view_count"]
            row["channel_video_count"] = data["channel_video_count"]
            row["raw_channel_json"] = data["raw_channel_json"]
            updated_count += 1

    print(f"Successfully updated {updated_count} rows.")

    # 6. Save back to CSV
    print(f"Saving to {output_file}...")
    with open(output_file, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if quota_exceeded:
        print("Process stopped due to API limits. Run the script again with a new --api-key and it will resume from where it left off!")

if __name__ == "__main__":
    main()