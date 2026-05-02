# Crawling — YouTube VN music (YouTube Data API)

Hướng dẫn chạy các script thu thập và làm giàu metadata từ YouTube Data API.

Yêu cầu
- Python 3.10+ (virtualenv/venv khuyến nghị)
- API key cho YouTube Data API v3
- Thư viện: `finalDV/requirements.txt`

Cài đặt
1. Tạo và kích hoạt môi trường ảo:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Cài đặt thư viện:

```bash
pip install -r finalDV/requirements.txt
```

Cấu hình
- Sao chép `finalDV/crawling/.env.example` thành `.env` và gán `YOUTUBE_API_KEY`:

```bash
cp finalDV/crawling/.env.example .env
# edit .env và thêm YOUTUBE_API_KEY=your_api_key
```

Sử dụng
- Kiểm tra config (không gọi API):

```bash
cd finalDV/crawling
python youtube_vn_music_crawler.py --config configs/test_config.json --validate-config
```

- Chạy crawl (ví dụ, từ repository root):

```bash
# Chạy test crawl với số luợng hạn chế (ví dụ 10 video) để kiểm tra hoạt động
python youtube_vn_music_crawler.py --config configs/test_config.json --output-dir data/test_run

# Chạy crawl đầy đủ (ví dụ, với config chính)
python youtube_vn_music_crawler.py --config configs/full_crawl_config.json --output-dir data/full_crawl_run
```

- Chạy script làm giàu thông tin kênh (resume nếu đã có file đích):

```bash
python finalDV/crawling/youtube_vn_channel_enrich.py --input-csv data/vn_music_simple.csv --output-csv data/vn_music_simple_enriched.csv --api-key "$YOUTUBE_API_KEY"
```

Ghi chú
- Nếu không truyền `--api-key`, script `youtube_vn_music_crawler.py` sẽ đọc biến môi trường được chỉ định trong file config (mặc định `YOUTUBE_API_KEY`).
- Script hỗ trợ checkpointing / resume (nếu output đã tồn tại, sẽ tiếp tục từ file đó).
- Theo dõi quota API và chạy cẩn trọng để tránh vượt hạn mức.
