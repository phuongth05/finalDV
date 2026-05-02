# Dashboard — Phân tích nhạc YouTube

Tài liệu hướng dẫn để chạy dashboard Streamlit hiển thị phân tích dữ liệu YouTube.

Yêu cầu
- Python 3.10+ (virtualenv/venv khuyến nghị)
- Thư viện: `finalDV/requirements.txt`

Cài đặt
1. Tạo và kích hoạt môi trường ảo:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Cài đặt thư viện:

```bash
pip install -r ../requirements.txt
```

Dữ liệu
- File dữ liệu mặc định: `data/youtube_vn_music_cleaned.csv` (đường dẫn relative với `finalDV/dashboard`).
- Nếu dùng file khác, đặt theo cấu trúc trên hoặc chỉnh đường dẫn trong `app.py` (biến `pd.read_csv("data/...")`).

Chạy

```bash
cd finalDV
streamlit run dashboard/app.py
```

Ghi chú
- Nếu Streamlit không mở trình duyệt tự động, truy cập http://localhost:8501
- Tinh chỉnh hiệu năng bằng cách giảm kích thước dữ liệu nếu máy chậm.
