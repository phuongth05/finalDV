# tabs/tab7_ai.py
import streamlit as st
import pandas as pd
import numpy as np
from groq import Groq
import traceback
from datetime import datetime
import json
from dotenv import load_dotenv
import os

def _build_dataframe_profile(filtered_df: pd.DataFrame, max_columns: int = 40, max_preview_rows: int = 3):
    """Tạo bản tóm tắt ngắn về dataframe để AI hiểu schema và nguồn gốc dữ liệu."""
    preview_columns = filtered_df.columns.tolist()[:max_columns]
    dtype_map = {col: str(filtered_df[col].dtype) for col in preview_columns}
    non_null_map = {col: int(filtered_df[col].notna().sum()) for col in preview_columns}

    derived_columns = [
        col for col in [
            "_row_id",
            "video_publish_date",
            "hour",
            "day",
            "genre",
            "title_length",
            "video_type",
            "engagement_rate",
            "channel_size",
            "publish_date",
        ]
        if col in filtered_df.columns
    ]

    tab_guidance = {
        "tab1": "Tổng quan: KPI, phân bố view/duration, top entities, peak periods, fluctuation.",
        "tab2": "Định nghĩa thành công: tương quan, density/heatmap, top 20% vs bottom 20%, viral vs flop.",
        "tab3": "Hành vi người dùng & Từ khóa: supply-demand, caption impact, audience segments, upload trends.",
        "tab4": "Thuật toán & Tối ưu nền tảng: timing, duration, SEO/NLP, platform behavior.",
        "tab5": "Modeling: correlation, VIF, stepwise regression, selected/rejected features, R².",
        "tab6": "Confounding & Synergy: Simpson's paradox, coefficient before/after control, synergy jump.",
    }

    preview_rows = filtered_df.head(max_preview_rows).to_dict(orient="records")

    return {
        "shape": [int(filtered_df.shape[0]), int(filtered_df.shape[1])],
        "columns": preview_columns,
        "dtypes": dtype_map,
        "non_null_counts": non_null_map,
        "derived_columns_from_app_py": derived_columns,
        "tabs_reference": tab_guidance,
        "sample_rows": preview_rows,
    }


def _build_code_prompt(question: str, filtered_df: pd.DataFrame, dataframe_profile: dict) -> str:
    column_list = ", ".join(filtered_df.columns.tolist())
    return f"""
Bạn là Data Engineer và Data Analyst.

Ngữ cảnh dữ liệu:
- `filtered_df` là DataFrame cuối cùng đã được xử lý trong `dashboard/app.py` sau khi load CSV, tạo cột dẫn xuất, và áp dụng bộ lọc hiện tại.
- Không được giả định có cột nào ngoài danh sách schema bên dưới.
- Nếu cột cần dùng không tồn tại, hãy kiểm tra bằng `if col in filtered_df.columns` và viết nhánh dự phòng.
- Khi gặp cột số, ưu tiên `pd.to_numeric(..., errors='coerce')`.
- Khi gặp cột ngày giờ, ưu tiên `pd.to_datetime(..., errors='coerce')`.
- Khi cần nhóm/so sánh, luôn `copy()` dataframe con trước khi gán cột mới để tránh SettingWithCopyWarning.
- Nếu dữ liệu không đủ, hãy trả về `ai_extracted_data` là dict mô tả rõ thiếu gì, không được để code lỗi.

Schema thực tế của `filtered_df`:
{column_list}

Tóm tắt dataframe:
{json.dumps(dataframe_profile, ensure_ascii=False, indent=2, default=str)}

Câu hỏi của người dùng:
"{question}"

Nhiệm vụ:
1. Viết code Python thuần dùng pandas/numpy để trích xuất số liệu trả lời câu hỏi.
2. Bắt buộc lưu kết quả cuối cùng vào biến `ai_extracted_data`.
3. Không dùng print, không hiển thị ra màn hình.
4. Tránh code quá dài, tránh logic phức tạp không cần thiết.
5. Nếu cần dùng logic theo tab, ưu tiên đúng vai trò:
   - Tab 1: KPI, phân bố, top entities, trend.
   - Tab 2: correlation, heatmap, top 20% vs bottom 20%.
   - Tab 3: supply-demand, caption impact, audience segments.
   - Tab 4: timing, duration, SEO/NLP, platform behavior.
   - Tab 5: correlation, VIF, stepwise regression.
   - Tab 6: confounding, Simpson's paradox, synergy.
6. Trả về code thuần, không markdown, không giải thích.

Gợi ý an toàn:
- Có thể bắt đầu bằng `df = filtered_df.copy()`.
- Nếu cần xem schema trong code, dùng `filtered_df.columns`, `filtered_df.dtypes`, `filtered_df.isna().sum()`.
- Nếu phải chọn top/bottom, dùng `nlargest`, `nsmallest`, `sort_values`, `groupby`.
"""


def _build_answer_prompt(question: str, extracted_data) -> str:
    return f"""
Bạn là Data Analyst chuyên nghiệp.
Người dùng hỏi: "{question}"
Dữ liệu đã trích xuất: {extracted_data}

Dựa hoàn toàn vào dữ liệu trên, hãy trả lời ngắn gọn, đúng trọng tâm, ưu tiên insight và khuyến nghị hành động.
Nếu dữ liệu có vẻ là số đã tổng hợp, chuẩn hoá, log-transform, hoặc chỉ là dữ liệu của một tập con sau filter thì phải nói rõ điều đó trước khi diễn giải.
Không được coi mọi con số là raw count nếu ngữ cảnh không xác nhận.
Khi thấy dải số hẹp hoặc dữ liệu dạng tương đối, hãy dùng ngôn ngữ như "trong tập dữ liệu đang lọc" hoặc "giá trị đã chuẩn hoá/tổng hợp" thay vì khẳng định trên toàn bộ dữ liệu gốc.
Nếu dữ liệu chưa đủ, hãy nói rõ thiếu gì thay vì suy đoán.
Trình bày đẹp bằng Markdown.
"""


def render_tab(filtered_df):
    st.subheader("🤖 Trợ lý AI Phân tích Chuyên sâu (Agentic AI)")
    st.markdown("Hệ thống hoạt động theo 2 bước: (1) AI viết code trích xuất dữ liệu 👉 (2) Bạn duyệt code 👉 (3) AI đọc dữ liệu thực tế và đưa ra câu trả lời chuyên sâu.")

    # ================= CẤU HÌNH API =================
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    os.getenv("GROQ_API_KEY")
    try:
        client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        st.error("Chưa cấu hình API Key Groq hợp lệ.")
        return

    dataframe_profile = _build_dataframe_profile(filtered_df)
    df_info = f"DataFrame 'filtered_df' có các cột: {', '.join(filtered_df.columns.tolist())}."
    
    q = st.text_input("Nhập yêu cầu phân tích (Ví dụ: Top 7 video là ai, phân tích xem tại sao họ thành công?)")

    # Lưu trữ các state để quản lý luồng 2 bước
    if "ai_code" not in st.session_state:
        st.session_state.ai_code = ""
    if "ai_answer" not in st.session_state:
        st.session_state.ai_answer = ""

    # ================= BƯỚC 1: AI VIẾT CODE TRÍCH XUẤT (CẦN DUYỆT) =================
    if st.button("✨ Bước 1: Yêu cầu AI lên phương án trích xuất"):
        if not q:
            st.warning("Vui lòng nhập yêu cầu.")
            return
            
        with st.spinner("AI đang viết code trích xuất dữ liệu..."):
            prompt_code = _build_code_prompt(q, filtered_df, dataframe_profile)
            try:
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt_code}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.1 
                )
                code = response.choices[0].message.content.replace("```python", "").replace("```", "").strip()
                st.session_state.ai_code = code
                st.session_state.ai_answer = "" # Reset câu trả lời cũ
            except Exception as e:
                st.error(f"Lỗi: {e}")

    # ================= BƯỚC 2: DUYỆT VÀ AI SINH CÂU TRẢ LỜI =================
    if st.session_state.ai_code:
        st.markdown("### 📝 Mã nguồn AI đề xuất (Trạng thái: Chờ duyệt)")
        
        edited_code = st.text_area("Kiểm tra và chỉnh sửa code trích xuất (nếu cần):", 
                                   value=st.session_state.ai_code, 
                                   height=200)

        if st.button("✅ Bước 2: Duyệt Code & Yêu cầu AI phân tích kết quả"):
            try:
                # 1. Khởi tạo môi trường chứa biến ai_extracted_data
                local_env = {"filtered_df": filtered_df, "pd": pd, "np": np, "ai_extracted_data": None}
                
                # 2. Chạy code do người dùng duyệt (thỏa mãn yêu cầu không thực thi ngầm)
                exec(edited_code, globals(), local_env)
                
                # Lấy dữ liệu ra
                extracted_data = local_env.get("ai_extracted_data")
                
                if extracted_data is None:
                    st.warning("Đoạn code chưa lưu dữ liệu vào biến `ai_extracted_data` như yêu cầu.")
                else:
                    st.success("Trích xuất dữ liệu thành công! AI đang đọc số liệu và phân tích...")
                    
                    # 3. Ghi log[cite: 1]
                    with open("data/ai_logs.txt", "a", encoding="utf-8") as f:
                        f.write(f"\n[{datetime.now()}]\n- Yêu cầu: {q}\n- Code: {edited_code}\n- Data: {extracted_data}\n{'-'*40}")
                    
                    # 4. LUỒNG 2: GỬI DATA CHO AI TRẢ LỜI BẰNG VĂN BẢN
                    prompt_answer = _build_answer_prompt(q, extracted_data)
                    
                    response_answer = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt_answer}],
                        model="llama-3.3-70b-versatile",
                        temperature=0.4 # Tăng nhiệt độ một chút để câu văn tự nhiên hơn
                    )
                    
                    st.session_state.ai_answer = response_answer.choices[0].message.content
                    
            except Exception as e:
                st.error("Có lỗi xảy ra khi chạy code trích xuất!")
                st.code(traceback.format_exc(), language="python")

    # Hiển thị câu trả lời cuối cùng
    if st.session_state.ai_answer:
        st.markdown("---")
        st.markdown("### 💡 Kết quả phân tích từ AI:")
        st.info(st.session_state.ai_answer)