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
            prompt_code = f"""
            Bạn là Data Engineer. DataFrame tên 'filtered_df'. Cột: {df_info}
            Câu hỏi của người dùng: "{q}"
            
            Nhiệm vụ: Viết code Python (pandas) để trích xuất các số liệu cần thiết nhằm trả lời câu hỏi trên.
            
            QUY TẮC:
            1. Phải có comment tiếng Việt giải thích code.
            2. KHÔNG in ra màn hình. BẮT BUỘC lưu kết quả cuối cùng vào một biến tên là `ai_extracted_data` (có thể là dict, list, hoặc chuỗi JSON).
            3. Trả về code thuần, không markdown.
            """
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
                    prompt_answer = f"""
                    Bạn là Data Analyst chuyên nghiệp.
                    Người dùng hỏi: "{q}"
                    Dữ liệu tôi đã trích xuất được từ database là: {extracted_data}
                    
                    Dựa TẤT CẢ vào dữ liệu trên, hãy trả lời câu hỏi của người dùng một cách logic, sâu sắc và dễ hiểu. Trình bày đẹp bằng Markdown.
                    """
                    
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