# tabs/tab7_ai.py
import streamlit as st
import pandas as pd
import numpy as np
from groq import Groq
import traceback
from datetime import datetime
import json
import re
from dotenv import load_dotenv
import os

def _build_dataframe_profile(filtered_df: pd.DataFrame, max_columns: int = 40):
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

    return {
        "shape": [int(filtered_df.shape[0]), int(filtered_df.shape[1])],
        "columns": preview_columns,
        "dtypes": dtype_map,
        "non_null_counts": non_null_map,
        "derived_columns_from_app_py": derived_columns,
        "tabs_reference": tab_guidance,
    }


@st.cache_data(show_spinner=False)
def _get_dataframe_profile(filtered_df: pd.DataFrame):
    return _build_dataframe_profile(filtered_df)


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
1. Trả về đúng 2 khối code Python, theo đúng thứ tự:
   - Khối 1: code trích xuất dữ liệu, bắt buộc lưu kết quả cuối cùng vào biến `ai_extracted_data`.
   - Khối 2: code tạo trực quan hoá, có thể tạo biến `ai_figures` (list figure plotly/matplotlib) và/hoặc `ai_images` (list ảnh path / bytes / PIL image).
2. Khối 1 và khối 2 đều phải là code Python thuần dùng pandas/numpy, không markdown, không giải thích.
3. Không dùng print, không hiển thị ra màn hình trong khối 1.
4. Khối 2 chỉ nên tập trung vào visualization, có thể dùng dữ liệu đã trích xuất hoặc dataframe phụ trợ cần thiết.
5. Tránh code quá dài, tránh logic phức tạp không cần thiết.
6. Trả về đúng 2 khối code, không thêm nội dung nào khác.

Gợi ý an toàn:
- Có thể bắt đầu bằng `df = filtered_df.copy()`.
- Nếu cần xem schema trong code, dùng `filtered_df.columns`, `filtered_df.dtypes`, `filtered_df.isna().sum()`.
- Nếu phải chọn top/bottom, dùng `nlargest`, `nsmallest`, `sort_values`, `groupby`.
"""


def _extract_python_code_blocks(text: str):
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if len(blocks) >= 2:
        return blocks[0].strip(), blocks[1].strip()
    if len(blocks) == 1:
        return blocks[0].strip(), ""
    return text.strip(), ""


def _build_answer_prompt(question: str, extracted_data) -> str:
    return f"""
Bạn là Data Analyst chuyên nghiệp.
Người dùng hỏi: "{question}"
Dữ liệu đã trích xuất: {extracted_data}

Dựa hoàn toàn vào dữ liệu trên, hãy trả lời ngắn gọn, đúng trọng tâm, ưu tiên insight và khuyến nghị hành động.
Nếu dữ liệu chưa đủ, hãy nói rõ thiếu gì thay vì suy đoán.
Trình bày đẹp bằng Markdown.
"""


@st.cache_data(show_spinner=False)
def _resolve_history_dir(log_dir: str = "data/ai_logs"):
    possible_paths = [
        log_dir,
        os.path.join(os.path.dirname(__file__), "..", "..", log_dir),
        os.path.abspath(log_dir),
    ]

    for path in possible_paths:
        if os.path.isdir(path):
            return path
    return None


@st.cache_data(show_spinner=False)
def _list_history_summaries(log_dir: str = "data/ai_logs", limit: int = 20):
    """Trả về dữ liệu lịch sử nhẹ nhàng để thanh bên không phân tích mọi tin nhắn."""
    summaries = []
    actual_path = _resolve_history_dir(log_dir)
    if not actual_path:
        return summaries

    log_files = [f for f in os.listdir(actual_path) if f.endswith(".jsonl")]
    log_files.sort(key=lambda name: os.path.getmtime(os.path.join(actual_path, name)), reverse=True)

    for log_file in log_files[:limit]:
        file_path = os.path.join(actual_path, log_file)
        preview = os.path.splitext(log_file)[0]
        has_code = False

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_index, line in enumerate(f):
                    if not line.strip():
                        continue

                    entry = json.loads(line)
                    if line_index == 0:
                        preview = str(entry.get("content", preview))
                    if entry.get("code") or entry.get("viz_code"):
                        has_code = True
                    if line_index >= 1:
                        break
        except Exception:
            pass

        summaries.append(
            {
                "path": file_path,
                "preview": preview,
                "has_code": has_code,
                "updated_at": datetime.fromtimestamp(os.path.getmtime(file_path)),
            }
        )

    return summaries


@st.cache_data(show_spinner=False)
def _load_conversation_from_log(file_path: str):
    """Tải một cuộc trò chuyện được lưu từ một tệp JSONL duy nhất."""
    messages = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "role" in entry and "content" in entry:
                    messages.append(entry)
    except Exception:
        pass

    return messages


def render_tab(filtered_df):
    st.subheader("🤖 AI Analytics Assistant")

    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    try:
        client = Groq(api_key=GROQ_API_KEY)
    except Exception:
        st.error("Chưa cấu hình API Key Groq hợp lệ.")
        return

    # ================= SESSION STATE =================
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "selected_history_file" not in st.session_state:
        st.session_state.selected_history_file = None

    if "selected_history_messages" not in st.session_state:
        st.session_state.selected_history_messages = []

    if "pending_code" not in st.session_state:
        st.session_state.pending_code = ""

    if "pending_viz_code" not in st.session_state:
        st.session_state.pending_viz_code = ""

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = ""

    if "pending_turn_id" not in st.session_state:
        st.session_state.pending_turn_id = 0

    # ================= SIDEBAR HISTORY =================
    with st.sidebar:
        st.markdown("## 🕘 Lịch sử trò chuyện")
        history_items = _list_history_summaries()

        if history_items:
            for display_idx, item in enumerate(history_items):
                col1, col2 = st.columns([4, 1])
                with col1:
                    button_label = f"👤 {item['preview'][:40]}"
                    if len(item["preview"]) > 40:
                        button_label += "..."

                    if st.button(button_label, key=f"hist_{display_idx}", use_container_width=True):
                        st.session_state.selected_history_file = item["path"]
                        st.session_state.selected_history_messages = _load_conversation_from_log(item["path"])
                        st.session_state.pending_code = ""
                        st.session_state.pending_viz_code = ""
                        st.session_state.pending_question = ""
                        st.session_state.pending_turn_id += 1
                        st.rerun()

                with col2:
                    if item["has_code"]:
                        if st.button("↻", key=f"load_{display_idx}", help="Tải mã của cuộc trò chuyện này"):
                            # Only select the history conversation for viewing.
                            # Do NOT populate pending_code/pending_viz_code here to avoid
                            # building the editor/frame. The user must press Re-run to
                            # execute or load the code into the run/editor UI.
                            conversation = _load_conversation_from_log(item["path"])
                            st.session_state.selected_history_file = item["path"]
                            st.session_state.selected_history_messages = conversation
                            st.rerun()
        else:
            st.markdown("Lịch sử trò chuyện trống")

    # ================= CHAT HISTORY =================
    selected_history = st.session_state.selected_history_messages

    if selected_history:
        with st.spinner("Đang tải cuộc trò chuyện..."):
            for msg in selected_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

                    if msg["role"] == "assistant":
                        if msg.get("code"):
                            with st.expander("Mã Nguồn"):
                                st.code(msg["code"], language="python")

                        if msg.get("viz_code"):
                            with st.expander("Mã Trực Quan"):
                                st.code(msg["viz_code"], language="python")
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Chạy Lại", use_container_width=True):
                try:
                    assistant_msg = next((msg for msg in reversed(selected_history) if msg.get("role") == "assistant"), None)
                    if assistant_msg:
                        code_to_run = assistant_msg.get("code", "")
                        viz_code_to_run = assistant_msg.get("viz_code", "")
                        question_text = assistant_msg.get("question", "")
                        
                        if code_to_run:
                            # Execute extraction code
                            local_env = {
                                "filtered_df": filtered_df,
                                "pd": pd,
                                "np": np,
                                "ai_extracted_data": None
                            }
                            
                            exec(code_to_run, globals(), local_env)
                            
                            # Execute viz code
                            if viz_code_to_run.strip():
                                local_env.setdefault("ai_figures", [])
                                local_env.setdefault("ai_images", [])
                                exec(viz_code_to_run, globals(), local_env)
                            
                            # Display results
                            figs = local_env.get("ai_figures") or []
                            imgs = local_env.get("ai_images") or []
                            
                            if figs:
                                st.markdown("### Kết Quả")
                                for fig in figs:
                                    try:
                                        st.plotly_chart(fig, use_container_width=True)
                                    except Exception:
                                        st.pyplot(fig)
                            
                            if imgs:
                                st.markdown("### Hình Ảnh")
                                for img in imgs:
                                    st.image(img)
                            
                            # Get fresh analysis from Groq
                            extracted_data = local_env.get("ai_extracted_data")
                            if extracted_data is not None:
                                with st.spinner("Đang tạo phân tích mới..."):
                                    prompt_answer = _build_answer_prompt(
                                        question_text,
                                        extracted_data
                                    )
                                    
                                    response_answer = client.chat.completions.create(
                                        messages=[{
                                            "role": "user",
                                            "content": prompt_answer
                                        }],
                                        model="llama-3.3-70b-versatile",
                                        temperature=0.4
                                    )
                                    
                                    fresh_answer = response_answer.choices[0].message.content
                                    st.markdown("### Phân Tích Mới")
                                    st.markdown(fresh_answer)
                except Exception:
                    st.error("Lỗi khi chạy lại!")
                    st.code(traceback.format_exc(), language="python")
        
        with col2:
            if st.button("Bắt Đầu Trò Chuyện Mới", use_container_width=True):
                st.session_state.messages = []
                st.session_state.selected_history_file = None
                st.session_state.selected_history_messages = []
                st.session_state.pending_code = ""
                st.session_state.pending_viz_code = ""
                st.session_state.pending_question = ""
                st.rerun()
    else:
        # No history selected - blank chat for new conversation
        if st.session_state.messages:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

                    if msg["role"] == "assistant":
                        if msg.get("code"):
                            with st.expander("Mã Nguồn"):
                                st.code(msg["code"], language="python")

                        if msg.get("viz_code"):
                            with st.expander("Mã Trực Quan"):
                                st.code(msg["viz_code"], language="python")
        else:
            st.markdown("_Sẵn sàng phân tích mới. Nhập câu hỏi của bạn dưới đây._")

    # ================= CHAT INPUT =================
    q = st.chat_input("Hỏi AI phân tích dữ liệu...")

    if q:
        dataframe_profile = _get_dataframe_profile(filtered_df)
        st.session_state.selected_history_file = None
        st.session_state.selected_history_messages = []
        st.session_state.pending_question = q
        st.session_state.pending_turn_id += 1
        st.session_state.messages.append({
            "role": "user",
            "content": q
        })

        with st.chat_message("user"):
            st.markdown(q)

        with st.spinner("AI đang suy nghĩ..."):

                try:
                    # ================= STEP 1: GENERATE CODE =================
                    prompt_code = _build_code_prompt(
                        q,
                        filtered_df,
                        dataframe_profile
                    )

                    response = client.chat.completions.create(
                        messages=[{
                            "role": "user",
                            "content": prompt_code
                        }],
                        model="llama-3.3-70b-versatile",
                        temperature=0.1
                    )

                    response_text = response.choices[0].message.content

                    code, viz_code = _extract_python_code_blocks(response_text)

                    st.session_state.pending_code = code
                    st.session_state.pending_viz_code = viz_code

                except Exception:
                    st.error("Có lỗi xảy ra!")
                    st.code(
                        traceback.format_exc(),
                        language="python"
                    )

    if st.session_state.pending_code:
        st.markdown("### Kế Hoạch Trích Xuất Do AI Tạo Ra")

        current_turn_id = st.session_state.pending_turn_id
        edited_code = st.text_area(
            "Mã Nguồn",
            value=st.session_state.pending_code,
            height=250,
            key=f"code_{current_turn_id}"
        )

        edited_viz_code = st.text_area(
            "Mã Trực Quan",
            value=st.session_state.pending_viz_code,
            height=250,
            key=f"viz_{current_turn_id}"
        )

        if st.button(" Chạy Phân Tích", key=f"run_{current_turn_id}"):
            try:
                active_question = st.session_state.pending_question or q

                local_env = {
                    "filtered_df": filtered_df,
                    "pd": pd,
                    "np": np,
                    "ai_extracted_data": None
                }

                exec(edited_code, globals(), local_env)

                if edited_viz_code.strip():
                    local_env.setdefault("ai_figures", [])
                    local_env.setdefault("ai_images", [])
                    exec(edited_viz_code, globals(), local_env)

                extracted_data = local_env.get("ai_extracted_data")

                if extracted_data is None:
                    st.warning("Code chưa tạo ai_extracted_data")
                else:
                    figs = local_env.get("ai_figures") or []
                    imgs = local_env.get("ai_images") or []

                    if figs:
                        for fig in figs:
                            try:
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception:
                                st.pyplot(fig)

                    if imgs:
                        for img in imgs:
                            st.image(img)

                    prompt_answer = _build_answer_prompt(
                        active_question,
                        extracted_data
                    )

                    response_answer = client.chat.completions.create(
                        messages=[{
                            "role": "user",
                            "content": prompt_answer
                        }],
                        model="llama-3.3-70b-versatile",
                        temperature=0.4
                    )

                    final_answer = response_answer.choices[0].message.content

                    st.markdown(final_answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_answer,
                        "code": edited_code,
                        "viz_code": edited_viz_code,
                        "question": active_question
                    })

                    # Save to JSONL file
                    log_dir = "data/ai_logs"
                    os.makedirs(log_dir, exist_ok=True)
                    
                    # Generate unique filename based on timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    log_file = os.path.join(log_dir, f"{timestamp}.jsonl")
                    
                    # Save as JSONL (one JSON object per line)
                    with open(log_file, "w", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "role": "user",
                            "content": active_question
                        }, ensure_ascii=False) + "\n")
                        f.write(json.dumps({
                            "role": "assistant",
                            "content": final_answer,
                            "code": edited_code,
                            "viz_code": edited_viz_code,
                            "question": active_question
                        }, ensure_ascii=False) + "\n")

                    st.session_state.pending_code = ""
                    st.session_state.pending_viz_code = ""
                    st.session_state.pending_question = ""

            except Exception:
                st.error("Có lỗi xảy ra!")
                st.code(traceback.format_exc(), language="python")