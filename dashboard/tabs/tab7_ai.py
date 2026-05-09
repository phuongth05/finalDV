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
1. Trả về đúng 2 khối code Python, theo đúng thứ tự:
   - Khối 1: code trích xuất dữ liệu, bắt buộc lưu kết quả cuối cùng vào biến `ai_extracted_data`.
   - Khối 2: code tạo trực quan hoá, có thể tạo biến `ai_figures` (list figure plotly/matplotlib) và/hoặc `ai_images` (list ảnh path / bytes / PIL image).
2. Khối 1 và khối 2 đều phải là code Python thuần dùng pandas/numpy, không markdown, không giải thích.
3. Không dùng print, không hiển thị ra màn hình trong khối 1.
4. Khối 2 chỉ nên tập trung vào visualization, có thể dùng dữ liệu đã trích xuất hoặc dataframe phụ trợ cần thiết.
5. Tránh code quá dài, tránh logic phức tạp không cần thiết.
6. Nếu cần dùng logic theo tab, ưu tiên đúng vai trò:
   - Tab 1: KPI, phân bố, top entities, trend.
   - Tab 2: correlation, heatmap, top 20% vs bottom 20%.
   - Tab 3: supply-demand, caption impact, audience segments.
   - Tab 4: timing, duration, SEO/NLP, platform behavior.
   - Tab 5: correlation, VIF, stepwise regression.
   - Tab 6: confounding, Simpson's paradox, synergy.
7. Trả về đúng 2 khối code, không thêm nội dung nào khác.

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
Nếu dữ liệu có vẻ là số đã tổng hợp, chuẩn hoá, log-transform, hoặc chỉ là dữ liệu của một tập con sau filter thì phải nói rõ điều đó trước khi diễn giải.
Không được coi mọi con số là raw count nếu ngữ cảnh không xác nhận.
Khi thấy dải số hẹp hoặc dữ liệu dạng tương đối, hãy dùng ngôn ngữ như "trong tập dữ liệu đang lọc" hoặc "giá trị đã chuẩn hoá/tổng hợp" thay vì khẳng định trên toàn bộ dữ liệu gốc.
Nếu dữ liệu chưa đủ, hãy nói rõ thiếu gì thay vì suy đoán.
Trình bày đẹp bằng Markdown.
"""


@st.cache_resource
def _load_history_from_log(log_dir: str = "data/ai_logs"):
    """Parse log files from directory and reconstruct chat history. Cached."""
    messages = []
    try:
        # Try different path variations
        possible_paths = [
            log_dir,
            os.path.join(os.path.dirname(__file__), "..", "..", log_dir),
            os.path.abspath(log_dir),
        ]
        
        actual_path = None
        for p in possible_paths:
            if os.path.isdir(p):
                actual_path = p
                break
        
        if not actual_path:
            return messages
        
        # Read all .jsonl files from the directory
        log_files = sorted([f for f in os.listdir(actual_path) if f.endswith('.jsonl')])
        
        for log_file in log_files:
            file_path = os.path.join(actual_path, log_file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            entry = json.loads(line)
                            if "role" in entry and "content" in entry:
                                messages.append(entry)
                        except json.JSONDecodeError:
                            pass
            except Exception as e:
                pass

    except Exception as e:
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

    dataframe_profile = _build_dataframe_profile(filtered_df)

    # ================= SESSION STATE =================
    if "messages" not in st.session_state:
        st.session_state.messages = _load_history_from_log()

    if "pending_code" not in st.session_state:
        st.session_state.pending_code = ""

    if "pending_viz_code" not in st.session_state:
        st.session_state.pending_viz_code = ""

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = ""

    if "pending_turn_id" not in st.session_state:
        st.session_state.pending_turn_id = 0

    if "selected_message_idx" not in st.session_state:
        st.session_state.selected_message_idx = None

    # ================= SIDEBAR HISTORY =================
    with st.sidebar:
        st.markdown("## 🕘 History")
        
        if st.session_state.messages:
            # Display last 20 messages in reverse (newest first)
            history_items = list(enumerate(reversed(st.session_state.messages[-20:])))
            for display_idx, (actual_idx, msg) in enumerate(history_items):
                if msg["role"] == "user":
                    content_preview = msg['content'][:40]
                    if len(msg['content']) > 40:
                        content_preview += "..."
                    
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        if st.button(f"👤 {content_preview}", key=f"hist_{actual_idx}_{display_idx}", use_container_width=True):
                            st.session_state.selected_message_idx = len(st.session_state.messages) - 1 - actual_idx
                            st.rerun()
                    
                    # Show "Load" button next to history if it has code
                    with col2:
                        message_at_idx = st.session_state.messages[len(st.session_state.messages) - 1 - actual_idx]
                        if message_at_idx.get("code"):
                            if st.button("↻", key=f"load_{actual_idx}_{display_idx}", help="Load this conversation's code"):
                                st.session_state.pending_code = message_at_idx.get("code", "")
                                st.session_state.pending_viz_code = message_at_idx.get("viz_code", "")
                                st.session_state.pending_question = message_at_idx.get("question", msg.get("content", ""))
                                st.session_state.pending_turn_id += 1
                                st.rerun()
        else:
            st.markdown("_No history yet_")

    # ================= CHAT HISTORY =================
    # Only show history if a specific message is selected from sidebar
    if st.session_state.selected_message_idx is not None:
        with st.spinner("⏳ Loading conversation..."):
            # Display only the selected conversation (user message + assistant response)
            selected_idx = st.session_state.selected_message_idx
            
            # User message is at selected_idx
            if selected_idx < len(st.session_state.messages):
                msg = st.session_state.messages[selected_idx]
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            
            # Assistant message is the next message after user message
            assistant_idx = selected_idx + 1
            if assistant_idx < len(st.session_state.messages) and st.session_state.messages[assistant_idx]["role"] == "assistant":
                msg = st.session_state.messages[assistant_idx]
                with st.chat_message("assistant"):
                    st.markdown(msg["content"])
                    
                    if msg.get("code"):
                        with st.expander("📝 Extraction Code"):
                            st.code(msg["code"], language="python")
                    
                    if msg.get("viz_code"):
                        with st.expander("🎨 Visualization Code"):
                            st.code(msg["viz_code"], language="python")
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Re-run", use_container_width=True):
                try:
                    # Get the assistant message with code
                    assistant_idx = st.session_state.selected_message_idx + 1
                    if assistant_idx < len(st.session_state.messages):
                        assistant_msg = st.session_state.messages[assistant_idx]
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
                                st.markdown("### 📊 Results")
                                for fig in figs:
                                    try:
                                        st.plotly_chart(fig, use_container_width=True)
                                    except Exception:
                                        st.pyplot(fig)
                            
                            if imgs:
                                st.markdown("### 📸 Images")
                                for img in imgs:
                                    st.image(img)
                            
                            # Get fresh analysis from Groq
                            extracted_data = local_env.get("ai_extracted_data")
                            if extracted_data is not None:
                                with st.spinner("🤖 Generating fresh analysis..."):
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
                                    st.markdown("### 💬 Fresh Analysis")
                                    st.markdown(fresh_answer)
                except Exception:
                    st.error("Lỗi khi chạy lại!")
                    st.code(traceback.format_exc(), language="python")
        
        with col2:
            if st.button("✨ Start New Chat", use_container_width=True):
                st.session_state.selected_message_idx = None
                st.session_state.pending_code = ""
                st.session_state.pending_viz_code = ""
                st.session_state.pending_question = ""
                st.rerun()
    else:
        # No history selected - blank chat for new conversation
        st.markdown("_💬 Ready for new analysis. Type your question below._")

    # ================= CHAT INPUT =================
    q = st.chat_input("Hỏi AI phân tích dữ liệu...")

    if q:
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
        st.markdown("### 🧠 AI Generated Extraction Plan")

        current_turn_id = st.session_state.pending_turn_id
        edited_code = st.text_area(
            "Extraction Code",
            value=st.session_state.pending_code,
            height=250,
            key=f"code_{current_turn_id}"
        )

        edited_viz_code = st.text_area(
            "Visualization Code",
            value=st.session_state.pending_viz_code,
            height=250,
            key=f"viz_{current_turn_id}"
        )

        if st.button("✅ Run Analysis", key=f"run_{current_turn_id}"):
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