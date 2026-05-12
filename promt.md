# AI Prompt Engineering for Analytics System

## 1. Prompting Design Hệ Thống

Mục tiêu:
- Thiết kế dashboard Streamlit cho phân tích dữ liệu Youtube Việt Nam.
- Hỗ trợ exploratory data analysis.
- Tích hợp AI insight generation.
- Tối ưu interactive visualization.
- Hỗ trợ mở rộng chatbot AI trong tương lai.

Input:
- Youtube Vietnam API dataset.
- Các cột gồm:
  - video_id
  - title
  - channel_name
  - publish_date
  - duration
  - views
  - likes
  - comments
  - tags
  - caption
  - category
  - licensed_content
  - upload_hour
  - upload_day

Yêu cầu:
1. Đề xuất cấu trúc dashboard phù hợp.
2. Xác định số lượng tab tối ưu.
3. Đề xuất visualization cho từng nhóm phân tích.
4. Đề xuất các business question quan trọng.
5. Thiết kế các mode hoạt động:
   - Dashboard mode
   - AI insight mode
   - Statistical analysis mode
   - Regression analysis mode
   - Confounder analysis mode

Ví dụ output dạng code để AI bám theo:
```python
import numpy as np
import pandas as pd
import streamlit as st

df = pd.read_csv("data/youtube_vn_music_cleaned.csv")
df["_row_id"] = df.index.astype(str)
df["video_publish_date"] = pd.to_datetime(df["video_publish_date"], errors="coerce")
df["hour"] = df["video_publish_date"].dt.hour
df["day"] = df["video_publish_date"].dt.day_name()
df["publish_date"] = df["video_publish_date"].dt.date

df["title_length"] = df["video_title"].astype(str).apply(len)
df["engagement_rate"] = (
    (pd.to_numeric(df["video_like_count"], errors="coerce")
     + pd.to_numeric(df["video_comment_count"], errors="coerce"))
    / pd.to_numeric(df["video_view_count"], errors="coerce")
).replace([np.inf, -np.inf], np.nan)

st.sidebar.markdown("## Bộ lọc chính")
date_range = st.sidebar.date_input(
    "Khoảng ngày",
    [df["video_publish_date"].min().date(), df["video_publish_date"].max().date()]
)
view_range = st.sidebar.slider("Khoảng lượt xem", 0, int(df["video_view_count"].max()), (0, int(df["video_view_count"].max())))

mask = (
    df["publish_date"].between(date_range[0], date_range[1])
    & df["video_view_count"].between(view_range[0], view_range[1])
)
filtered_df = df.loc[mask].copy()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview", "Success", "Audience", "Timing", "Modeling", "Confounders", "AI Assistant"
])
```

---


## 2. Prompting VIF và Stepwise Regression

Mục tiêu:
- Tự động interpret kết quả thống kê.

Input:
- Correlation matrix
- VIF analysis
- Stepwise Regression
- Regression coefficients
- p-values
- Adjusted R²

Yêu cầu:
1. Phát hiện multicollinearity.
2. Giải thích biến bị loại bỏ.
3. Phân tích selected features.
4. Giải thích feature importance.
5. Đánh giá độ ổn định mô hình.
6. Sinh business insights.

Ví dụ output dạng code để AI bám theo:
```python
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from statsmodels.stats.outliers_influence import variance_inflation_factor

NUMERIC_COLS = [
    "video_duration",
    "video_tags_count",
    "title_length",
    "description_length",
    "channel_subscriber_count",
    "channel_view_count",
    "channel_video_count",
    "hour",
    "video_age_days",
]

num_exist = [c for c in NUMERIC_COLS if c in df_model.columns]
corr_cols = [c for c in num_exist + ["video_view_count"] if c in df_model.columns]
corr_matrix = df_model[corr_cols].corr()

X_vif = df_model[num_exist].dropna()
non_const = [c for c in num_exist if c in X_vif.columns and X_vif[c].std() > 1e-10]
X_vif = X_vif[non_const]

vif_rows = []
for i, col in enumerate(non_const):
    v = variance_inflation_factor(X_vif.values, i)
    vif_rows.append({"Biến": col, "VIF": round(float(v), 2)})

vif_data = pd.DataFrame(vif_rows)
fig_vif = px.bar(vif_data.sort_values("VIF"), x="VIF", y="Biến", orientation="h")
st.plotly_chart(fig_vif, use_container_width=True)
st.dataframe(vif_data, hide_index=True)
```

---

## 3. Prompting Confounder và Simpson's Paradox

Input:
- Regression output
- Correlation matrix
- Subgroup analysis
- Coefficient changes
- Feature interaction

Yêu cầu:
1. Tìm confounder variables.
2. Phát hiện sign flipping.
3. Phát hiện Simpson's Paradox.
4. Giải thích hidden relationships.
5. Xác định misleading correlations.
6. Đưa ra statistical interpretation.

Ví dụ output dạng code để AI bám theo:
```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from sklearn.preprocessing import StandardScaler

def _run_models_for_flip(df, target, var_main, var_confounder):
  need_cols = [target, var_main, var_confounder]
  for c in need_cols:
    if c not in df.columns:
      return None, None

  df_valid = df[need_cols].replace([np.inf, -np.inf], np.nan).dropna()
  if len(df_valid) < 10:
    return None, None

  scaler = StandardScaler()
  X_scaled = pd.DataFrame(
    scaler.fit_transform(df_valid[[var_main, var_confounder]]),
    columns=[var_main, var_confounder],
    index=df_valid.index,
  )
  y = df_valid[target]

  X1 = sm.add_constant(X_scaled[[var_main]], has_constant="add")
  m1 = sm.OLS(y, X1).fit()

  X2 = sm.add_constant(X_scaled[[var_main, var_confounder]], has_constant="add")
  m2 = sm.OLS(y, X2).fit()

  return float(m1.params.get(var_main, 0)), float(m2.params.get(var_main, 0))

single_coef, multi_coef = _run_models_for_flip(
  df_model,
  "log_views",
  "video_duration",
  "channel_subscriber_count",
)

st.write({
  "single_coef": single_coef,
  "multi_coef": multi_coef,
  "sign_flip": bool(single_coef is not None and multi_coef is not None and np.sign(single_coef) != np.sign(multi_coef)),
})
```

---

## 4. Prompting Phát Triển Hệ Thống AI Analytics

Mục tiêu:
- Xây dựng AI analytics system bằng Streamlit.
- Tự động extract insight.
- Lưu context vào context.json.
- Hỗ trợ AI chatbot reasoning.

Mục tiêu prompt:
- Tóm tắt schema thật của `filtered_df`.
- Cung cấp bản đồ cột an toàn để AI không bịa cột.
- Sinh code trích xuất và code trực quan tách riêng.
- Nếu code lỗi, trả về lỗi ngắn gọn + prompt sửa tối thiểu.

Ví dụ output dạng code để AI bám theo:
```python
import json
import re
import traceback
import streamlit as st

def _build_column_usage_guide(filtered_df):
  available = set(filtered_df.columns)
  lines = ["Bản đồ cột thực tế và cách dùng an toàn:"]
  if "video_view_count" in available:
    lines.append("- video_view_count: cột trung tâm cho phân bố, top/bottom, trend, tương quan, modeling.")
  if {"video_like_count", "video_comment_count"}.issubset(available):
    lines.append("- video_like_count, video_comment_count: dùng cho engagement_rate và success analysis.")
  if {"video_publish_date", "hour", "day"}.intersection(available):
    lines.append("- video_publish_date, hour, day: dùng cho timing, heatmap, trend theo thời gian.")
  if {"channel_title", "channel_subscriber_count", "channel_size"}.intersection(available):
    lines.append("- channel_title, channel_subscriber_count, channel_size: dùng cho quy mô kênh và confounder.")
  return "\n".join(lines)

def _build_code_prompt(question, filtered_df, dataframe_profile):
  return f"""
Ngữ cảnh dữ liệu:
- filtered_df là DataFrame cuối cùng sau filter trong dashboard/app.py.
- Không giả định có cột ngoài schema thực tế.
- Nếu cần cột chưa chắc chắn, phải kiểm tra bằng if col in filtered_df.columns.
- Khi dùng số, ưu tiên pd.to_numeric(..., errors='coerce').
- Khi dùng ngày giờ, ưu tiên pd.to_datetime(..., errors='coerce').
- Nếu dữ liệu không đủ, trả về ai_extracted_data là dict mô tả thiếu gì.

Schema thực tế:
{", ".join(filtered_df.columns.tolist())}

{_build_column_usage_guide(filtered_df)}

Tóm tắt dataframe:
{json.dumps(dataframe_profile, ensure_ascii=False, indent=2, default=str)}

Câu hỏi:
"{question}"

Nhiệm vụ:
1. Trả về đúng 2 khối code Python.
2. Khối 1: trích xuất dữ liệu và lưu vào ai_extracted_data.
3. Khối 2: vẽ ai_figures / ai_images nếu cần.
4. Không giải thích, không markdown ngoài code fence.
"""

def _build_answer_prompt(question, extracted_data, recent_dialogue=""):
  return f"""
Bạn là Data Analyst chuyên nghiệp.
Người dùng hỏi: "{question}"
Dữ liệu đã trích xuất: {extracted_data}

Hãy trả lời theo 4 phần:
1. Nhận xét biểu đồ.
2. Dẫn dắt câu chuyện.
3. Liên kết biểu đồ nếu có.
4. Kết luận cuối cùng.
"""

def _extract_runtime_error_excerpt(exc):
  tb = traceback.extract_tb(exc.__traceback__) if exc.__traceback__ else []
  last = tb[-1] if tb else None
  if last:
    return f"File: {last.filename}\nLine {last.lineno}:\n{(last.line or '').strip()}\n\n{type(exc).__name__}: {exc}"
  return f"{type(exc).__name__}: {exc}"

def _build_retry_fix_prompt(code, concise_error, context="", stage="code"):
  return f"""
Đây là code Python bị lỗi ở phần {stage}.

Code:
{code}

Lỗi:
{concise_error}

Ngữ cảnh:
{context}

Hãy sửa tối thiểu để code chạy được.
Không rewrite toàn bộ.
Giữ nguyên logic cũ.
Chỉ trả về code Python.
"""
```