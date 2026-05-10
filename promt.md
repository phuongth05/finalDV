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

---

## 2. Prompting Visualization Design

Yêu cầu:
- Ưu tiên Plotly interactive charts.
- Hỗ trợ cross-filtering.
- Hỗ trợ drill-down analysis.
- Visualization phải dễ interpret.

Visualization cần đề xuất:
- KPI cards
- Histogram
- Boxplot
- Scatter plot
- Heatmap
- Correlation matrix
- Timeline chart
- Wordcloud
- Regression plot

Mỗi visualization cần giải thích:
- mục tiêu sử dụng,
- insight có thể rút ra,
- interaction hỗ trợ.

---

## 3. Prompting VIF và Stepwise Regression

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

---

## 4. Prompting Confounder và Simpson's Paradox

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

---

## 5. Prompting Phát Triển Hệ Thống AI Analytics

Mục tiêu:
- Xây dựng AI analytics system bằng Streamlit.
- Tự động extract insight.
- Lưu context vào context.json.
- Hỗ trợ AI chatbot reasoning.

Context extraction:

### Tab 1 — Tổng quan
- KPI
- phân bố dữ liệu
- outlier
- top entities
- trend

### Tab 2 — Định nghĩa thành công
- correlation
- heatmap
- success zone
- viral vs flop
- top 20% vs bottom 20%

### Tab 3 — Hành vi người dùng
- supply-demand
- caption impact
- audience segments
- keyword trends

### Tab 4 — Thuật toán nền tảng
- upload timing
- duration optimization
- SEO keyword
- emotional keyword
- metadata effectiveness

### Tab 5 — Modeling
- correlation matrix
- VIF
- selected variables
- removed variables
- regression interpretation

### Tab 6 — Confounding
- Simpson's paradox
- sign flipping
- synergy
- hidden relationships
- confounder variables

Output:
- context.json
- summarized insights
- AI-readable analytical memory