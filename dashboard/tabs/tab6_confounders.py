# tabs/tab6_confounders.py

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from sklearn.preprocessing import StandardScaler

def _prep_confounder_data(filtered_df: pd.DataFrame):
    """Tiền xử lý các cột cần thiết cho phân tích Confounder."""
    df_c = filtered_df.copy()
    df_c = df_c.loc[:, ~df_c.columns.duplicated()]

    # Tạo target
    df_c['log_views'] = np.log1p(pd.to_numeric(df_c.get('video_view_count', np.nan), errors='coerce'))

    # Tạo các biến Binary cần thiết
    df_c['has_caption'] = (df_c.get('video_caption_status', '').astype(str).str.lower().str.strip().isin(['true', '1', 'yes'])).astype(int)
    df_c['is_licensed'] = (df_c.get('video_licensed_content', '').astype(str).str.lower().str.strip().isin(['true', '1', 'yes'])).astype(int)
    
    if 'video_publish_date' in df_c.columns:
        df_c['video_publish_date'] = pd.to_datetime(df_c['video_publish_date'], errors='coerce')
        ref_date = pd.Timestamp.today().normalize()
        df_c['video_age_days'] = (ref_date - df_c['video_publish_date']).dt.days.clip(lower=0)

    # Đảm bảo numeric
    for col in ['channel_subscriber_count', 'channel_video_count', 'video_age_days']:
        if col in df_c.columns:
            df_c[col] = pd.to_numeric(df_c[col], errors='coerce')
            df_c[col] = df_c[col].fillna(df_c[col].median())

    df_c = df_c.replace([np.inf, -np.inf], np.nan).dropna(subset=['log_views'])
    return df_c

def _run_models_for_flip(df, target, var_main, var_confounder):
    """Chạy mô hình chuẩn hóa để lấy hệ số đơn và đa biến."""
    df_valid = df[[target, var_main, var_confounder]].dropna()
    if len(df_valid) < 3:
        return None, None
        
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(df_valid[[var_main, var_confounder]]), 
                            columns=[var_main, var_confounder], index=df_valid.index)
    y = df_valid[target]

    # Mô hình đơn
    X1_c = sm.add_constant(X_scaled[[var_main]], has_constant='add')
    model_single = sm.OLS(y, X1_c).fit()
    coef_single = model_single.params.get(var_main, 0)

    # Mô hình đa
    X2_c = sm.add_constant(X_scaled[[var_main, var_confounder]], has_constant='add')
    model_multi = sm.OLS(y, X2_c).fit()
    coef_multi = model_multi.params.get(var_main, 0)
    
    return coef_single, coef_multi

def _run_models_for_r2(df, target, var1, var2):
    """Chạy mô hình để lấy R2 đơn và đa biến."""
    df_valid = df[[target, var1, var2]].dropna()
    if len(df_valid) < 3:
        return 0, 0, 0
        
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(df_valid[[var1, var2]]), 
                            columns=[var1, var2], index=df_valid.index)
    y = df_valid[target]

    m1 = sm.OLS(y, sm.add_constant(X_scaled[[var1]], has_constant='add')).fit()
    m2 = sm.OLS(y, sm.add_constant(X_scaled[[var2]], has_constant='add')).fit()
    m_both = sm.OLS(y, sm.add_constant(X_scaled[[var1, var2]], has_constant='add')).fit()
    
    return m1.rsquared, m2.rsquared, m_both.rsquared

def render_tab(filtered_df: pd.DataFrame):
    st.subheader("Phân tích Yếu tố Gây nhiễu & Cộng hưởng")
    
    df_model = _prep_confounder_data(filtered_df)

    if len(df_model) < 50:
        st.warning("Dữ liệu hiện tại quá ít để phân tích Confounder chính xác. Hãy mở rộng bộ lọc.")
        return

    sub6a, sub6b = st.tabs([
        "A: Hiện tượng Đảo Dấu (Confounding)",
        "B: Cặp Biến Cộng Hưởng (Synergy R²)"
    ])

    # =================================================
    # SUB-TAB 6A: SIMPSON'S PARADOX
    # =================================================
    with sub6a:
        st.markdown("### 🎭 Kẻ Hai Mặt: Nghịch lý Simpson (Đảo dấu)")
        st.info(
            "**Nghịch lý Simpson** xảy ra khi một biến có vẻ tác động tích cực đến lượt xem (đứng một mình), "
            "nhưng khi ghép chung với biến kiểm soát quy mô (Confounder), bản chất tiêu cực của nó mới lộ diện."
        )

        st.markdown("#### Ví dụ kinh điển: Số lượng Video vs Quy mô Kênh")
        st.caption(
            "Phân tích hệ số của biến **Số lượng Video (channel_video_count)** lên Lượt xem, "
            "trước và sau khi đưa **Quy mô Kênh (channel_subscriber_count)** vào kiểm soát."
        )

        coef_single, coef_multi = _run_models_for_flip(
            df_model, 'log_views', 'channel_video_count', 'channel_subscriber_count'
        )

        if coef_single is not None:
            col1, col2, col3 = st.columns(3)
            col1.metric("Hệ số khi đứng 1 mình", f"{coef_single:.4f}", "Đánh lừa")
            col2.metric("Hệ số khi bị kiểm soát", f"{coef_multi:.4f}", "Sự thật", delta_color="inverse")
            
            flip_detected = np.sign(coef_single) != np.sign(coef_multi)
            if flip_detected:
                col3.error("🔴 PHÁT HIỆN ĐẢO DẤU!")
            else:
                col3.success("➖ Không bị đảo dấu trong tệp dữ liệu này")

            # Vẽ biểu đồ Bar Chart
            fig_flip = go.Figure()
            fig_flip.add_trace(go.Bar(
                x=['Đứng một mình (Đơn biến)', 'Ghép với Quy mô Kênh (Đa biến)'],
                y=[coef_single, coef_multi],
                marker_color=['#2ecc71' if coef_single > 0 else '#e74c3c', 
                              '#2ecc71' if coef_multi > 0 else '#e74c3c'],
                text=[f"{coef_single:.4f}", f"{coef_multi:.4f}"],
                textposition='auto'
            ))
            fig_flip.add_hline(y=0, line_dash='solid', line_color='black', line_width=2)
            fig_flip.update_layout(
                title='Sự thay đổi Hệ số của "Số lượng Video" tác động lên Lượt xem',
                yaxis_title='Hệ số Chuẩn hóa (Std Coef)',
                height=400
            )
            st.plotly_chart(fig_flip, use_container_width=True)

            st.markdown(
                "> **💡 Insight Thực Chiến:** Đừng lầm tưởng đăng càng nhiều video thì view càng cao. "
                "Thực chất, các kênh lớn thường đăng nhiều, nhưng nếu xét hai kênh có cùng lượng Sub, "
                "kênh nào **spam quá nhiều video** sẽ làm **GIẢM** lượt xem trung bình của từng video. "
                "*(Chất lượng quan trọng hơn Số lượng!)*"
            )

    # =================================================
    # SUB-TAB 6B: SYNERGISTIC PAIRS
    # =================================================
    with sub6b:
        st.markdown("### 🤝 Sức mạnh Cặp đôi (R² Synergy)")
        st.info(
            "Những biến khi đứng một mình có thể giải thích dữ liệu tốt, nhưng khi **ghép chung với nhau**, "
            "chúng không hề 'giẫm chân nhau' mà cộng hưởng để tạo ra khả năng giải thích R² tăng vọt."
        )

        # Các cặp mặc định từ insight
        pair_options = {
            "Bản quyền + Phụ đề (Combo Chuẩn SEO)": ['is_licensed', 'has_caption'],
            "Bản quyền + Tuổi thọ (Nhạc lâu năm)": ['is_licensed', 'video_age_days'],
            "Quy mô Kênh + Phụ đề (Tiếp cận quốc tế)": ['channel_subscriber_count', 'has_caption']
        }

        selected_pair_name = st.selectbox("Chọn Cặp biến để phân tích:", list(pair_options.keys()))
        var1, var2 = pair_options[selected_pair_name]

        if var1 in df_model.columns and var2 in df_model.columns:
            r2_1, r2_2, r2_both = _run_models_for_r2(df_model, 'log_views', var1, var2)
            max_single = max(r2_1, r2_2)
            r2_jump = r2_both - max_single

            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            col_r1.metric(f"R² ({var1})", f"{r2_1:.4f}")
            col_r2.metric(f"R² ({var2})", f"{r2_2:.4f}")
            col_r3.metric("R² (Khi Ghép chung)", f"{r2_both:.4f}", f"+{r2_jump:.4f} so với mức cao nhất")

            if r2_jump > 0.02:
                col_r4.success("🌟 Cộng hưởng Rất Tốt")
            elif r2_jump > 0.005:
                col_r4.info("✅ Cộng hưởng Khá")
            else:
                col_r4.warning("⚠️ Giẫm chân nhau (Trùng lặp thông tin)")

            # Biểu đồ Waterfall (Thác nước) thể hiện sự gia tăng
            fig_waterfall = go.Figure(go.Waterfall(
                name="20", orientation="v",
                measure=["relative", "relative", "total"],
                x=[f"Chỉ dùng {var1}", f"Thêm {var2}", "Tổng sức mạnh (R² Cặp)"],
                textposition="outside",
                text=[f"{r2_1:.4f}", f"+{r2_both - r2_1:.4f}", f"{r2_both:.4f}"],
                y=[r2_1, r2_both - r2_1, r2_both],
                connector={"line":{"color":"rgb(63, 63, 63)"}},
                decreasing={"marker":{"color":"#e74c3c"}},
                increasing={"marker":{"color":"#3498db"}},
                totals={"marker":{"color":"#2ecc71"}}
            ))

            fig_waterfall.update_layout(
                title=f"Sức mạnh giải thích (R²) tăng lên khi kết hợp: {var1} + {var2}",
                yaxis_title="R-Squared",
                height=450,
                showlegend=False
            )
            st.plotly_chart(fig_waterfall, use_container_width=True)
            
            st.markdown(f"> **💡 Khuyến nghị Model:** Cặp `{var1}` và `{var2}` là mảnh ghép bổ sung hoàn hảo, nên cùng xuất hiện trong mô hình tối ưu.")
        else:
            st.error("Dữ liệu hiện tại bị thiếu biến để phân tích cặp này.")