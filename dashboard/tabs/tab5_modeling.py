import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st


def render_tab(filtered_df):
    st.subheader("Hồi quy từng bước")

    features = [
        'video_tags_count',
        'channel_subscriber_count',
        'title_length',
        'channel_view_count',
        'channel_video_count'
    ]

    df_model = filtered_df.copy()
    df_model = df_model.replace([np.inf, -np.inf], np.nan)

    for col in features:
        df_model[col] = df_model[col].fillna(df_model[col].median())

    df_model = df_model.dropna(subset=['video_view_count'])
    if len(df_model) < 10:
        st.warning("⚠️ Dữ liệu hiện tại quá ít (ít hơn 10 video) để chạy mô hình Hồi quy đáng tin cậy. Các Cross-filter bạn đang chọn đã thu hẹp dữ liệu. Vui lòng nhấn nút 'Bỏ' bớt bộ lọc ở Sidebar để xem Tab này.")
        return

    def stepwise(data, target, feats):
        selected = []
        log = []

        data = data.copy()
        for c in list(feats) + [target]:
            if c in data.columns:
                data[c] = pd.to_numeric(data[c], errors='coerce')

        data = data.replace([np.inf, -np.inf], np.nan)

        while feats:
            scores = []
            for f in feats:
                cols = selected + [f]
                X = pd.DataFrame(sm.add_constant(data[cols], has_constant='add'), index=data.index)
                y = data[target]
                mask = X.notna().all(axis=1) & y.notna()
                if mask.sum() == 0:
                    scores.append((f, np.inf))
                    continue
                try:
                    model_try = sm.OLS(y[mask], X[mask]).fit()
                    pval = model_try.pvalues.get(f, np.inf)
                except Exception:
                    pval = np.inf
                scores.append((f, pval))

            best = min(scores, key=lambda x: x[1])

            if best[1] < 0.05:
                selected.append(best[0])
                feats.remove(best[0])
                log.append(f"Added {best[0]} (p={best[1]:.4f})")
            else:
                break

        if selected:
            X_final = pd.DataFrame(sm.add_constant(data[selected], has_constant='add'), index=data.index)
        else:
            X_final = pd.DataFrame({'const': np.ones(len(data))}, index=data.index)

        y_final = data[target]
        mask_final = X_final.notna().all(axis=1) & y_final.notna()
        if mask_final.sum() == 0:
            final_model = None
        else:
            final_model = sm.OLS(y_final[mask_final], X_final[mask_final]).fit()

        return selected, final_model, log

    selected, model, log = stepwise(df_model, 'video_view_count', features.copy())

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Biến được chọn")
        for l in log:
            st.code(l)

    with col2:
        st.write("### Tóm tắt mô hình")
        if model is None:
            st.info("Không đủ dữ liệu để ước lượng mô hình.")
        else:
            st.text(model.summary().tables[1])
