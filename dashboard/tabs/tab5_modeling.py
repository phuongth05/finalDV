# tabs/tab5_modeling.py

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

NUMERIC_COLS = [
    'video_duration',
    'video_tags_count',
    'title_length',
    'description_length',
    'channel_subscriber_count',
    'channel_view_count',
    'channel_video_count',
    'hour',
]

BINARY_COLS = [
    'is_hd',
    'has_caption',
    'is_licensed',
    'is_for_kids',
    'is_official',
    'is_remix',
    'is_lyric',
    'is_cover',
    'is_collab',
    'is_weekend',
    'posted_morning',
    'posted_afternoon',
    'posted_evening',
]


def _build_df_model(filtered_df: pd.DataFrame):
    """Chuẩn bị df_model từ filtered_df."""
    df_model = filtered_df.copy()

    # Xóa duplicate columns
    df_model = df_model.loc[:, ~df_model.columns.duplicated()]

    # ── Binary từ cột có sẵn ────────────────────────
    df_model['is_hd'] = (
        df_model['video_definition'].astype(str)
        .str.lower().str.strip() == 'hd'
    ).astype(int)

    df_model['has_caption'] = (
        df_model['video_caption_status'].astype(str)
        .str.lower().str.strip()
        .isin(['true', '1', 'yes'])
    ).astype(int)

    df_model['is_licensed'] = (
        df_model['video_licensed_content'].astype(str)
        .str.lower().str.strip()
        .isin(['true', '1', 'yes'])
    ).astype(int)

    df_model['is_for_kids'] = (
        df_model['video_made_for_kids'].astype(str)
        .str.lower().str.strip()
        .isin(['true', '1', 'yes'])
    ).astype(int)

    # ── Binary từ title ──────────────────────────────
    df_model['is_official'] = (
        df_model['video_title'].astype(str).str.lower()
        .str.contains('official|chính thức', regex=True)
    ).astype(int)

    df_model['is_remix'] = (
        df_model['video_title'].astype(str).str.lower()
        .str.contains('remix', regex=False)
    ).astype(int)

    df_model['is_lyric'] = (
        df_model['video_title'].astype(str).str.lower()
        .str.contains('lyric', regex=False)
    ).astype(int)

    df_model['is_cover'] = (
        df_model['video_title'].astype(str).str.lower()
        .str.contains('cover', regex=False)
    ).astype(int)

    df_model['is_collab'] = (
        df_model['video_title'].astype(str).str.lower()
        .str.contains(r'ft\.|feat\.| x ', regex=True)
    ).astype(int)

    # ── Thời gian ────────────────────────────────────
    df_model['is_weekend'] = (
        df_model['video_publish_date'].dt.dayofweek >= 5
    ).astype(int)

    df_model['posted_morning'] = (
        df_model['hour'].between(6, 11)
    ).astype(int)

    df_model['posted_afternoon'] = (
        df_model['hour'].between(12, 17)
    ).astype(int)

    df_model['posted_evening'] = (
        df_model['hour'].between(18, 22)
    ).astype(int)

    

    
    # ── Genre one-hot ────────────────────────────────
    # Xóa cột genre_ cũ nếu có
    old_genre_cols = [
        c for c in df_model.columns
        if c.startswith('genre_')
    ]
    df_model = df_model.drop(
        columns=old_genre_cols, errors='ignore'
    )

    genre_dummies = pd.get_dummies(
        df_model['genre'], prefix='genre', drop_first=True
    ).astype(int)
    df_model = pd.concat([df_model, genre_dummies], axis=1)
    genre_dummy_cols = list(genre_dummies.columns)

    # Xóa duplicate lần nữa sau concat
    df_model = df_model.loc[:, ~df_model.columns.duplicated()]

    # ── Text features ────────────────────────────────
    df_model['title_length'] = (
        df_model['video_title'].astype(str).str.len()
    )
    df_model['description_length'] = (
        df_model['video_description'].astype(str).str.len()
    )

    # ── Log transform ────────────────────────────────
    df_model['log_views'] = np.log1p(
        pd.to_numeric(df_model['video_view_count'], errors='coerce')
    )
    df_model['log_likes'] = np.log1p(
        pd.to_numeric(df_model['video_like_count'], errors='coerce')
    )

    # ── Engagement rate ──────────────────────────────
    _views = pd.to_numeric(df_model['video_view_count'], errors='coerce')
    _likes = pd.to_numeric(df_model['video_like_count'], errors='coerce')
    _comments = pd.to_numeric(df_model['video_comment_count'], errors='coerce')
    df_model['engagement_rate'] = (
        (_likes + _comments) / _views
    ).replace([np.inf, -np.inf], np.nan)

    # ── Clean ────────────────────────────────────────
    df_model = df_model.replace([np.inf, -np.inf], np.nan)

    for col in NUMERIC_COLS:
        if col in df_model.columns:
            df_model[col] = pd.to_numeric(
                df_model[col], errors='coerce'
            )
            df_model[col] = df_model[col].fillna(
                df_model[col].median()
            )

    for col in BINARY_COLS + genre_dummy_cols:
        if col in df_model.columns:
            df_model[col] = df_model[col].fillna(0)

    df_model = df_model.dropna(subset=['video_view_count'])

    return df_model, genre_dummy_cols


def _stepwise_selection(
    data, target, feats,
    direction='both',
    sl_enter=0.05,
    sl_remove=0.10
):
    """Thực hiện stepwise regression."""
    included = []
    history = []

    data = data.copy()
    data = data.loc[:, ~data.columns.duplicated()]

    feats = [f for f in feats if f in data.columns]
    remaining = list(feats)

    for c in feats + [target]:
        if c not in data.columns:
            continue
        col_data = data[c]
        if isinstance(col_data, pd.DataFrame):
            col_data = col_data.iloc[:, 0]
        data[c] = pd.to_numeric(col_data, errors='coerce')

    data = data.replace([np.inf, -np.inf], np.nan)

    cols_to_check = [c for c in ([target] + feats) if c in data.columns]
    data = data.dropna(subset=cols_to_check)

    if len(data) < len(feats) + 2:
        return [], None, pd.DataFrame()

    def get_r2(inc):
        if not inc:
            return 0.0
        try:
            Xt = sm.add_constant(data[inc], has_constant='add')
            yt = data[target]
            mk = Xt.notna().all(axis=1) & yt.notna()
            if mk.sum() < len(inc) + 2:
                return 0.0
            return sm.OLS(yt[mk], Xt[mk]).fit().rsquared
        except Exception:
            return 0.0

    if direction in ['forward', 'both']:
        step = 0
        while remaining:
            step += 1
            best_p = sl_enter
            best_f = None

            for f in remaining:
                cols = included + [f]
                try:
                    X = sm.add_constant(data[cols], has_constant='add')
                    y = data[target]
                    mk = X.notna().all(axis=1) & y.notna()
                    if mk.sum() < len(cols) + 2:
                        continue
                    m = sm.OLS(y[mk], X[mk]).fit()
                    p = m.pvalues.get(f, np.inf)
                except Exception:
                    p = np.inf

                if p < best_p:
                    best_p = p
                    best_f = f

            if best_f is None:
                break

            included.append(best_f)
            remaining.remove(best_f)
            history.append({
                'Step': step,
                'Action': f'➕ {best_f}',
                'p-value': round(best_p, 4),
                'R²': round(get_r2(included), 4),
                'Số biến': len(included)
            })

            if direction == 'both' and len(included) > 1:
                try:
                    Xc = sm.add_constant(data[included], has_constant='add')
                    yc = data[target]
                    mkc = Xc.notna().all(axis=1) & yc.notna()
                    mc = sm.OLS(yc[mkc], Xc[mkc]).fit()
                    pv = mc.pvalues.iloc[1:]
                    wp = pv.max()
                    if wp > sl_remove:
                        wf = pv.idxmax()
                        included.remove(wf)
                        remaining.append(wf)
                        step += 1
                        history.append({
                            'Step': step,
                            'Action': f'➖ {wf}',
                            'p-value': round(wp, 4),
                            'R²': round(get_r2(included), 4),
                            'Số biến': len(included)
                        })
                except Exception:
                    pass

    elif direction == 'backward':
        included = list(feats)
        step = 0
        while len(included) > 1:
            step += 1
            try:
                X = sm.add_constant(data[included], has_constant='add')
                y = data[target]
                mk = X.notna().all(axis=1) & y.notna()
                m = sm.OLS(y[mk], X[mk]).fit()
                pv = m.pvalues.iloc[1:]
                wp = pv.max()
                if wp <= sl_enter:
                    break
                wf = pv.idxmax()
                included.remove(wf)
                history.append({
                    'Step': step,
                    'Action': f'➖ {wf}',
                    'p-value': round(wp, 4),
                    'R²': round(get_r2(included), 4),
                    'Số biến': len(included)
                })
            except Exception:
                break

    final_model = None
    if included:
        try:
            Xf = sm.add_constant(data[included], has_constant='add')
            yf = data[target]
            mkf = Xf.notna().all(axis=1) & yf.notna()
            if mkf.sum() > len(included) + 1:
                final_model = sm.OLS(yf[mkf], Xf[mkf]).fit()
        except Exception:
            pass

    return included, final_model, pd.DataFrame(history)


def render_tab(filtered_df: pd.DataFrame):
    """Hàm render chính được gọi từ app.py."""

    st.subheader("Stepwise Regression")

    # Build df_model
    df_model, genre_dummy_cols = _build_df_model(filtered_df)

    # 3 sub-tabs
    sub4a, sub4b = st.tabs([
        "A: Correlation & VIF",
        "B: Stepwise Regression",
    ])

    # =================================================
    # SUB-TAB 4A
    # =================================================
    with sub4a:
        st.markdown("### 📊 Correlation & VIF")
        st.caption(
            "Phân tích các biến **số** để phát hiện "
            "đa cộng tuyến. Kết quả giúp quyết định "
            "loại biến nào trước khi đưa vào 4B."
        )

        num_exist = [c for c in NUMERIC_COLS if c in df_model.columns]

        # Correlation Matrix
        st.markdown("#### 🔥 Correlation Matrix")
        corr_cols = [
            c for c in num_exist + ['video_view_count']
            if c in df_model.columns
        ]
        corr_matrix = df_model[corr_cols].corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            aspect='auto',
            title='Ma trận tương quan – Biến Numeric'
        )
        fig_corr.update_layout(height=550)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Cặp tương quan cao
        st.markdown("#### ⚠️ Cặp biến tương quan cao (|r| > 0.7)")
        high_corr_pairs = []
        cols_list = corr_matrix.columns.tolist()
        for i in range(len(cols_list)):
            for j in range(i + 1, len(cols_list)):
                r = corr_matrix.iloc[i, j]
                if abs(r) > 0.7:
                    high_corr_pairs.append({
                        'Biến 1': cols_list[i],
                        'Biến 2': cols_list[j],
                        'r': round(r, 4),
                        'Mức độ': '🔴 Rất cao' if abs(r) > 0.85 else '🟠 Cao',
                        'Khuyến nghị': 'Cân nhắc loại 1 trong 2 ở 4B'
                    })

        if high_corr_pairs:
            st.dataframe(
                pd.DataFrame(high_corr_pairs),
                hide_index=True,
                use_container_width=True
            )
        else:
            st.success("Không có cặp biến nào có |r| > 0.7")

        # VIF
        st.markdown("---")
        st.markdown("#### 📐 VIF – Variance Inflation Factor")
        st.info(
            "- VIF = 1: Hoàn toàn độc lập ✅  \n"
            "- VIF < 5: Chấp nhận được ✅  \n"
            "- 5 ≤ VIF < 10: Cần xem xét ⚠️  \n"
            "- VIF ≥ 10: Đa cộng tuyến nghiêm trọng ❌"
        )

        X_vif = df_model[num_exist].dropna()
        non_const = [c for c in num_exist if X_vif[c].std() > 1e-10]
        X_vif = X_vif[non_const]

        vif_rows = []
        for i, col in enumerate(non_const):
            try:
                v = variance_inflation_factor(X_vif.values, i)
                vif_rows.append({'Biến': col, 'VIF': round(float(v), 2)})
            except Exception:
                vif_rows.append({'Biến': col, 'VIF': None})

        vif_data = pd.DataFrame(vif_rows)

        def classify_vif(v):
            if v is None: return '⚠️ Lỗi'
            if v < 5: return '✅ Tốt'
            elif v < 10: return '⚠️ Cần xem xét'
            else: return '❌ Nghiêm trọng'

        vif_data['Mức độ'] = vif_data['VIF'].apply(classify_vif)
        vif_data = vif_data.sort_values('VIF', ascending=False, na_position='last')

        col_v1, col_v2 = st.columns([3, 2])
        with col_v1:
            vif_plot = vif_data.dropna(subset=['VIF']).sort_values('VIF')
            colors_vif = [
                '#2ecc71' if v < 5 else '#f39c12' if v < 10 else '#e74c3c'
                for v in vif_plot['VIF']
            ]
            fig_vif = go.Figure(go.Bar(
                x=vif_plot['VIF'].values,
                y=vif_plot['Biến'].values,
                orientation='h',
                marker_color=colors_vif,
                text=vif_plot['VIF'].round(2).values,
                textposition='outside'
            ))
            fig_vif.add_vline(x=5, line_dash='dash', line_color='orange', annotation_text='VIF = 5')
            fig_vif.add_vline(x=10, line_dash='dash', line_color='red', annotation_text='VIF = 10')
            fig_vif.update_layout(title='VIF – Biến Numeric', xaxis_title='VIF Score', height=420)
            st.plotly_chart(fig_vif, use_container_width=True)

        with col_v2:
            st.dataframe(vif_data, hide_index=True, height=380)

        st.markdown("#### 📌 Nhận xét & Gợi ý cho 4B")

        high_vif_list = vif_data[vif_data['VIF'].notna() & (vif_data['VIF'] >= 10)]['Biến'].tolist()
        med_vif_list = vif_data[vif_data['VIF'].notna() & (vif_data['VIF'] >= 5) & (vif_data['VIF'] < 10)]['Biến'].tolist()
        ok_vif_list = vif_data[vif_data['VIF'].notna() & (vif_data['VIF'] < 5)]['Biến'].tolist()

        if high_vif_list:
            st.error(f"❌ **VIF ≥ 10 – nên loại ở 4B:** `{'`, `'.join(high_vif_list)}`")
        if med_vif_list:
            st.warning(f"⚠️ **VIF 5–10 – cần xem xét:** `{'`, `'.join(med_vif_list)}`")
        if ok_vif_list:
            st.success(f"✅ **VIF < 5 – an toàn:** `{'`, `'.join(ok_vif_list)}`")

        st.session_state['high_vif_vars'] = high_vif_list
        st.session_state['ok_vif_vars'] = ok_vif_list

    # =================================================
    # SUB-TAB 4B
    # =================================================
    with sub4b:
        st.markdown("### 🔄 Stepwise Regression")
        st.info(
            "Dùng cả biến **numeric** (đã lọc VIF) "
            "và biến **encoding** từ các cột categorical có sẵn."
        )

        high_vif_from_4a = st.session_state.get('high_vif_vars', [])
        if high_vif_from_4a:
            st.warning(
                f"💡 **Từ 4A:** Biến VIF cao đã bỏ chọn mặc định: "
                f"`{'`, `'.join(high_vif_from_4a)}`"
            )

        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            target_var = st.selectbox(
                "Biến phụ thuộc (Y)",
                ['log_views', 'video_view_count', 'log_likes'],
                index=0,
                help="Khuyến nghị log_views"
            )
        with col_c2:
            method = st.selectbox(
                "Phương pháp",
                ['forward', 'backward', 'both'],
                index=2
            )
        with col_c3:
            p_enter = st.number_input(
                "Ngưỡng p-value",
                value=0.05, min_value=0.01, max_value=0.10, step=0.01
            )

        st.markdown("#### Chọn biến đưa vào Stepwise")
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.markdown("**📐 Biến Numeric**")
            num_exist_4b = [c for c in NUMERIC_COLS if c in df_model.columns]
            num_default = [c for c in num_exist_4b if c not in high_vif_from_4a]
            selected_numeric = st.multiselect(
                "Chọn biến numeric",
                options=num_exist_4b,
                default=num_default
            )

        with col_g2:
            st.markdown("**🔘 Biến Encoding**")
            bin_exist = [c for c in BINARY_COLS if c in df_model.columns]
            genre_exist = [c for c in genre_dummy_cols if c in df_model.columns]

            use_binary = st.checkbox(
                f"Binary ({len(bin_exist)} biến: is_hd, is_official...)",
                value=True
            )
            use_genre = st.checkbox(
                f"Genre one-hot ({len(genre_exist)} biến)",
                value=True
            )

            selected_encoded = []
            if use_binary:
                selected_encoded += bin_exist
            if use_genre:
                selected_encoded += genre_exist

        final_pool = [
            c for c in (selected_numeric + selected_encoded)
            if c in df_model.columns
        ]

        st.caption(
            f"✅ **{len(final_pool)} biến** sẽ vào Stepwise | "
            f"Numeric: {len(selected_numeric)} | "
            f"Binary: {len(bin_exist) if use_binary else 0} | "
            f"Genre: {len(genre_exist) if use_genre else 0}"
        )

        if st.button("▶️ Chạy Stepwise", type="primary"):
            if not final_pool:
                st.error("Vui lòng chọn ít nhất 1 biến!")
            else:
                with st.spinner("Đang chạy Stepwise..."):
                    selected, model, hist_df = _stepwise_selection(
                        df_model, target_var, final_pool,
                        direction=method,
                        sl_enter=p_enter,
                        sl_remove=p_enter + 0.05
                    )

                st.markdown("#### 📋 Quá trình chọn biến")
                if len(hist_df) > 0:
                    st.dataframe(hist_df, hide_index=True, use_container_width=True)

                    fig_r2 = px.line(
                        hist_df, x='Step', y='R²',
                        markers=True, text='Action',
                        title='R² qua từng bước'
                    )
                    fig_r2.update_traces(textposition='top center', marker_size=8)
                    fig_r2.update_layout(height=350)
                    st.plotly_chart(fig_r2, use_container_width=True)
                else:
                    st.warning("Không có bước nào được ghi lại.")

                if model is not None:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("R²", f"{model.rsquared:.4f}")
                    c2.metric("Adj R²", f"{model.rsquared_adj:.4f}")
                    c3.metric("Biến được chọn", len(selected))

                    sel_num = [v for v in selected if v in NUMERIC_COLS]
                    sel_bin = [v for v in selected if v in BINARY_COLS]
                    sel_genre = [v for v in selected if v in genre_dummy_cols]

                    st.markdown("#### ✅ Biến được chọn")
                    cs1, cs2, cs3 = st.columns(3)

                    def _show_vars(col, title, var_list, model):
                        with col:
                            st.markdown(title)
                            for v in var_list:
                                st.code(
                                    f"{v}\n"
                                    f"coef={model.params.get(v, 0):.4f}\n"
                                    f"p={model.pvalues.get(v, 1):.4f}"
                                )
                            if not var_list:
                                st.caption("Không có")

                    _show_vars(cs1, "**📐 Numeric**", sel_num, model)
                    _show_vars(cs2, "**🔘 Binary**", sel_bin, model)
                    _show_vars(cs3, "**🎵 Genre**", sel_genre, model)

                    # Standardized Coefficient
                    st.markdown("---")
                    st.markdown("#### 📊 Tầm quan trọng (Standardized Coefficient)")

                    try:
                        selected_unique = list(dict.fromkeys(selected))

                        X_s = df_model[selected_unique].copy()
                        X_s = X_s.loc[:, ~X_s.columns.duplicated()]
                        selected_unique = X_s.columns.tolist()

                        for c in selected_unique:
                            col_data = X_s[c]
                            if isinstance(col_data, pd.DataFrame):
                                col_data = col_data.iloc[:, 0]
                            X_s[c] = pd.to_numeric(col_data, errors='coerce')

                        X_s = X_s.replace([np.inf, -np.inf], np.nan)

                        y_s = df_model[target_var].copy()
                        if isinstance(y_s, pd.DataFrame):
                            y_s = y_s.iloc[:, 0]
                        y_s = pd.to_numeric(y_s, errors='coerce')

                        valid_mask = X_s.notna().all(axis=1) & y_s.notna()
                        X_s = X_s[valid_mask]
                        y_s = y_s[valid_mask]

                        if len(X_s) < len(selected_unique) + 2:
                            st.warning("Không đủ dữ liệu để vẽ standardized coefficient.")
                        else:
                            scaler = StandardScaler()
                            X_scaled = pd.DataFrame(
                                scaler.fit_transform(X_s),
                                columns=selected_unique,
                                index=X_s.index
                            )

                            Xs_c = sm.add_constant(X_scaled, has_constant='add')
                            m_std = sm.OLS(y_s, Xs_c).fit()

                            params = m_std.params.iloc[1:]
                            pvals = m_std.pvalues.iloc[1:]

                            if len(params) > 0:
                                coef_df = pd.DataFrame({
                                    'Biến': params.index.tolist(),
                                    'Std Coef': params.values,
                                    'p-value': pvals.values
                                }).reset_index(drop=True)

                                coef_df['|Coef|'] = coef_df['Std Coef'].abs()
                                coef_df['Nhóm'] = coef_df['Biến'].apply(
                                    lambda x:
                                    'Numeric' if x in NUMERIC_COLS
                                    else 'Genre' if x in genre_dummy_cols
                                    else 'Binary'
                                )
                                coef_df = coef_df.sort_values('|Coef|', ascending=True).reset_index(drop=True)

                                fig_coef = px.bar(
                                    coef_df,
                                    x='Std Coef', y='Biến',
                                    orientation='h',
                                    color='Nhóm',
                                    color_discrete_map={
                                        'Numeric': '#3498db',
                                        'Binary': '#2ecc71',
                                        'Genre': '#e74c3c'
                                    },
                                    title='Tầm quan trọng của từng biến (Standardized Coefficient)',
                                    text=coef_df['Std Coef'].round(3),
                                    height=max(400, len(coef_df) * 32)
                                )
                                fig_coef.add_vline(x=0, line_color='black', line_width=1)
                                fig_coef.update_traces(textposition='outside')
                                st.plotly_chart(fig_coef, use_container_width=True)

                                st.dataframe(
                                    coef_df[['Biến', 'Nhóm', 'Std Coef', 'p-value', '|Coef|']]
                                    .sort_values('|Coef|', ascending=False)
                                    .reset_index(drop=True),
                                    column_config={
                                        'Std Coef': st.column_config.NumberColumn(format='%.4f'),
                                        'p-value': st.column_config.NumberColumn(format='%.4f'),
                                        '|Coef|': st.column_config.NumberColumn(format='%.4f'),
                                    },
                                    hide_index=True
                                )

                                st.caption(
                                    f"R² = {m_std.rsquared:.4f} | "
                                    f"Adj R² = {m_std.rsquared_adj:.4f} | "
                                    f"Số biến: {len(selected_unique)}"
                                )

                    except Exception as e:
                        st.warning(f"Không vẽ được standardized coef: {e}")

                else:
                    st.error(
                        "Không có biến nào được chọn. "
                        "Thử giảm ngưỡng p-value hoặc thêm biến."
                    )