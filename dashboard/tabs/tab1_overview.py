import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _fmt_k(value, decimals=0):
    try:
        v = float(value)
    except Exception:
        return value
    sign = '-' if v < 0 else ''
    v_abs = abs(v)
    return f"{sign}{v_abs/1000:,.{decimals}f}K"


def _pack_bubbles(values, padding=0.08):
    values = np.array(values, dtype=float)
    if values.size == 0:
        return [], np.array([])
    if values.max() <= 0:
        values = np.ones_like(values)

    norm = values / values.max()
    radii = np.sqrt(norm) * 0.9 + 0.1
    positions = []
    placed_radii = []

    for idx, radius in enumerate(radii):
        if idx == 0:
            positions.append((0.0, 0.0))
            placed_radii.append(radius)
            continue

        placed = False
        for step in range(2500):
            angle = step * 0.55
            spiral = 0.08 * step
            x = spiral * np.cos(angle)
            y = spiral * np.sin(angle)

            ok = True
            for (px, py), pr in zip(positions, placed_radii):
                if (x - px) ** 2 + (y - py) ** 2 < (radius + pr + padding) ** 2:
                    ok = False
                    break

            if ok:
                positions.append((x, y))
                placed_radii.append(radius)
                placed = True
                break

        if not placed:
            positions.append((spiral * np.cos(angle), spiral * np.sin(angle)))
            placed_radii.append(radius)

    return positions, np.array(placed_radii)


def _short_label(text, max_len=18):
    text = str(text)
    return text if len(text) <= max_len else text[:max_len].rstrip() + "..."


def _packed_bubble_chart(df, value_col, label_col, title, hover_cols=None, hover_labels=None):
    values = pd.to_numeric(df[value_col], errors="coerce").fillna(0).tolist()
    labels = df[label_col].astype(str).tolist()
    positions, radii = _pack_bubbles(values)

    if len(radii) == 0:
        return go.Figure()

    sizes = (radii / max(radii)) * 70 + 20
    x_vals = [pos[0] for pos in positions]
    y_vals = [pos[1] for pos in positions]

    hovertext = labels
    customdata = None
    hovertemplate = "<b>%{hovertext}</b><br>Giá trị: %{marker.color:,.0f}<extra></extra>"

    if hover_cols:
        hover_labels = hover_labels or hover_cols
        hover_values = []
        for col in hover_cols:
            if col in df.columns:
                hover_values.append(pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy())
            else:
                hover_values.append(np.zeros(len(df)))

        if hover_values:
            customdata = np.column_stack(hover_values)
            hover_lines = "<br>".join(
                f"{label}: %{{customdata[{idx}]:,.0f}}" for idx, label in enumerate(hover_labels)
            )
            hovertemplate = f"<b>%{{hovertext}}</b><br>{hover_lines}<extra></extra>"

    scatter_kwargs = dict(
        x=x_vals,
        y=y_vals,
        mode='markers+text',
        text=[_short_label(label) for label in labels],
        textposition='middle center',
        textfont=dict(color="#FF9D3C", size=12),
        marker=dict(
            size=sizes,
            color=values,
            colorscale='Blues',
            showscale=False,
            line=dict(color='white', width=1),
            opacity=0.9
        ),
        hovertext=hovertext,
        hovertemplate=hovertemplate
    )
    if customdata is not None:
        scatter_kwargs["customdata"] = customdata

    fig = go.Figure(go.Scatter(**scatter_kwargs))
    fig.update_layout(
        title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig


def render_tab(filtered_df):
    st.subheader("Tổng quan")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric("Số video", f"{len(filtered_df):,}")
    col2.metric("Lượt xem trung bình", _fmt_k(filtered_df['video_view_count'].mean(), decimals=0))
    col3.metric("Lượt xem trung vị", _fmt_k(filtered_df['video_view_count'].median(), decimals=0))
    col4.metric("Lượt xem cao nhất", _fmt_k(filtered_df['video_view_count'].max(), decimals=0))
    col5.metric("Số tags trung bình", f"{filtered_df['video_tags_count'].mean():.1f}")
    col6.metric("Số người đăng ký trung bình", _fmt_k(filtered_df['channel_subscriber_count'].mean(), decimals=0))

    total_views = filtered_df['video_view_count'].sum()
    total_likes = filtered_df['video_like_count'].sum() if 'video_like_count' in filtered_df.columns else 0
    total_comments = filtered_df['video_comment_count'].sum() if 'video_comment_count' in filtered_df.columns else 0
    engagement_rate = (total_likes + total_comments) / total_views if total_views else 0
    licensed_rate = (
        filtered_df['video_licensed_content'].fillna(False).astype(bool).mean()
        if 'video_licensed_content' in filtered_df.columns
        else 0
    )
    caption_rate = (
        filtered_df['video_caption_status'].astype(str).str.lower().eq('true').mean()
        if 'video_caption_status' in filtered_df.columns
        else 0
    )

    col7, col8, col9, col10, col11, col12 = st.columns(6)
    col7.metric("Tổng lượt xem", _fmt_k(total_views, decimals=0))
    col8.metric("Tổng lượt thích", _fmt_k(total_likes, decimals=0))
    col9.metric("Tổng bình luận", _fmt_k(total_comments, decimals=0))
    col10.metric("Tỷ lệ tương tác", f"{engagement_rate * 100:.2f}%")
    col11.metric("Tỷ lệ bản quyền", f"{licensed_rate * 100:.1f}%")
    col12.metric("Tỷ lệ có phụ đề", f"{caption_rate * 100:.1f}%")

    st.caption("Ghi chú: Các chỉ số KPI ở trên được hiển thị theo đơn vị K (K = nghìn).")

    st.markdown("### Phân bố lượt xem")

    fig = px.histogram(
        filtered_df,
        x='video_view_count',
        nbins=60,
        labels={'video_view_count': 'Lượt xem'}
    )
    fig.update_layout(xaxis_title="Lượt xem", yaxis_title="Số video")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Biểu đồ cho thấy phân bố số video theo lượt xem. Trục X là lượt xem, trục Y là số video. "
        "Nếu phân bố lệch phải, thị trường có nhiều video lượt xem thấp và ít video bứt phá." 
    )

    st.markdown("### Tổng quan kênh và bản quyền")
    col_left, col_right = st.columns(2)

    with col_left:
        if {'channel_title', 'video_view_count'}.issubset(filtered_df.columns):
            top_channels = (
                filtered_df.groupby('channel_title')['video_view_count']
                .sum()
                .nlargest(10)
                .sort_values(ascending=True)
                .reset_index()
            )
            fig_channels = px.bar(
                top_channels,
                x='video_view_count',
                y='channel_title',
                orientation='h',
                labels={'video_view_count': 'Tổng lượt xem', 'channel_title': 'Kênh'},
                title="Top 10 kênh theo tổng lượt xem"
            )
            st.plotly_chart(fig_channels, use_container_width=True)
            st.caption(
                "Xếp hạng các kênh đóng góp nhiều lượt xem nhất trong tập dữ liệu hiện tại. "
                "Biểu đồ này giúp nhận diện những kênh dẫn dắt lưu lượng và mức độ tập trung thị trường." 
            )
        else:
            st.info("Thiếu dữ liệu kênh để hiển thị top channels.")

    with col_right:
        if 'video_licensed_content' in filtered_df.columns:
            license_counts = (
                filtered_df['video_licensed_content']
                .fillna(False)
                .astype(bool)
                .map({True: 'Chính thức (Official)', False: 'Tự do (Cover/Remix)'})
                .value_counts()
                .reset_index()
            )
            license_counts.columns = ['Loai', 'SoLuong']
            fig_license = px.pie(
                license_counts,
                names='Loai',
                values='SoLuong',
                title="Tỷ trọng bản quyền"
            )
            fig_license.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_license, use_container_width=True)
            st.caption(
                "Tỷ trọng số video theo nhóm bản quyền giúp hiểu mức độ phổ biến của nội dung chính thức "
                "so với cover/remix trong tập dữ liệu đang phân tích." 
            )
        else:
            st.info("Thiếu dữ liệu bản quyền để hiển thị tỷ trọng.")

    st.markdown("### Xu hướng lượt xem theo thời gian")
    if {'video_publish_date', 'video_view_count'}.issubset(filtered_df.columns):
        df_time = filtered_df.dropna(subset=['video_publish_date']).copy()
        df_time['month'] = df_time['video_publish_date'].dt.to_period('M').dt.to_timestamp()
        monthly_views = df_time.groupby('month')['video_view_count'].sum().reset_index()
        fig_time = px.line(
            monthly_views,
            x='month',
            y='video_view_count',
            labels={'month': 'Tháng', 'video_view_count': 'Tổng lượt xem'},
            title="Tổng lượt xem theo tháng"
        )
        st.plotly_chart(fig_time, use_container_width=True)
        st.caption(
            "Đường xu hướng thể hiện tổng lượt xem theo tháng. "
            "Nhìn vào độ dốc và các điểm tăng/giảm để xác định giai đoạn bùng nổ hoặc suy giảm." 
        )
    else:
        st.info("Thiếu dữ liệu thời gian để hiển thị xu hướng.")

    st.markdown("### Phân bố độ dài video")

    if 'video_duration' in filtered_df.columns:
        duration_minutes = filtered_df['video_duration'] / 60
        
        # Tách outliers (>100 phút)
        main_duration = duration_minutes[duration_minutes <= 100]
        outliers_df = filtered_df.loc[duration_minutes > 100].copy()
        outliers_df['duration_minutes'] = duration_minutes[duration_minutes > 100].values
        
        fig_duration = px.histogram(
            main_duration,
            nbins=50,
            labels={'value': 'Thời lượng (phút)', 'count': 'Số video','variable' : "Biến"},
            title="Phân bố thời lượng video (≤100 phút)"
        )
        fig_duration.update_layout(xaxis_title="Thời lượng (phút)", yaxis_title="Số video")
        st.plotly_chart(fig_duration, use_container_width=True)
        st.caption(
            "Phân bố thời lượng giúp nhìn mức độ phổ biến của các độ dài video. "
            "Vùng tập trung cao là khoảng thời lượng được đăng nhiều nhất." 
        )
        
        if len(outliers_df) > 0:
            st.info(f"⚠️ Phát hiện {len(outliers_df)} video vượt quá 100 phút.")

            if 'video_title' in outliers_df.columns:
                top_outliers = outliers_df.sort_values('duration_minutes', ascending=True).tail(20).copy()

                hover_cols = ['duration_minutes']
                hover_labels = ['Thời lượng (phút)']

                if 'video_view_count' in top_outliers.columns:
                    top_outliers['video_view_count'] = top_outliers['video_view_count'].fillna(0)
                    hover_cols.append('video_view_count')
                    hover_labels.append('Lượt xem')

                fig_outliers = _packed_bubble_chart(
                    top_outliers,
                    value_col='duration_minutes',
                    label_col='video_title',
                    title=f"Top {len(top_outliers)} video có thời lượng dài nhất",
                    hover_cols=hover_cols,
                    hover_labels=hover_labels
                )
                st.plotly_chart(fig_outliers, use_container_width=True)

                st.caption(
                    "Kích thước bong bóng thể hiện thời lượng của video. "
                    "Đây thường là các video nhạc không lời hoặc nhạc hòa tấu."
                )
            else:
                st.warning("Thiếu cột 'video_title' để vẽ biểu đồ bong bóng cho các video ngoại lai.")
        
    else:
        st.info("Thiếu dữ liệu thời lượng để hiển thị phân bố.")
