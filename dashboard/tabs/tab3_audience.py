import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

MUSIC_GENRES = {
    'Rap/HipHop': ['rap', 'hip hop', 'hiphop', 'hip-hop', 'v-rap', 'vrap', 'rap việt', 'rap viet'],
    'Cover': ['cover', 'cover lại', 'phiên bản cover', 'cover version', 'acoustic cover'],
    'Nhạc Trẻ': ['nhạc trẻ', 'v-pop', 'vpop', 'pop việt', 'nhạc pop', 'nhạc trẻ hay'],
    'Nhạc Tết': ['nhạc tết', 'nhạc xuân', 'tết', 'lì xì', 'mừng xuân', 'new year'],
    'Nhạc Ballad': ['ballad', 'balad', 'tình ca', 'slow', 'slow rock'],
    'K-Pop': ['kpop', 'k-pop', 'blackpink', 'bts', 'twice', 'stray kids', 'newjeans', 'exo', 'aespa', 'itzy'],
    'Nhạc Dance/EDM': ['dance', 'edm', 'electro', 'remix', 'dj', 'house music', 'house', 'club', 'dubstep', 'trap'],
    'Rock': ['rock', 'hard rock', 'metal', 'punk', 'rock ballad'],
    'Nhạc Indie': ['indie', 'folk', 'acoustic', 'lofi', 'lo-fi'],
    'R&B/Soul': ['r&b', 'rnb', 'soul', 'neo soul'],
    'Nhạc Việt Cổ': ['nhạc cổ', 'xẩm', 'ca trù', 'quan họ', 'chèo', 'tuồng'],
    'Nhạc Hòa Tấu': ['nhạc hòa tấu', 'instrumental', 'jazz', 'nhạc cụ', 'piano', 'guitar'],
    'Nhạc Truyền Thống': ['nhạc truyền thống', 'dân ca', 'cải lương', 'lý', 'guitar classical'],
    'Pop': ['pop', 'upbeat', 'pop dance'],
    'Giai điệu Khác': []
}


def _pack_bubbles(values, padding=0.08):
    values = np.array(values, dtype=float)
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
    values = df[value_col].astype(float).fillna(0).tolist()
    labels = df[label_col].astype(str).tolist()
    positions, radii = _pack_bubbles(values)

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


def _collapse_top(series, top_n=8, other_label='Khác'):
    top = series.nlargest(top_n)
    other = series.sum() - top.sum()
    if other > 0:
        if other_label in top.index:
            top.loc[other_label] += other
        else:
            top = pd.concat([top, pd.Series({other_label: other})])
    return top


def _ensure_genre_label(frame: pd.DataFrame) -> pd.DataFrame:
    if "genre_label" in frame.columns:
        return frame

    frame = frame.copy()
    text_candidates = ['video_title', 'video_description', 'video_tags', 'video_tag', 'video_tags_list']
    text_cols = [col for col in text_candidates if col in frame.columns]
    if text_cols:
        combined_text = frame[text_cols].fillna('').astype(str).agg(' '.join, axis=1).str.lower()
    else:
        combined_text = pd.Series('', index=frame.index)

    genre_label = pd.Series('Giai điệu Khác', index=frame.index)
    for genre, keywords in MUSIC_GENRES.items():
        if not keywords:
            continue
        pattern = r'(?:' + '|'.join(re.escape(k) for k in keywords) + r')'
        matched = combined_text.str.contains(pattern, regex=True)
        genre_label = genre_label.mask(matched & (genre_label == 'Giai điệu Khác'), genre)

    frame['genre_label'] = genre_label
    return frame


def render_tab(filtered_df, base_filtered_df, active_cross_filters, apply_cross_filters, sync_chart_selection):
    st.info("Người xem trên nền tảng đang thực sự quan tâm đến những loại nội dung âm nhạc nào, và xu hướng phát hành của các kênh có đang phản ánh đúng nhu cầu đó hay không?")

    # Tiền xử lý cho Q2
    df_q2 = filtered_df.dropna(subset=['video_view_count']).copy()
    df_q2['video_made_for_kids'] = df_q2['video_made_for_kids'].fillna(False).astype(bool)
    df_q2['video_licensed_content'] = df_q2['video_licensed_content'].fillna(False).astype(bool)

    df_q2 = _ensure_genre_label(df_q2)

    # Treemap section removed as requested.

    # --- BIỂU ĐỒ: BOXPLOT CHO PHỤ ĐỀ ---
    st.subheader("1. Vai trò của Phụ đề (Lyrics/CC) đối với lượt xem")
    df_q2['has_caption'] = df_q2['video_caption_status'].astype(str).str.lower() == 'true'

    fig5 = px.box(
        df_q2,
        x='has_caption',
        y='video_view_count',
        color='has_caption',
        log_y=True,
        color_discrete_sequence=['#1f77b4', '#AED6F1'],
        labels={'has_caption': 'Có phụ đề (CC)', 'video_view_count': 'Lượt xem (Log)'},
        title="Box Plot: Phân phối lượt xem của video Có vs Không có phụ đề"
    )
    fig5.update_traces(
        hovertemplate=(
            "Có phụ đề: %{x}<br>"
            "Lượt xem gốc: %{y:,}<extra></extra>"
        )
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.caption(
        "Boxplot mô tả phân phối lượt xem giữa hai nhóm có và không có phụ đề. "
        "Quan sát vị trí hộp và median giúp biết nhóm nào có mức view điển hình cao hơn, "
        "đồng thời nhìn được độ phân tán của từng nhóm." 
    )

    # --- BIỂU ĐỒ 2.3: TOP VIDEO THEO LƯỢT XEM / LƯỢT THÍCH ---
    st.subheader("2. Top video theo lượt xem và lượt thích")
    col_view, col_like = st.columns(2)

    with col_view:
        if {'video_title', 'video_view_count', 'video_like_count'}.issubset(df_q2.columns):
            top_view = (
                df_q2[['video_title', 'video_view_count', 'video_like_count']]
                .dropna(subset=['video_view_count'])
                .copy()
                .sort_values('video_view_count')
                .tail(10)
            )
            top_view['video_like_count'] = top_view['video_like_count'].fillna(0)
            fig_top_view = _packed_bubble_chart(
                top_view,
                value_col='video_view_count',
                label_col='video_title',
                title="Top 10 video theo lượt xem (Packed Bubble)",
                hover_cols=['video_view_count', 'video_like_count'],
                hover_labels=['Lượt xem', 'Lượt thích']
            )
            st.plotly_chart(fig_top_view, use_container_width=True)
        else:
            st.info("Thiếu dữ liệu video title, lượt xem hoặc lượt thích để hiển thị top view.")

    with col_like:
        if {'video_title', 'video_like_count', 'video_view_count'}.issubset(df_q2.columns):
            top_like = (
                df_q2[['video_title', 'video_like_count', 'video_view_count']]
                .dropna(subset=['video_like_count'])
                .copy()
                .sort_values('video_like_count')
                .tail(10)
            )
            top_like['video_view_count'] = top_like['video_view_count'].fillna(0)
            fig_top_like = _packed_bubble_chart(
                top_like,
                value_col='video_like_count',
                label_col='video_title',
                title="Top 10 video theo lượt thích (Packed Bubble)",
                hover_cols=['video_like_count', 'video_view_count'],
                hover_labels=['Lượt thích', 'Lượt xem']
            )
            st.plotly_chart(fig_top_like, use_container_width=True)
        else:
            st.info("Thiếu dữ liệu video title, lượt thích hoặc lượt xem để hiển thị top like.")

    st.caption(
        "Hai biểu đồ xếp hạng giúp nhận diện các video nổi bật theo mức độ quan tâm (view) "
        "và mức độ yêu thích (like). Chênh lệch giữa hai danh sách cho thấy video nào "
        "được xem nhiều nhưng chưa chắc được thích tương ứng." 
    )

    # --- BIỂU ĐỒ 2.4: PHÂN PHỐI THỂ LOẠI / FORMAT UPLOAD ---
    st.subheader("3. Phân phối thể loại và chất lượng video được đăng tải")
    format_candidates = ['video_definition', 'video_dimension', 'format']
    genre_col = 'genre_label'
    format_col = next((col for col in format_candidates if col in df_q2.columns), None)

    col_genre, col_format = st.columns(2)

    with col_genre:
        genre_series = df_q2[genre_col].astype(str).replace({'Khác': 'Giai điệu Khác', 'khác': 'Giai điệu Khác'})
        genre_counts = genre_series.value_counts()
        genre_counts = _collapse_top(genre_counts, top_n=8, other_label='Giai điệu Khác')
        genre_df = genre_counts.reset_index()
        genre_df.columns = ['TheLoai', 'SoVideo']
        fig_genre = px.pie(
            genre_df,
            names='TheLoai',
            values='SoVideo',
            title="Tỷ trọng thể loại được đăng tải (lọc từ title/description/tags)"
        )
        fig_genre.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_genre, use_container_width=True)

    with col_format:
        if format_col:
            format_counts = df_q2[format_col].astype(str).value_counts()
            format_counts = _collapse_top(format_counts, top_n=6, other_label='Khác')
            format_df = format_counts.reset_index()
            format_df.columns = ['Format', 'SoVideo']
            fig_format = px.pie(
                format_df,
                names='Format',
                values='SoVideo',
                title="Tỷ trọng chất lượng video đăng tải"
            )
            fig_format.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_format, use_container_width=True)
        else:
            st.info("Không tìm thấy cột format trong dữ liệu.")

    st.caption(
        "Phân phối này phản ánh hành vi của người đăng: Họ ưu tiên thể loại hoặc chất lượng video nào khi đăng tải. "
        "Nếu một nhóm chiếm tỷ trọng quá lớn, đây là dấu hiệu thị trường đang bị lệch về một hướng nội dung." 
    )

    # --- BIỂU ĐỒ 2.5: SO SÁNH UPLOAD VS VIEW THEO NHÓM ---
    st.subheader("4. So sánh phân phối tỷ lệ thể loại và phân phối lượt xem")
    compare_col = genre_col
    if compare_col:
        compare_source = apply_cross_filters(
            base_filtered_df,
            active_cross_filters,
            exclude_key="cross_compare_genre",
        )
        compare_source = _ensure_genre_label(compare_source)
        compare_df = compare_source[[compare_col, 'video_view_count']].dropna().copy()
        compare_df[compare_col] = compare_df[compare_col].astype(str)

        upload_counts = compare_df[compare_col].value_counts().nlargest(8)
        view_sums = compare_df.groupby(compare_col)['video_view_count'].sum().reindex(upload_counts.index)

        upload_share = (upload_counts / upload_counts.sum() * 100).round(2)
        view_share = (view_sums / view_sums.sum() * 100).round(2)

        fig_compare = make_subplots(specs=[[{"secondary_y": True}]])
        fig_compare.add_trace(
            go.Bar(
                x=upload_share.index.tolist(),
                y=upload_share.values,
                name='Tỷ lệ upload (%)',
                marker_color='rgba(54, 162, 235, 0.75)',
                customdata=upload_share.index.tolist(),
                hovertemplate="Thể loại: %{customdata}<br>Tỷ lệ upload: %{y:.2f}%<extra></extra>",
            ),
            secondary_y=False
        )
        fig_compare.add_trace(
            go.Scatter(
                x=view_share.index.tolist(),
                y=view_share.values,
                name='Tỷ lệ lượt xem (%)',
                marker_color='rgba(255, 99, 132, 0.9)',
                mode='lines+markers',
                line=dict(width=3),
                customdata=view_share.index.tolist(),
                hovertemplate="Thể loại: %{customdata}<br>Tỷ lệ lượt xem: %{y:.2f}%<extra></extra>",
            ),
            secondary_y=True
        )
        fig_compare.update_layout(
            title="So sánh tỷ lệ lượt xem theo thể loại",
            xaxis_title="Thể loại",
            legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5),
            clickmode='event+select',
        )
        fig_compare.update_yaxes(title_text='Tỷ lệ thể loại được đăng tải (%)', secondary_y=False)
        fig_compare.update_yaxes(title_text='Tỷ lệ lượt xem (%)', secondary_y=True)
        compare_event = st.plotly_chart(
            fig_compare,
            use_container_width=True,
            key="cross_compare_genre",
            on_select="rerun",
        )
        sync_chart_selection("cross_compare_genre", compare_event)
        st.caption(
            "Cột thể hiện tỷ lệ đăng tải, đường thể hiện tỷ lệ lượt xem theo cùng một nhóm. "
            "Nhìn vào khoảng cách giữa cột và đường để nhận ra nhóm nào vượt trội về mức độ quan tâm." 
        )
    else:
        st.info("Không có cột thể loại hoặc format để so sánh phân phối.")

    st.subheader("5. Xu hướng thể loại theo thời gian")
    if 'video_publish_date' in df_q2.columns:
        df_time = df_q2.dropna(subset=['video_publish_date']).copy()
        df_time['month'] = df_time['video_publish_date'].dt.to_period('M').dt.to_timestamp()
        trend = df_time.groupby(['month', 'genre_label']).size().reset_index(name='SoVideo')
        fig_area = px.area(
            trend,
            x='month',
            y='SoVideo',
            color='genre_label',
            title="Xu hướng thể loại theo thời gian (theo số video đăng)",
            labels={'month': 'Tháng', 'SoVideo': 'Số video', 'genre_label' : "Thể loại"}
        )
        st.plotly_chart(fig_area, use_container_width=True)
        st.caption(
            "Stacked area chart cho thấy mức độ đăng tải của từng thể loại theo thời gian. "
            "Phần diện tích tăng/giảm phản ánh xu hướng nội dung và mức độ thay đổi thể loại âm nhạc của người đăng." 
        )
    else:
        st.info("Cần có cột ngày đăng (video_publish_date) để hiển thị xu hướng thể loại theo thời gian.")
