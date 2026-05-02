import streamlit as st


def render_tab():
    st.subheader("Phân tích yếu tố gây nhiễu")

    st.info("Mục này được giữ ở dạng mô tả vì phần cross-filter của dashboard đang tập trung vào các plot có ý nghĩa rõ ràng hơn để tránh gây nhiễu cho luồng lọc chung.")

    st.markdown("""
    ### Nhận định:
    - Quy mô kênh có thể làm lệch kết luận về khung giờ đăng  
    - Thời điểm đăng chưa chắc là nguyên nhân thực sự  
    """)
