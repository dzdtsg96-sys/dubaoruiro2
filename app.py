import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# =============================================================================
# 1) CONFIGURATION (Bắt buộc đầu tiên)
# =============================================================================
st.set_page_config(
    layout="wide",
    page_title="Hệ thống Phát hiện Giao dịch Gian lận",
    page_icon="🛡️"
)

# =============================================================================
# 2) CUSTOM CSS - GIAO DIỆN HIỆN ĐẠI, DỄ NHÌN, SẮC NÉT
# =============================================================================
st.markdown("""
    <style>
        /* Tùy chỉnh màu nền và font nền tảng */
        .main {
            background-color: #f8fafc;
        }
        /* Làm đẹp thanh Sidebar */
        [data-testid="stSidebar"] {
            background-color: #0f172a;
            color: #ffffff;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p {
            color: #f8fafc !important;
        }
        /* Làm đẹp các Tab thành thanh điều hướng cao cấp */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #f1f5f9;
            padding: 8px 12px;
            border-radius: 12px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
            border-radius: 8px;
            color: #475569;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #1e3a8a;
            background-color: #e2e8f0;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1e40af !important;
            color: white !important;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        }
        /* Làm nổi bật các khối Containers / Hộp thông báo */
        div.stMetric {
            background-color: white;
            padding: 16px;
            border-radius: 12px;
            box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1);
            border-left: 5px solid #3b82f6;
        }
        .stAlert {
            border-radius: 12px !important;
        }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# 3) HÀM CACHE NẠP FILE
# =============================================================================
@st.cache_data
def load_data(file_bytes, file_name):
    try:
        data_stream = io.BytesIO(file_bytes)
        if file_name.endswith('.csv'):
            df = pd.read_csv(data_stream)
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_stream)
        else:
            return None
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file dữ liệu: {e}")
        return None

FEATURE_COLUMNS = [f"X_{i}" for i in range(1, 15)]
TARGET_COLUMN = "default"

# =============================================================================
# 4) SIDEBAR - VÙNG ĐIỀU KHIỂN ĐẬM CHẤT CÔNG NGHỆ (DARK NAVY)
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Trung tâm Điều khiển")
    st.caption("Tải tệp dữ liệu huấn luyện và điều chỉnh siêu tham số mô hình AI.")
    st.divider()
    
    uploaded_file = st.file_uploader(
        "Tải lên dữ liệu huấn luyện mẫu", 
        type=["csv", "xlsx"],
        help="Chọn tệp dataset1.csv hoặc định dạng Excel tương tự chứa các cột X_1 đến X_14 và cột default."
    )
    
    st.divider()
    st.markdown("#### 🤖 Tham số thuật toán")
    st.caption("Mô hình: **Random Forest Classifier**")
    
    n_estimators = st.slider("Số lượng cây (n_estimators)", min_value=10, max_value=300, value=100, step=10)
    criterion = st.selectbox("Tiêu chí phân hoạch", options=["gini", "entropy", "log_loss"], index=0)
    max_depth = st.slider("Độ sâu tối đa (max_depth)", min_value=1, max_value=50, value=15, step=1)
    min_samples_split = st.slider("Mẫu tối thiểu tách nút", min_value=2, max_value=20, value=2, step=1)
    random_state = st.number_input("Trạng thái ngẫu nhiên (random_state)", min_value=0, max_value=9999, value=42, step=1)
    
    with st.expander("🛠️ Tham số tinh chỉnh sâu"):
        min_samples_leaf = st.slider("Mẫu tối thiểu tại nút lá", min_value=1, max_value=20, value=1, step=1)
        bootstrap = st.checkbox("Sử dụng Bootstrap samples", value=True)

    st.divider()
    train_clicked = st.button("🚀 Khởi chạy Huấn luyện", type="primary", use_container_width=True)

# =============================================================================
# 5) TIÊU ĐỀ ỨNG DỤNG
# =============================================================================
st.title("🛡️ Hệ thống Phát hiện Giao dịch Gian lận")
st.caption("Giải pháp quản trị rủi ro thông minh dựa trên mô hình học máy phân loại nhị phân.")

if uploaded_file is None:
    st.info("💡 **Hệ thống sẵn sàng:** Vui lòng kéo sang bảng điều khiển bên trái (**Sidebar**), tiến hành tải lên tệp dữ liệu mẫu để kích hoạt toàn bộ tính năng phân tích và chẩn đoán.")
    st.stop()

file_bytes = uploaded_file.getvalue()
df_raw = load_data(file_bytes, uploaded_file.name)

if df_raw is None:
    st.error("❌ Không thể đọc
