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
# THỨ TỰ GHÉP 1) set_page_config (PHẢI LÀ LỆNH STREAMLIT ĐẦU TIÊN)
# =============================================================================
st.set_page_config(
    layout="wide",
    page_title="Hệ thống Phát hiện Giao dịch Gian lận",
    page_icon="🛡️"
)

# =============================================================================
# THỨ TỰ GHÉP 2) IMPORT & CÁC HÀM CACHE DÙNG CHUNG
# =============================================================================
@st.cache_data
def load_data(file_bytes, file_name):
    """
    HÀM NẠP DỮ LIỆU DÙNG CHUNG: Nhận bytes của file (để hashable cho st.cache_data),
    sử dụng io.BytesIO để tránh hoàn toàn lỗi đọc bytes của Pandas.
    """
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

# Khai báo tập biến đặc trưng X và mục tiêu y trích xuất từ dữ liệu thực tế
FEATURE_COLUMNS = [f"X_{i}" for i in range(1, 15)]
TARGET_COLUMN = "default"

# =============================================================================
# THỨ TỰ GHÉP 3) SIDEBAR (THÀNH PHẦN 1 - VÙNG CẤU HÌNH BỀN VỮNG)
# =============================================================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    st.divider()
    
    # TẢI DỮ LIỆU HUẤN LUYỆN MẪU
    uploaded_file = st.file_uploader(
        "Tải lên dữ liệu huấn luyện mẫu", 
        type=["csv", "xlsx"],
        help="Chọn tệp dữ liệu huấn luyện mẫu (ví dụ: dataset1.csv) chứa các cột X_1 đến X_14 và biến mục tiêu default."
    )
    
    st.divider()
    st.subheader("Tham số mô hình AI")
    st.caption("**Mô hình: Random Forest Classifier**")
    
    # THAM SỐ MÔ HÌNH (Lấy giá trị mặc định tối ưu hợp lý từ quy trình)
    n_estimators = st.slider(
        "Số lượng cây (n_estimators)",
        min_value=10, max_value=300, value=100, step=10,
        help="Số lượng cây quyết định được xây dựng trong rừng ngẫu nhiên."
    )
    
    criterion = st.selectbox(
        "Tiêu chí phân hoạch (criterion)",
        options=["gini", "entropy", "log_loss"], index=0,
        help="Hàm đo lường chất lượng phân tách tại các nút."
    )
    
    max_depth = st.slider(
        "Độ sâu tối đa (max_depth)",
        min_value=1, max_value=50, value=15, step=1,
        help="Độ sâu tối đa của mỗi cây quyết định."
    )
    
    min_samples_split = st.slider(
        "Mẫu tối thiểu để tách nút (min_samples_split)",
        min_value=2, max_value=20, value=2, step=1,
        help="Số lượng mẫu tối thiểu cần thiết để phân tách một nút nội bộ."
    )
    
    random_state = st.number_input(
        "Trạng thái ngẫu nhiên (random_state)",
        min_value=0, max_value=9999, value=42, step=1,
        help="Đảm bảo tính tái lập kết quả huấn luyện giữa các lần chạy."
    )
    
    # Gom tham số nâng cao vào expander gọn gàng
    with st.expander("🛠️ Cấu hình nâng cao"):
        min_samples_leaf = st.slider(
            "Mẫu tối thiểu tại nút lá",
            min_value=1, max_value=20, value=1, step=1,
            help="Số lượng mẫu tối thiểu bắt buộc phải có tại một nút lá."
        )
        bootstrap = st.checkbox(
            "Sử dụng Bootstrap samples", value=True,
            help="Có áp dụng kỹ thuật lấy mẫu bootstrap khi xây dựng các cây hay không."
        )

    st.divider()
    
    # NÚT HÀNH ĐỘNG DUY NHẤT ĐỂ KÍCH HOẠT HUẤN LUYỆN
    train_clicked = st.button(
        "🚀 Huấn luyện Mô hình", 
        type="primary", 
        use_container_width=True,
        help="Điểm duy nhất kích hoạt quá trình khớp mô hình (fit) và đánh giá."
    )

# =============================================================================
# THỨ TỰ GHÉP 4) HEADER + KIỂM TRA TRẠNG THÁI DỮ LIỆU (THÀNH PHẦN 2)
# =============================================================================
st.title("🛡️ Hệ thống Phát hiện Giao dịch Gian lận")
st.caption("Ứng dụng hỗ trợ phân tích rủi ro tín dụng và tự động chẩn đoán hành vi giao dịch gian lận (default) dựa trên nền tảng Học máy.")

# KIỂM TRA XỬ LÝ TRẠNG THÁI RỖNG
if uploaded_file is None:
    st.info("💡 Hướng dẫn: Vui lòng tải lên tệp dữ liệu mẫu (.csv hoặc .xlsx) tại Sidebar bên trái để bắt đầu khám phá ứng dụng.")
    st.stop()

# ĐÃ CÓ FILE -> Nạp dữ liệu qua hàm cache dùng chung với đối tượng bytes
file_bytes = uploaded_file.getvalue()
df_raw = load_data(file_bytes, uploaded_file.name)

if df_raw is None:
    st.error("❌ Định dạng file không hợp lệ hoặc không thể đọc được dữ liệu. Vui lòng kiểm tra lại.")
    st.stop()

st.caption(f"📌 Đang dùng tệp dữ liệu: `{uploaded_file.name}` | Quy mô: **{df_raw.shape[0]:,}** dòng, **{df_raw.shape[1]}** cột.")
st.divider()

# =============================================================================
# THỨ TỰ GHÉP 5) KHỐI TRAIN (Chạy khi bấm nút, lưu kết quả vào session_state)
# =============================================================================
if train_clicked:
    with st.spinner("🔄 Hệ thống đang tiến hành huấn luyện mô hình... Vui lòng đợi..."):
        # Kiểm tra tính toàn vẹn của dữ liệu đầu vào
        missing_cols = [c for c in FEATURE_COLUMNS if c not in df_raw.columns]
        if missing_cols or (TARGET_COLUMN not in df_raw.columns):
            st.error(f"❌ Dữ liệu thiếu cột bắt buộc của pipeline: {missing_cols} {TARGET_COLUMN if TARGET_COLUMN not in df_raw.columns else ''}")
        else:
            X = df_raw[FEATURE_COLUMNS]
            y = df_raw[TARGET_COLUMN]
            
            # Phân tách tập train/test theo tỷ lệ 80:20 đồng bộ chuẩn ML
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
            
            # Cấu hình và khớp mô hình (fit)
            model = RandomForestClassifier(
                n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                bootstrap=bootstrap, random_state=random_state, n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Dự đoán để lấy chỉ tiêu kiểm định
            y_pred = model.predict(X_test)
            
            # Lưu trữ 3 thành phần vào session_state để mọi tab dùng chung mà KHÔNG train lại
            st.session_state['trained_model'] = model
            st.session_state['evaluation_metrics'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            st.session_state['feature_importances'] = pd.DataFrame({
                'Feature': FEATURE_COLUMNS,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            
            st.success("🎉 Huấn luyện hoàn tất! Các Tab thông tin đã được kích hoạt và cập nhật kết quả mới nhất.")

# =============================================================================
# THỨ TỰ GHÉP 6) st.tabs CHỨA TOÀN BỘ CÁC THÀNH PHẦN CÒN LẠI
# =============================================================================
tab_overview, tab_viz, tab_metrics, tab_inference = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa dữ liệu", 
    "🎯 Kết quả & Kiểm định", 
    "🔮 Sử dụng mô hình dự báo"
])

# -----------------------------------------------------------------------------
# THÀNH PHẦN 3: TAB "TỔNG QUAN DỮ LIỆU"
# -----------------------------------------------------------------------------
with tab_overview:
    st.subheader("📋 Phân tích Thống kê Đặc điểm Dữ liệu")
    
    # KÍCH THƯỚC DỮ LIỆU
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Tổng số dòng giao dịch", f"{df_raw.shape[0]:,}")
    col_m2.metric("Số lượng cột đặc trưng", f"{df_raw.shape[1]}")
    col_m3.metric("Dung lượng file tải lên", f"{(len(file_bytes)/(1024*1024)):.2f} MB")
    
    # XEM DỮ LIỆU THÔ (Gói trong container cuộn gọn gàng)
    st.markdown("#### 🔍 Xem trước 5 hàng đầu tiên")
    with st.container(height=250):
        st.dataframe(df_raw.head(), use_container_width=True)
        
    # THỐNG KÊ MÔ TẢ (Chỉ mô tả các biến đưa vào mô hình X và y)
    st.markdown("#### 📉 Thống kê mô tả các biến cấu thành mô hình")
    model_vars = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c in df_raw.columns]
    if model_vars:
        st.dataframe(df_raw[model_vars].describe(), use_container_width=True)

# -----------------------------------------------------------------------------
# THÀNH PHẦN 4: TAB "TRỰC QUAN HÓA DỮ LIỆU"
# -----------------------------------------------------------------------------
with tab_viz:
    st.subheader("📊 Khám phá phân phối phân bổ các thuộc tính")
    
    # ƯU TIÊN BIẾN MỤC TIÊU (y) TRƯỚC TIÊN
    if TARGET_COLUMN in df_raw.columns:
        target_counts = df_raw[TARGET_COLUMN].value_counts().reset_index()
        target_counts.columns = [TARGET_COLUMN, 'Số lượng']
        target_counts[TARGET_COLUMN] = target_counts[TARGET_COLUMN].map({0: 'Hợp lệ (0)', 1: 'Gian lận (1)'})
        
        fig_y = px.bar(
            target_counts, x=TARGET_COLUMN, y='Số lượng', color=TARGET_COLUMN,
            title="Tần suất phân phối của Biến mục tiêu (default)",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_y.update_layout(height=300)
        st.plotly_chart(fig_y, use_container_width=True)
        
    st.markdown("#### 🧩 Đồ thị phân phối các biến đầu vào độc lập (X)")
    
    # Dùng multiselect để xử lý khi có nhiều biến (>4)
    default_viz = ["X_1", "X_2", "X_5", "X_13"]
    selected_viz = st.multiselect(
        "Chọn các biến X muốn hiển thị đồ thị phân phối:",
        options=FEATURE_COLUMNS,
        default=[f for f in default_viz if f in df_raw.columns]
    )
    
    if selected_viz:
        # Lưới 2x2 hoặc tự động chia cột cân đối bằng plotly chiều cao cố định
        grid_cols = st.columns(2)
        for idx, feat in enumerate(selected_viz):
            with grid_cols[idx % 2]:
                if df_raw[feat].dtype in [np.float64, np.int64]:
                    fig = px.histogram(
                        df_raw, x=feat, color=TARGET_COLUMN if TARGET_COLUMN in df_raw.columns else None,
                        marginal="box", title=f"Phân phối tần suất & Ngoại lai của {feat}",
                        barmode="overlay", height=280
                    )
                else:
                    v_c = df_raw[feat].value_counts().reset_index()
                    v_c.columns = [feat, 'Count']
                    fig = px.bar(v_c, x=feat, y='Count', title=f"Phân phối phân loại {feat}", height=280)
                st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# THÀNH PHẦN 5: TAB "KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH"
# -----------------------------------------------------------------------------
with tab_metrics:
    st.subheader("🎯 Hiệu năng mô hình phân loại Rừng ngẫu nhiên")
    
    # ĐIỀU PHỐI TRẠNG THÁI: Kiểm tra session_state, hướng dẫn nếu chưa train
    if 'trained_model' not in st.session_state:
        st.info("💡 Hướng dẫn: Mô hình chưa được khởi tạo. Vui lòng cấu hình tham số và bấm nút '🚀 Huấn luyện Mô hình' ở thanh Sidebar bên trái để hiển thị báo cáo kiểm định.")
        st.stop()
        
    metrics = st.session_state['evaluation_metrics']
    feat_imp = st.session_state['feature_importances']
    
    # Chỉ tiêu vô hướng hiển thị dạng cột metric cân đối
    c_1, c_2, c_3, c_4 = st.columns(4)
    c_1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    c_2.metric("Precision", f"{metrics['precision']:.4f}")
    c_3.metric("Recall (Độ nhạy)", f"{metrics['recall']:.4f}")
    c_4.metric("F1-Score", f"{metrics['f1']:.4f}")
    
    st.divider()
