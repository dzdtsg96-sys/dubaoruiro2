import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# -----------------------------------------------------------------------------
# LỆNH STREAMLIT ĐẦU TIÊN: Cấu hình Trang ứng dụng
# -----------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Hệ thống Phát hiện Giao dịch Gian lận",
    page_icon="🛡️"
)

# -----------------------------------------------------------------------------
# 2. HÀM NẠP VÀ TIỀN XỬ LÝ DỮ LIỆU (CACHE DÙNG CHUNG)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(file_bytes, file_name):
    """
    Nạp dữ liệu từ bytes để tối ưu hóa việc băm dữ liệu (hashable) cho cache của Streamlit.
    Hàm đảm bảo cấu trúc dữ liệu nhất quán với quy trình trong Notebook.
    """
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_bytes)
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_bytes)
        else:
            return None
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file dữ liệu: {e}")
        return None

# Định nghĩa danh sách các biến đặc trưng đầu vào X dựa trên phân tích Notebook
FEATURE_COLUMNS = [f"X_{i}" for i in range(1, 15)]
TARGET_COLUMN = "default"

# -----------------------------------------------------------------------------
# 3. THÀNH PHẦN 1: SIDEBAR - VÙNG CẤU HÌNH & TẢI DỮ LIỆU
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    st.divider()
    
    # Khu vực tải lên tệp dữ liệu mẫu huấn luyện
    uploaded_file = st.file_uploader(
        "Tải lên dữ liệu huấn luyện mẫu", 
        type=["csv", "xlsx"],
        help="Chọn tệp dataset1.csv hoặc định dạng Excel tương tự chứa các cột X_1 đến X_14 và cột default."
    )
    
    st.divider()
    st.subheader("🤖 Tham số mô hình AI")
    st.caption("**Random Forest Classifier**")
    
    # Trích xuất cấu hình siêu tham số và cho phép tinh chỉnh động từ Sidebar
    n_estimators = st.slider(
        "Số lượng cây (n_estimators)",
        min_value=10,
        max_value=300,
        value=100,
        step=10,
        help="Số lượng cây quyết định trong rừng."
    )
    
    criterion = st.selectbox(
        "Tiêu chí đo lường chất lượng phân hoạch (criterion)",
        options=["gini", "entropy", "log_loss"],
        index=0,
        help="Hàm đo lường chất lượng phân tách của các nút."
    )
    
    max_depth = st.slider(
        "Độ sâu tối đa của cây (max_depth)",
        min_value=1,
        max_value=50,
        value=15,
        step=1,
        help="Độ sâu tối đa của các cây quyết định trong mô hình (None nếu không giới hạn)."
    )
    
    min_samples_split = st.slider(
        "Mẫu tối thiểu để phân tách nút (min_samples_split)",
        min_value=2,
        max_value=20,
        value=2,
        step=1,
        help="Số lượng mẫu tối thiểu cần thiết để phân tách một nút nội bộ."
    )
    
    random_state = st.number_input(
        "Trạng thái ngẫu nhiên (random_state)",
        min_value=0,
        max_value=9999,
        value=42,
        step=1,
        help="Đảm bảo tính tái lập của kết quả huấn luyện mô hình."
    )
    
    # Gom các tham số nâng cao vào expander gọn gàng
    with st.expander("🛠️ Tham số nâng cao"):
        min_samples_leaf = st.slider(
            "Mẫu tối thiểu tại nút lá",
            min_value=1,
            max_value=20,
            value=1,
            step=1,
            help="Số lượng mẫu tối thiểu cần có tại một nút lá."
        )
        bootstrap = st.checkbox(
            "Sử dụng Bootstrap samples",
            value=True,
            help="Có sử dụng các mẫu bootstrap khi xây dựng các cây hay không."
        )

    st.divider()
    
    # Nút hành động kích hoạt Huấn luyện mô hình duy nhất tại Sidebar
    train_clicked = st.button(
        "🚀 Huấn luyện Mô hình", 
        type="primary", 
        use_container_width=True,
        help="Bấm để thực hiện phân tách dữ liệu và huấn luyện mô hình phân loại ngẫu nhiên."
    )

# -----------------------------------------------------------------------------
# 4. THÀNH PHẦN 2: HEADER - VÙNG ĐỊNH HƯỚNG CHÍNH
# -----------------------------------------------------------------------------
st.title("🛡️ Ứng dụng Phát hiện Giao dịch Gian lận")
st.caption(
    "Ứng dụng Web thông minh hỗ trợ phân tích dữ liệu rủi ro tín dụng và tự động phát hiện "
    "các hành vi giao dịch gian lận/mất khả năng thanh toán (default) dựa trên thuật toán Học máy RandomForest."
)

# Kiểm tra trạng thái dữ liệu rỗng và điều hướng
if uploaded_file is None:
    st.info("💡 Hướng dẫn: Vui lòng tải lên tệp dữ liệu huấn luyện mẫu (.csv hoặc .xlsx) ở vùng Sidebar bên trái để bắt đầu khám phá hệ thống.")
    st.stop()

# Đọc dữ liệu thô thông qua hàm cache nếu dữ liệu đã được chọn
file_bytes = uploaded_file.getvalue()
df_raw = load_data(file_bytes, uploaded_file.name)

if df_raw is None:
    st.error("❌ Không thể đọc nội dung file. Vui lòng kiểm tra lại định dạng tệp dữ liệu đầu vào.")
    st.stop()

st.caption(f"✅ Đang sử dụng tệp dữ liệu: `{uploaded_file.name}` | Tổng số bản ghi: **{df_raw.shape[0]}** dòng và **{df_raw.shape[1]}** cột.")
st.divider()

# -----------------------------------------------------------------------------
# 5. KHỐI XỬ LÝ HUẤN LUYỆN MÔ HÌNH (LƯU VÀO SESSION STATE)
# -----------------------------------------------------------------------------
if train_clicked:
    with st.spinner("🔄 Hệ thống đang tiến hành phân tách và huấn luyện mô hình AI... Vui lòng đợi trong giây lát..."):
        # Kiểm tra xem dữ liệu có đầy đủ các biến đặc trưng và biến mục tiêu hay không
        missing_feats = [col for col in FEATURE_COLUMNS if col not in df_raw.columns]
        
        if missing_feats or (TARGET_COLUMN not in df_raw.columns):
            st.error(f"❌ Dữ liệu không tương thích! Thiếu các cột bắt buộc: {missing_feats if missing_feats else ''} {TARGET_COLUMN if TARGET_COLUMN not in df_raw.columns else ''}")
        else:
            # Tách tập X, y
            X = df_raw[FEATURE_COLUMNS]
            y = df_raw[TARGET_COLUMN]
            
            # Phân tách tập huấn luyện và tập kiểm định tương tự cấu trúc chuẩn ML
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
            
            # Khởi tạo mô hình theo các cấu hình siêu tham số tùy chọn ở Sidebar
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                bootstrap=bootstrap,
                random_state=random_state,
                n_jobs=-1
            )
            
            # Huấn luyện mô hình
            model.fit(X_train, y_train)
            
            # Đánh giá trên tập kiểm định
            y_pred = model.predict(X_test)
            y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            # Lưu trữ 3 thành phần cốt lõi vào session_state để tái sử dụng xuyên suốt các tab phụ thuộc
            st.session_state['trained_model'] = model
            st.session_state['evaluation_metrics'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'y_test': y_test,
                'y_pred': y_pred
            }
            st.session_state['feature_importances'] = pd.DataFrame({
                'Feature': FEATURE_COLUMNS,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            
            st.success("🎉 Huấn luyện mô hình thành công! Hãy chuyển sang các Tab bên dưới để kiểm tra kết quả chi tiết và dự báo.")

# -----------------------------------------------------------------------------
# 6. THÂN GIAO DIỆN CHÍNH - HỆ THỐNG PHÂN CHIA CÁC TAB CHỨC NĂNG
# -----------------------------------------------------------------------------
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
    st.subheader("📋 Phân tích Thống kê Dữ liệu Thô")
    
    # Hiển thị số liệu kích thước dữ liệu cân đối thông qua st.columns
    col_m1, col_m2, col_m3 = st.columns(3)
    file_size_mb = len(file_bytes) / (1024 * 1024)
    
    col_m1.metric("Số lượng dòng giao dịch", f"{df_raw.shape[0]:,}")
    col_m2.metric("Số lượng cột đặc trưng", f"{df_raw.shape[1]}")
    col_m3.metric("Dung lượng tệp tải lên", f"{file_size_mb:.2f} MB")
    
    st.markdown("#### 🔍 Xem trước một số bản ghi đầu tiên (Head Data)")
    st.dataframe(df_raw.head(10), use_container_width=True)
    
    st.markdown("#### 📉 Bảng mô tả thống kê các biến mô hình (X & y)")
    # Chỉ thống kê mô tả các biến thực tế được đưa vào mô hình theo quy tắc chung
    available_model_cols = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col in df_raw.columns]
    if available_model_cols:
        st.dataframe(df_raw[available_model_cols].describe(), use_container_width=True)
    else:
        st.warning("Không tìm thấy các cột đặc trưng mô hình mẫu tương thích để mô tả.")

# -----------------------------------------------------------------------------
# THÀNH PHẦN 4: TAB "TRỰC QUAN HÓA DỮ LIỆU"
# -----------------------------------------------------------------------------
with tab_viz:
    st.subheader("📊 Khám phá phân phối phân bổ các biến đặc trưng")
    
    # Ưu tiên hiển thị biến mục tiêu trước
    if TARGET_COLUMN in df_raw.columns:
        target_counts = df_raw[TARGET_COLUMN].value_counts().reset_index()
        target_counts.columns = [TARGET_COLUMN, 'Bản ghi']
        target_counts[TARGET_COLUMN] = target_counts[TARGET_COLUMN].map({0: 'Hợp lệ (0)', 1: 'Gian lận/Rủi ro (1)'})
        
        fig_target = px.bar(
            target_counts, 
            x=TARGET_COLUMN, 
            y='Bản ghi', 
            color=TARGET_COLUMN,
            title="Biểu đồ phân phối phân lớp Mục tiêu (default)",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_target.update_layout(height=350)
        st.plotly_chart(fig_target, use_container_width=True)
    
    st.markdown("#### 🧩 Trực quan tương tác phân phối các biến đầu vào (X)")
    
    # Hỗ trợ lựa chọn đa biến động nếu có quá nhiều biến đặc trưng
    default_selected_features = ["X_1", "X_2", "X_5", "X_13"] # Các biến đặc trưng mẫu nổi bật đại diện
    selected_feats = st.multiselect(
        "Chọn các biến đặc trưng X muốn trực quan hóa đồ thị phân phối:",
        options=FEATURE_COLUMNS,
        default=[f for f in default_selected_features if f in df_raw.columns]
    )
    
    if selected_feats:
        # Bố trí biểu đồ dạng lưới cân đối 2x2 hoặc tuần tự tùy thuộc số lượng lựa chọn
        grid_cols = st.columns(2)
        for idx, feat in enumerate(selected_feats):
            col_target = grid_cols[idx % 2]
            with col_target:
                if df_raw[feat].dtype in [np.float64, np.int64]:
                    # Biến liên tục sử dụng Histogram phân bổ kèm Boxplot xem ngoại lai
                    fig = px.histogram(
                        df_raw, 
                        x=feat, 
                        color=TARGET_COLUMN if TARGET_COLUMN in df_raw.columns else None,
                        marginal="box",
                        title=f"Phân phối tần suất của biến {feat}",
                        barmode="overlay",
                        height=300
                    )
                else:
                    # Biến phân loại rời rạc sử dụng đồ thị Bar chart thông thường
                    v_counts = df_raw[feat].value_counts().reset_index()
                    v_counts.columns = [feat, 'count']
                    fig = px.bar(v_counts, x=feat, y='count', title=f"Phân bố phân loại của {feat}", height=300)
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Vui lòng lựa chọn ít nhất một biến đặc trưng X để vẽ biểu đồ trực quan hóa dữ liệu.")

# -----------------------------------------------------------------------------
# THÀNH PHẦN 5: TAB "KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH"
# -----------------------------------------------------------------------------
with tab_metrics:
    st.subheader("🎯 Đánh giá hiệu năng và các chỉ số đo lường kiểm định")
    
    # Kiểm tra điều phối trạng thái session_state xem mô hình đã được train chưa
    if 'trained_model' not in st.session_state:
        st.info("💡 Trạng thái: Hệ thống chưa ghi nhận mô hình AI được huấn luyện. Vui lòng nhấn nút '🚀 Huấn luyện Mô hình' ở Sidebar bên trái để xem kết quả kiểm định chi tiết tại đây.")
    else:
        metrics = st.session_state['evaluation_metrics']
        feat_imp = st.session_state['feature_importances']
        
        # Trình bày chỉ tiêu vô hướng phân loại có giám sát sử dụng st.metric
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        col_c1.metric("Độ chính xác toàn cục (Accuracy)", f"{metrics['accuracy']:.4f}")
        col_c2.metric("Độ chính xác mô hình (Precision)", f"{metrics['precision']:.4f}")
        col_c3.metric("Độ nhạy phân loại (Recall)", f"{metrics['recall']:.4f}")
        col_c4.metric("Chỉ số F1-Score", f"{metrics['f1']:.4f}")
        
        st.divider()
        
        col_l, col_r = st.columns(2)
        
        with col_l:
            st.markdown("#### 📐 Ma trận nhầm lẫn (Confusion Matrix)")
            cm = metrics['confusion_matrix']
            # Chuyển đổi trực quan hóa sang đồ thị heatmap tương tác Plotly thay cho in văn bản tĩnh thô
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Nhãn Dự Đoán (Predicted)", y="Nhãn Thực Tế (Actual)", color="Số lượng"),
                x=['Hợp lệ (0)', 'Gian lận (1)'],
                y=['Hợp lệ (0)', 'Gian lận (1)'],
                color_continuous_scale='Blues',
                height=350
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
            st.markdown("#### 📑 Báo cáo phân loại chi tiết (Classification Report)")
            rep_df = pd.DataFrame(metrics['classification_report']).transpose()
            st.dataframe(rep_df.style.format(precision=4), use_container_width=True)
            
        with col_r:
            st.markdown("#### 🏆 Mức độ quan trọng của các đặc trưng (Feature Importance)")
            fig_imp = px.bar(
                feat_imp,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Độ quan trọng đóng góp của từng cột biến X',
                color='Importance',
                color_continuous_scale='Viridis',
                height=450
            )
            fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_imp, use_container_width=True)

# -----------------------------------------------------------------------------
# THÀNH PHẦN 6: TAB "SỬ DỤNG MÔ HÌNH"
# -----------------------------------------------------------------------------
with tab_inference:
    st.subheader("🔮 Chẩn đoán và Dự đoán Rủi ro Giao dịch Mới")
    
    if 'trained_model' not in st.session_state:
        st.info("💡 Trạng thái: Tính năng dự báo đang bị khóa. Bạn cần bấm nút '🚀 Huấn luyện Mô hình' ở Sidebar bên trái trước để kích hoạt bộ suy luận thông minh.")
    else:
        model = st.session_state['trained_model']
        
        # Lựa chọn 2 chế độ bằng st.radio ở đầu tab
        mode = st.radio(
            "Chọn phương thức nhập dữ liệu đầu vào giao dịch:",
            options=["Chế độ 1: Nhập thông số trực tiếp qua Form mẫu", "Chế độ 2: Tải file dữ liệu kiểm thử hàng loạt (Batch Prediction)"],
            horizontal=True
        )
        
        st.divider()
        
        # ---------------------------------------------------------------------
        # CHẾ ĐỘ 1 — NHẬP TRỰC TIẾP
        # ---------------------------------------------------------------------
        if "Chế độ 1" in mode:
            st.markdown("#### 📝 Cấu hình giá trị thuộc tính cho một giao dịch đơn lẻ")
            
            # Sử dụng st.form bao quanh tất cả các biến đầu vào để kiểm soát rerun giao diện
            with st.form("inference_form"):
                st.markdown("##### Cung cấp thông số kỹ thuật (Cột X_1 đến X_14):")
                
                # Sắp xếp các widget nhập liệu dạng lưới để tối ưu diện tích hiển thị trực quan
                inf_cols = st.columns(3)
                input_data = {}
                
                for idx, feat in enumerate(FEATURE_COLUMNS):
                    col_target = inf_cols[idx % 3]
                    with col_target:
                        # Lấy giá trị thống kê thực tế từ dữ liệu huấn luyện làm mốc mặc định (trung vị, min, max)
                        default_val = float(df_raw[feat].median())
                        min_val = float(df_raw[feat].min())
                        max_val = float(df_raw[feat].max())
                        
                        input_data[feat] = st.number_input(
                            f"Nhập thông số {feat}",
                            min_value=min_val - 100.0,
                            max_value=max_val + 100.0,
                            value=default_val,
                            format="%.6f",
                            help=f"Giá trị thực tế trong tập mẫu dao động từ {min_val:.4f} đến {max_val:.4f}. Giá trị mặc định là trung vị."
                        )
                        
                submit_pred = st.form_submit_button("🔍 Tiến hành Dự báo kết quả", type="primary", use_container_width=True)
                
            if submit_pred:
                # Tổ chức cấu trúc biến đầu vào chính xác
                X_single = pd.DataFrame([input_data])[FEATURE_COLUMNS]
                
                # Thực thi mô hình đã lưu
                pred_class = model.predict(X_single)[0]
                pred_proba = model.predict_proba(X_single)[0] if hasattr(model, "predict_proba") else None
                
                st.markdown("#### 🏁 Kết quả dự đoán thu được:")
                col_res1, col_res2 = st.columns(2)
                
                if pred_class == 1:
                    col_res1.error("🚨 CẢNH BÁO: Giao dịch có nguy cơ GIAN LẬN / RỦI RO CAO! (default = 1)")
                else:
                    col_res1.success("🟢 AN TOÀN: Giao dịch được thẩm định HỢP LỆ. (default = 0)")
                    
                if pred_proba is not None:
                    prob_risk = pred_proba[1] * 100
                    col_res2.metric("Xác suất rủi ro gian lận", f"{prob_risk:.2f} %")
                    st.progress(pred_proba[1])
                
        # ---------------------------------------------------------------------
        # CHẾ ĐỘ 2 — TẢI FILE THEO CẤU TRÚC X_TEST
        # ---------------------------------------------------------------------
        else:
            st.markdown("#### 📂 Dự báo tự động hàng loạt bằng File mẫu dữ liệu mới")
            st.caption("Yêu cầu file tải lên phải cấu trúc chứa các cột đặc trưng độc lập: từ `X_1` tới `X_14`.")
            
            batch_file = st.file_uploader(
                "Tải lên tệp dữ liệu kiểm thử mới để chấm điểm", 
                type=["csv", "xlsx"],
                key="batch_uploader"
            )
            
            if batch_file is not None:
                # Đọc dữ liệu kiểm thử
                df_batch = load_data(batch_file.getvalue(), batch_file.name)
                
                if df_batch is not None:
                    # Kiểm tra schema (thiếu/thừa cột đặc trưng X bắt buộc)
                    missing_batch_feats = [col for col in FEATURE_COLUMNS if col not in df_batch.columns]
                    
                    if missing_batch_feats:
                        st.error(f"❌ Cấu trúc File không hợp lệ! Thiếu các cột đặc trưng bắt buộc sau: {missing_batch_feats}")
                    else:
                        X_batch = df_batch[FEATURE_COLUMNS]
                        
                        # Dự báo đồng loạt không cần huấn luyện lại
                        batch_preds = model.predict(X_batch)
                        df_result = df_batch.copy()
                        df_result['Predicted_Default'] = batch_preds
                        
                        if hasattr(model, "predict_proba"):
                            batch_probs = model.predict_proba(X_batch)[:, 1]
                            df_result['Fraud_Probability'] = batch_probs
                        
                        st.markdown("##### 📈 Bảng kết quả chấm điểm rủi ro giao dịch hàng loạt:")
                        # Hiển thị trong container cuộn gọn gàng
                        with st.container(height=350):
                            st.dataframe(df_result, use_container_width=True)
                        
                        # Thống kê nhanh tỷ lệ cảnh báo
                        total_cnt = len(df_result)
                        fraud_cnt = int((batch_preds == 1).sum())
                        st.warning(f"🔔 Phát hiện hệ thống: Có **{fraud_cnt}/{total_cnt}** giao dịch được phân loại rủi ro tiềm ẩn (Chiếm tỷ lệ: {(fraud_cnt/total_cnt)*100:.2f}%).")
                        
                        # Nút xuất và tải file kết quả đầu ra định dạng CSV chuẩn utf-8-sig
                        csv_output = df_result.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 Tải xuống tệp kết quả dự báo (.CSV)",
                            data=csv_output,
                            file_name="ket_qua_du_bao_gian_lan_hang_loat.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
