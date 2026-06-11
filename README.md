# 🛡️ Ứng dụng Web Phát hiện Giao dịch Gian lận (Mô hình Random Forest)

Ứng dụng Web được đóng gói hoàn chỉnh từ quy trình nghiên cứu học máy của Notebook phòng chống gian lận tài chính (`phat_hien_giao_dich_gian_lan.ipynb`). Mã nguồn ứng dụng đã được sửa đổi triệt để lỗi luồng dữ liệu thô bằng `io.BytesIO`.

## 📌 Khối chức năng phân chia theo Tab
1. **📊 Tổng quan dữ liệu:** Hiển thị tổng dòng/cột, xem trước dữ liệu mẫu và tính toán bảng thống kê mô tả (`describe`) tập trung duy nhất vào các biến phục vụ mô hình.
2. **📈 Trực quan hóa dữ liệu:** Vẽ phân phối của lớp mục tiêu rủi ro gian lận (`default`) trước, sau đó biểu diễn đồ thị Histogram/Boxplot tương tác động cho các biến đặc trưng đầu vào X do người dùng chỉ định.
3. **🎯 Kết quả & Kiểm định:** Đánh giá hiệu năng của mô hình Phân loại rừng ngẫu nhiên bao gồm Accuracy, Precision, Recall, F1-Score, vẽ Ma trận nhầm lẫn tương tác ngẫu nhiên và mức độ quan trọng thuộc tính (Feature Importance).
4. **🔮 Sử dụng mô hình:** Dự báo giao dịch đơn lẻ qua Form nhập liệu hoặc tải tệp Excel/CSV lên để thực hiện quét chấm điểm hàng loạt (Batch Inference) mà không cần huấn luyện lại.

## 💾 Cấu trúc tệp dữ liệu đầu vào kỳ vọng
Hệ thống yêu cầu các tệp dữ liệu mẫu phải chứa định dạng trường thông tin khớp cấu trúc:
- **Biến độc lập (X):** Các cột từ `X_1` tới `X_14` chứa giá trị số đo lường đặc trưng giao dịch.
- **Biến mục tiêu (y):** Cột mang tên `default` nhận giá trị nhị phân `0` (Hợp lệ) hoặc `1` (Gian lận).

## 🛠️ Hướng dẫn khởi chạy ứng dụng
Chạy các lệnh sau tại cửa sổ Terminal của thư mục chứa dự án:
```bash
# Bước 1: Cài đặt gói thư viện bắt buộc
pip install -r requirements.txt

# Bước 2: Kích hoạt ứng dụng Web Streamlit
streamlit run app.py
