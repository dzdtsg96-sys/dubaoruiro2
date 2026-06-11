# 🛡️ Ứng dụng Web Phát hiện Giao dịch Gian lận và Đánh giá Tín dụng

Ứng dụng web tương tác thông minh được chuyển đổi từ Notebook nghiên cứu và thử nghiệm mô hình Học máy (`phat_hien_giao_dich_gian_lan.ipynb`). Hệ thống được xây dựng hoàn chỉnh trên nền tảng **Streamlit** kết hợp thư viện **Scikit-Learn** và **Plotly**, cho phép người dùng vận hành, phân tích dữ liệu, tinh chỉnh tham số huấn luyện trực quan và dự đoán rủi ro gian lận giao dịch theo thời gian thực hoặc hàng loạt.

## 📌 Các Tính Năng Chính Của Ứng Dụng

Ứng dụng được thiết kế phân chia bố cục giao diện khoa học (zoning) bao gồm 4 Tab chính tại phân vùng nội dung chính độc lập:
1. **📊 Tab Tổng quan dữ liệu:** Xem nhanh thông số kích thước tệp tải lên, cấu trúc dữ liệu thô (Head) và phân tích các chỉ số thống kê mô tả (`describe`) tập trung duy nhất vào bộ biến đặc trưng đầu vào mô hình.
2. **📈 Tab Trực quan hóa dữ liệu:** Thể hiện trực quan hóa phân bổ của lớp mục tiêu rủi ro gian lận (`default`) và đồ thị phân phối tương tác (Histogram/Boxplot) của các biến đặc trưng đầu vào liên tục/rời rạc do người dùng tùy chọn chọn lựa.
3. **🎯 Tab Kết quả & Kiểm định:** Đánh giá hiệu năng chi tiết của thuật toán phân loại Rừng ngẫu nhiên (Random Forest Classifier). Tự động kết xuất các độ đo cốt lõi bao gồm Accuracy, Precision, Recall, F1-Score, đồ thị Ma trận nhầm lẫn (Confusion Matrix Heatmap) cùng biểu đồ phân tích độ quan trọng của các biến độc lập đóng góp vào mô hình.
4. **🔮 Tab Sử dụng mô hình dự báo:**
   - *Chế độ nhập thông số trực tiếp:* Cho phép điền các giá trị đặc trưng giao dịch đơn lẻ qua biểu mẫu trực quan (giá trị khởi tạo mặc định bằng số trung vị của tập mẫu) để kiểm tra mức độ an toàn hoặc cảnh báo gian lận kèm xác suất phần trăm rủi ro.
   - *Chế độ tải file hàng loạt:* Hỗ trợ tải một tệp dữ liệu kiểm thử mới, tự động kiểm tra đối chiếu schema biến và cho phép tải xuống file CSV kết quả dự báo chấm điểm rủi ro.

## 💾 Cấu Trúc File Dữ Liệu Đầu Vào Kỳ Vọng

Ứng dụng hỗ trợ đọc các định dạng tệp tin `.csv` hoặc `.xlsx`. Cấu trúc bảng dữ liệu cần đảm bảo bao gồm ít nhất các cột trường thông tin sau:
- **Biến độc lập (X):** `X_1`, `X_2`, `X_3`, `X_4`, `X_5`, `X_6`, `X_7`, `X_8`, `X_9`, `X_10`, `X_11`, `X_12`, `X_13`, `X_14` (Chứa các giá trị số thực/số nguyên đo lường đặc trưng giao dịch).
- **Biến mục tiêu (y - Chỉ cần cho tập huấn luyện mẫu ban đầu):** Cột mang tên `default` chứa nhãn nhị phân nhận giá trị `0` (Giao dịch an toàn / hợp lệ) hoặc `1` (Giao dịch gian lận / mất khả năng thanh toán).

## 🛠️ Hướng Dẫn Cài Đặt và Khởi Chạy Ứng Dụng

Thực hiện theo các bước đơn giản sau để thiết lập môi trường và khởi chạy ứng dụng web Streamlit trên máy cục bộ của bạn:

**Bước 1: Tải mã nguồn về thư mục làm việc**
Đảm bảo bạn đã lưu hai tệp `app.py` và `requirements.txt` vào cùng một thư mục làm việc.

**Bước 2: Cài đặt các thư viện cần thiết qua Terminal/Command Prompt**
Chạy câu lệnh dưới đây để tự động cài đặt toàn bộ danh sách gói thư viện Python theo đúng phiên bản khuyến nghị:
```bash
pip install -r requirements.txt
