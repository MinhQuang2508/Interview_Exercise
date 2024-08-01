# Phân Loại Ký Tự Quang Học với Deep Learning và Triplet Loss

## Giới Thiệu

Triển khai một mô hình Deep Learning để phân biệt các ký tự quang học trên tập dữ liệu MNIST. Mô hình sử dụng NumPy và Triplet Loss để huấn luyện và đánh giá hiệu suất. So sánh hiệu suất của mô hình Deep Learning với các phương pháp Machine Learning.

## Mục Tiêu

- Xây dựng một mô hình Deep Learning với Triplet Loss để phân biệt ký tự quang học.
- So sánh hiệu suất của mô hình Deep Learning với các phương pháp Machine Learning.
- Phân tích ưu và nhược điểm của Deep Learning trong bài toán MNIST.

## Phương Pháp

### **Mô Hình Deep Learning**

- **Kiến Trúc Mô Hình**: Mô hình bao gồm hai lớp ẩn, mỗi lớp sử dụng các trọng số và độ lệch. Lớp đầu tiên sử dụng hàm kích hoạt ReLU, trong khi lớp thứ hai không sử dụng hàm kích hoạt.
- **Hàm Mất Mát (Loss Function)**: Sử dụng Triplet Loss để tối ưu hóa mô hình. Triplet Loss giúp học các đặc trưng sao cho khoảng cách giữa các điểm cùng lớp gần hơn so với khoảng cách giữa các điểm khác lớp.

### **Dữ Liệu MNIST**

- **Tập Dữ Liệu**: MNIST là một tập dữ liệu nổi tiếng bao gồm 60,000 hình ảnh huấn luyện và 10,000 hình ảnh kiểm tra của các chữ số từ 0 đến 9. Mỗi hình ảnh có kích thước 28x28 pixel.
- **Tiền Xử Lý Dữ Liệu**: Dữ liệu hình ảnh được chuẩn hóa và chuyển đổi thành các vector để phù hợp với đầu vào của mô hình.

## Phân Tích So Sánh

### **Ưu Điểm của Deep Learning**

- **Khả Năng Học Các Đặc Trưng Phức Tạp**: Mô hình Deep Learning có khả năng học và trích xuất các đặc trưng phức tạp từ dữ liệu, mà không cần phải thiết kế đặc trưng thủ công.
- **Khả Năng Tổng Quát**: Với kích thước tập dữ liệu lớn hơn, mô hình Deep Learning có thể tổng quát tốt hơn và học được các mối quan hệ phức tạp giữa các đặc trưng.

### **Nhược Điểm của Deep Learning**

- **Yêu Cầu Tài Nguyên Cao**: Mô hình Deep Learning thường yêu cầu nhiều tài nguyên tính toán và thời gian huấn luyện lâu hơn.
- **Hiệu Suất Với Dữ Liệu Nhỏ**: Trong trường hợp dữ liệu nhỏ hoặc không đủ lớn, mô hình Deep Learning có thể không đạt được hiệu suất tốt hơn so với các phương pháp Machine Learning truyền thống.

## Kết Quả

- **Hiệu Suất của Mô Hình Deep Learning**: Mô hình đạt được độ chính xác khoảng 83% trên tập kiểm tra MNIST.
  
![DeepLearning](E:\Interview_QAI\Q3\model2.png)

- **Hiệu Suất của Machine Learning**: Phương pháp Machine Learning truyền thống đạt được độ chính xác khoảng 93% trên cùng tập dữ liệu.
