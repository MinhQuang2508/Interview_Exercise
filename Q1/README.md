# Phân loại Chữ số MNIST bằng Machine Learning

## Giới thiệu

Bài tập này tập trung vào việc xây dựng các mô hình phân loại chữ số từ bộ dữ liệu MNIST mà không sử dụng các thư viện học sâu. Chúng ta sử dụng các thuật toán Machine Learning truyền thống và chỉ dùng `numpy` để thực hiện các phép toán và tính toán cần thiết. Mục tiêu là phân loại các hình ảnh chữ số viết tay từ 0 đến 9 thành các lớp tương ứng.

## Các Thành Phần

1. **`knn_model.py`**: Script để xây dựng và đánh giá mô hình K-Nearest Neighbors (KNN) cho bài toán phân loại chữ số MNIST. Script này cũng bao gồm phần tìm giá trị \( k \) tốt nhất cho mô hình KNN.

2. **`svm_model.py`**: Script để huấn luyện và đánh giá mô hình Support Vector Machine (SVM) trên dữ liệu MNIST. Script này sử dụng thư viện `scikit-learn`.

## Cấu Trúc Dự Án

- **`knn_model.py`**:
  - Xây dựng mô hình KNN từ đầu bằng cách sử dụng `numpy`.
  - Tìm giá trị \( k \) tối ưu cho mô hình KNN bằng cách thử nghiệm với nhiều giá trị \( k \) khác nhau.
  - Đánh giá mô hình KNN với giá trị \( k \) tốt nhất tìm được.

- **`svm_model.py`**:
  - Huấn luyện mô hình SVM với kernel tuyến tính.
  - Đánh giá mô hình SVM và in ra độ chính xác.
  ## Result
![KNN](E:\Interview_QAI\Q1\KNN.png)

![SVM Model](E:\Interview_QAI\Q1\svm.png)


