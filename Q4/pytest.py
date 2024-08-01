import numpy as np
import requests
from PIL import Image
import io

def preprocess_image(image_path):
    # Đọc ảnh từ tệp PNG
    with Image.open(image_path) as img:
        # Chuyển đổi ảnh thành ảnh grayscale (1 kênh)
        img = img.convert('L')
        # Thay đổi kích thước nếu cần (MNIST là 28x28)
        img = img.resize((28, 28))
        # Chuyển đổi ảnh thành mảng numpy
        img_array = np.array(img)
        # Thêm chiều cho batch (1 ảnh)
        img_array = img_array[None, :, :]
        return img_array

def predict(image_path):
    # Xử lý ảnh
    img_array = preprocess_image(image_path)
    
    # Gửi yêu cầu POST đến API
    url = 'http://127.0.0.1:5000/predict'
    response = requests.post(url, json={'images': img_array.tolist()})
    
    if response.status_code == 200:
        # In kết quả dự đoán
        print('Prediction:', response.json())
    else:
        print('Error:', response.json())

# Thay thế với đường dẫn đến ảnh PNG của bạn
image_path = 'E:\Interview_QAI\Q4\image.png'
predict(image_path)
