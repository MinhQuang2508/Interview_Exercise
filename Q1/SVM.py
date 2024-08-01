import numpy as np
import struct
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Load mnist image
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images
# Load mnist label
def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Load data
train_images = load_mnist_images(r'E:\Interview_QAI\mnist_dataset\train-images.idx3-ubyte')
train_labels = load_mnist_labels(r'E:\Interview_QAI\mnist_dataset\train-labels.idx1-ubyte')
test_images = load_mnist_images(r'E:\Interview_QAI\mnist_dataset\t10k-images.idx3-ubyte')
test_labels = load_mnist_labels(r'E:\Interview_QAI\mnist_dataset\t10k-labels.idx1-ubyte')


# Normalize the images
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Flatten images for SVM
train_images_flat = train_images.reshape(train_images.shape[0], -1)
test_images_flat = test_images.reshape(test_images.shape[0], -1)

# Standardize features
scaler = StandardScaler()
train_images_flat = scaler.fit_transform(train_images_flat)
test_images_flat = scaler.transform(test_images_flat)

# Train SVM classifier
svm_clf = svm.SVC(gamma='scale', kernel='linear')
svm_clf.fit(train_images_flat, train_labels)

# Save the model
joblib.dump(svm_clf, 'svm_mnist_model.pkl')

# Save the scaler parameters
np.save('scaler_params.npy', {'mean': scaler.mean_, 'scale': scaler.scale_})

# Predict and evaluate SVM
svm_predictions = svm_clf.predict(test_images_flat)
svm_accuracy = accuracy_score(test_labels, svm_predictions)
print(f'SVM Accuracy: {svm_accuracy:.2f}')
