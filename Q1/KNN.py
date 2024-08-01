import numpy as np
import struct

# Load mnist images
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows, cols)
    return images

# Load mnist labels
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

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn(train_images, train_labels, test_image, k):
    distances = np.array([euclidean_distance(train_image, test_image) for train_image in train_images])
    k_nearest_indices = distances.argsort()[:k]
    k_nearest_labels = train_labels[k_nearest_indices]
    unique, counts = np.unique(k_nearest_labels, return_counts=True)
    return unique[np.argmax(counts)]

# Random 1000 samples to find the best k
def find_best_k(train_images, train_labels, test_images, test_labels, k_values, num_samples=1000):
    accuracies = []
    random_indices = np.random.choice(test_images.shape[0], num_samples, replace=False)

    for k in k_values:
        correct_predictions = 0
        for i in random_indices:
            test_image = test_images[i]
            predicted_label = knn(train_images, train_labels, test_image, k)
            true_label = test_labels[i]
            if predicted_label == true_label:
                correct_predictions += 1
        
        accuracy = correct_predictions / num_samples
        accuracies.append(accuracy)    
    best_k = k_values[np.argmax(accuracies)]
    return best_k

k_values = range(1, 11)  # Test k from 1 to 10

best_k = find_best_k(train_images, train_labels, test_images, test_labels, k_values)

# Evaluate the model on a subset of the test set using the best k
correct_predictions = 0
num_samples = 50 # Number of samples to test

random_indices = np.random.choice(test_images.shape[0], num_samples, replace=False)

for i in random_indices:
    test_image = test_images[i]
    predicted_label = knn(train_images, train_labels, test_image, k=best_k)
    true_label = test_labels[i]
    print(f'Predicted label: {predicted_label}, True label: {true_label}')
    if predicted_label == true_label:
        correct_predictions += 1

accuracy = correct_predictions / num_samples
print(f'Accuracy on {num_samples} random samples using best k={best_k}: {accuracy:.2f}')
