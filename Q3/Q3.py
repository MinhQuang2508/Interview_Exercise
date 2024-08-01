import numpy as np
import struct

# Load MNIST data functions
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images

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

# Normalize and reshape images
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255
train_images = train_images.reshape((train_images.shape[0], -1))
test_images = test_images.reshape((test_images.shape[0], -1))

# Euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

# Triplet Loss function
def triplet_loss(anchor, positive, negative, alpha=0.2):
    positive_distance = euclidean_distance(anchor, positive)
    negative_distance = euclidean_distance(anchor, negative)
    return np.mean(np.maximum(positive_distance - negative_distance + alpha, 0))

# Build a simple model with 2 layers
def build_model(input_dim):
    np.random.seed(0)
    weights = {
        'W1': np.random.randn(input_dim, 64) * 0.01,  # First layer weights
        'b1': np.zeros(64),                           # First layer biases
        'W2': np.random.randn(64, 64) * 0.01,         # Second layer weights
        'b2': np.zeros(64)                            # Second layer biases
    }
    return weights

# Forward pass
def forward_pass(x, weights):
    z1 = np.dot(x, weights['W1']) + weights['b1']
    a1 = np.maximum(0, z1)  # ReLU activation
    z2 = np.dot(a1, weights['W2']) + weights['b2']
    return a1, z2

# Compute gradients using backpropagation
def compute_gradients(anchor, positive, negative, weights, alpha=0.2):
    a1_anchor, z2_anchor = forward_pass(anchor, weights)
    a1_positive, z2_positive = forward_pass(positive, weights)
    a1_negative, z2_negative = forward_pass(negative, weights)

    positive_distance = euclidean_distance(z2_anchor, z2_positive)
    negative_distance = euclidean_distance(z2_anchor, z2_negative)
    
    grad = {
        'W1': np.zeros_like(weights['W1']),
        'b1': np.zeros_like(weights['b1']),
        'W2': np.zeros_like(weights['W2']),
        'b2': np.zeros_like(weights['b2'])
    }
    
    if np.any(positive_distance - negative_distance + alpha > 0):
        mask = positive_distance - negative_distance + alpha > 0
        dL_dz2_anchor = 2 * (z2_anchor[mask] - z2_positive[mask]) - 2 * (z2_anchor[mask] - z2_negative[mask])
        dL_dz2_positive = -2 * (z2_positive[mask] - z2_anchor[mask])
        dL_dz2_negative = 2 * (z2_negative[mask] - z2_anchor[mask])
        
        dL_da1_anchor = np.dot(dL_dz2_anchor, weights['W2'].T)
        dL_dz1_anchor = dL_da1_anchor * (a1_anchor[mask] > 0)
        
        grad['W2'] = np.dot(a1_anchor[mask].T, dL_dz2_anchor) / len(anchor)
        grad['b2'] = np.sum(dL_dz2_anchor, axis=0) / len(anchor)
        grad['W1'] = np.dot(anchor.T, dL_dz1_anchor) / len(anchor)
        grad['b1'] = np.sum(dL_dz1_anchor, axis=0) / len(anchor)
    
    return grad

# Training the model
def train_model(train_images, train_labels, num_epochs=10, batch_size=64, learning_rate=0.001):
    input_dim = train_images.shape[1]
    weights = build_model(input_dim)
    
    for epoch in range(num_epochs):
        indices = np.random.permutation(len(train_images))
        for start in range(0, len(train_images), batch_size):
            end = min(start + batch_size, len(train_images))
            batch_indices = indices[start:end]
            batch_images = train_images[batch_indices]
            batch_labels = train_labels[batch_indices]
            
            for i in range(len(batch_images)):
                anchor = batch_images[i].reshape(1, -1)
                positive = batch_images[i].reshape(1, -1)  # Simplified selection for demo
                negative_indices = np.where(batch_labels != batch_labels[i])[0]
                if len(negative_indices) > 0:
                    negative = batch_images[np.random.choice(negative_indices)].reshape(1, -1)
                else:
                    continue

                gradients = compute_gradients(anchor, positive, negative, weights)
                
                for param in weights:
                    weights[param] -= learning_rate * gradients[param]
                
                loss = triplet_loss(forward_pass(anchor, weights)[1], 
                                    forward_pass(positive, weights)[1], 
                                    forward_pass(negative, weights)[1])
                print(f'Epoch {epoch + 1}, Loss: {loss}')
    
    return weights

# Save model weights
def save_model(weights, filename):
    np.savez(filename, **weights)

# Load model weights
def load_model(filename):
    data = np.load(filename)
    weights = {key: data[key] for key in data.files}
    return weights

# Predict function
def predict(image, weights):
    _, z2 = forward_pass(image.reshape(1, -1), weights)
    return z2

def convert_to_features(images, weights):
    features = []
    for image in images:
        features.append(predict(image, weights).flatten())
    return np.array(features)

# Find closest image
def find_closest_image(image, train_features):
    features = predict(image, weights).flatten()
    distances = np.linalg.norm(train_features - features, axis=1)
    return np.argmin(distances)

# Evaluate model
def evaluate_model(test_images, test_labels, train_images, train_labels, weights):
    train_features = convert_to_features(train_images, weights)
    correct = 0
    for i in range(len(test_images)):
        closest_index = find_closest_image(test_images[i], train_features)
        if test_labels[i] == train_labels[closest_index]:
            correct += 1
    accuracy = correct / len(test_images)
    return accuracy

# Train the model
weights = train_model(train_images, train_labels)

# Save the trained model
save_model(weights, 'mnist_triplet_model2.npz')

# Load the model for evaluation
weights = load_model('mnist_triplet_model2.npz')

# Evaluate the model
accuracy = evaluate_model(test_images, test_labels, train_images, train_labels, weights)
print(f'Model accuracy: {accuracy * 100:.2f}%')
