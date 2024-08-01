from flask import Flask, request, jsonify
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the model and scaler
model = joblib.load(r'E:\Interview_QAI\svm_mnist_model.pkl')
scaler_params = np.load(r'E:\Interview_QAI\scaler_params.npy', allow_pickle=True).item()
scaler = StandardScaler()
scaler.mean_ = scaler_params['mean']
scaler.scale_ = scaler_params['scale']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the incoming JSON data
        data = request.get_json()
        images = np.array(data['images'])
        
        # Normalize and flatten the images
        images = images.astype('float32') / 255
        images_flat = images.reshape(images.shape[0], -1)
        images_flat = scaler.transform(images_flat)
        
        # Make predictions
        predictions = model.predict(images_flat)
        
        return jsonify(predictions.tolist())
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
