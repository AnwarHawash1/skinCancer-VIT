from flask import Flask, request, render_template, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import ViTFeatureExtractor
from model import CustomViTModel  # Assuming you've saved your model architecture in a file named 'model.py'

app = Flask(__name__)

# Load the ViT feature extractor
model_name = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

# Load the skin cancer classification model
num_classes = 7
model = CustomViTModel(num_classes=num_classes)
model.load_state_dict(torch.load("skin_cancer_model.pth"))
model.eval()

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Preprocessing function
def preprocessing(image):
    image = Image.open(image)
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

# Thresholds and classes
thresholds = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
classes = ['benign_keratosis-like_lesions', 'basal_cell_carcinoma', 'actinic_keratoses', 'vascular_lesions', 'melanocytic_Nevi', 'melanoma', 'dermatofibroma']

# Model prediction function
def model_predict(image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        return probabilities.squeeze().tolist()

@app.route('/')
def index():
    return render_template('index.html', appName="Skin Cancer Classification")

@app.route('/predictApi', methods=["POST"])
def api():
    try:
        if 'fileup' not in request.files:
            return jsonify({'Error': "Please try again. The Image doesn't exist"})
        image = request.files.get('fileup')
        image_tensor = preprocessing(image)
        probabilities = model_predict(image_tensor)
        max_prob = max(probabilities)
        if max_prob < 0.9:
            return jsonify({'Error': 'No confidence in any class prediction.'})
        prediction_idx = probabilities.index(max_prob)
        prediction = classes[prediction_idx]
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'Error': 'An error occurred', 'Message': str(e)})

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            image = request.files['fileup']
            image_tensor = preprocessing(image)
            probabilities = model_predict(image_tensor)
            max_prob = max(probabilities)
            if max_prob < 0.9:
                return render_template('index.html', prediction='No confidence in any class prediction.', appName="Skin Cancer Classification")
            else:
                prediction_idx = probabilities.index(max_prob)
                prediction = classes[prediction_idx]
                return render_template('index.html', prediction=prediction, appName="Skin Cancer Classification")
        except Exception as e:
            return render_template('index.html', prediction='Error: ' + str(e), appName="Skin Cancer Classification")
    else:
        return render_template('index.html', appName="Skin Cancer Classification")

if __name__ == '__main__':
    app.run(debug=True)
