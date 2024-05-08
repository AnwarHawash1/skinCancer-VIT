from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor

app = Flask(__name__)

# Load the pre-trained ViT model and modify the classification head
model_name = "google/vit-base-patch16-224-in21k"
model = ViTForImageClassification.from_pretrained(model_name)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
num_classes = 7
model.classifier = torch.nn.Linear(model.config.hidden_size, num_classes)

# Load the trained weights
model.load_state_dict(torch.load("skin_cancer_model.pth", map_location=torch.device('cpu')))
model.eval()

# Define preprocessing function
def preprocess_image(image_bytes):
    image = Image.open(image_bytes)
    image = transforms.Resize((224, 224))(image)
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0)

# Define prediction function
def predict(image_bytes):
    tensor = preprocess_image(image_bytes)
    with torch.no_grad():
        outputs = model(tensor)
    predicted_class = torch.argmax(outputs.logits).item()
    return predicted_class

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    prediction = predict(file)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
