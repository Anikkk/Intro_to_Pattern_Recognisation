import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

class ProduceClassifier(nn.Module):
    def __init__(self, num_classes=4, pretrained=False, freeze_backbone=False):
        super(ProduceClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
            for param in self.resnet.fc.parameters():
                param.requires_grad = True
            print("Backbone frozen - only training final classification layer")
    
    def forward(self, x):
        return self.resnet(x)

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Function to load a model
def load_model(model_path, num_classes, class_names):
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model = ProduceClassifier(num_classes=num_classes, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"✅ Model {model_path} loaded successfully!")
        return model, class_names
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None, class_names

# Load produce classifier
produce_model_path = 'model/best_produce_classifier.pth'
produce_class_names = ['Broccoli', 'Mango', 'Nut', 'Pepper']
produce_model, produce_class_names = load_model(produce_model_path, num_classes=4, class_names=produce_class_names)

print(f"Produce class names: {produce_class_names}")

# Load variation classifiers
variation_model_map = {
    'Mango': 'model/mango_variation_classifier.pth',
    'Pepper': 'model/pepper_variation_classifier.pth',
    'Nut': 'model/nut_variation_classifier.pth',
    'Broccoli': 'model/broccoli_variation_classifier.pth'
}
variation_class_names = ['chopped', 'sliced', 'whole']
variation_models = {}

for produce, path in variation_model_map.items():
    if os.path.exists(path):
        model, _ = load_model(path, num_classes=3, class_names=variation_class_names)
        variation_models[produce] = model
    else:
        print(f"Variation model {path} not found.")

# Prediction function
def predict(image: Image.Image, model, class_names) -> tuple[str, float]:
    try:
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            print(f"Probabilities: {probabilities}")
            confidence, predicted = torch.max(probabilities, 1)
            class_index = predicted.item()
        return class_names[class_index], confidence.item()
    except Exception as e:
        print(f"Prediction error: {e}")
        return "unknown", 0.0

# Two-stage prediction
def classify_image(image: Image.Image) -> tuple[dict, dict]:
    produce_result = {'class': 'unknown', 'confidence': 0.0}
    variation_result = {'class': 'unknown', 'confidence': 0.0}

    if produce_model is None:
        print("Produce classifier not loaded.")
        return produce_result, variation_result

    # First stage: Predict produce type
    produce_class, produce_confidence = predict(image, produce_model, produce_class_names)
    produce_result = {'class': produce_class, 'confidence': produce_confidence}

    # Second stage: Predict variation (if applicable)
    variation_model = variation_models.get(produce_class)
    if variation_model:
        variation_class, variation_confidence = predict(image, variation_model, variation_class_names)
        variation_result = {'class': variation_class, 'confidence': variation_confidence}
    else:
        print(f"No variation model for {produce_class}.")

    return produce_result, variation_result