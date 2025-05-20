import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np

# Load pre-trained ResNet-50
resnet = models.resnet50(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

def extract_features(image_path, model=resnet):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        features = model(image)
    return features.squeeze().numpy().flatten()

import joblib
from linear_svm_cnn import OneVsRestSVM

# # Extract features from a sample image
# sample_data = extract_features('/content/drive/MyDrive/prml/png/arm/403.png')

# # Load the trained model and predict
# model = OneVsRestSVM.load_model('onevsrest_svm_model.pkl')
# y_pred = model.predict(sample_data)
# print("Predicted class:", y_pred[0])

def predict_svm(image_path):
    sample_data = extract_features(image_path)
    model = OneVsRestSVM.load_model('onevsrest_svm_model.pkl')

    # Predict encoded label
    encoded_pred = model.predict(sample_data)[0]

    # Load CSV to map to human-readable label
    label_data = pd.read_csv('data_codes.csv')
    row = label_data[label_data['encoded_part'] == encoded_pred]
    return row['extracted_part'].iloc[0] if not row.empty else str(encoded_pred)