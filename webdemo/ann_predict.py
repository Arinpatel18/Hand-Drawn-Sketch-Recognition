# ann_predict.py
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from ann import NeuralNetwork

# Load model and tools
ANN = NeuralNetwork.load("ann_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("label_encoder.pkl")
label_data = pd.read_csv("data_codes.csv")

# Pretrained ResNet50 for feature extraction
resnet = nn.Sequential(*list(nn.Sequential(*list(torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).children())[:-1])))
resnet.eval()

def extract_features(image_path, model=resnet):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image)
    return features.squeeze(0).flatten().tolist()

def predict_from_image(image_path):
    sample_data_list = extract_features(image_path)
    X_scaled = scaler.transform([sample_data_list])
    y_pred_probs = ANN.predict(X_scaled)
    y_pred_label = np.argmax(y_pred_probs, axis=1)
    decoded = encoder.inverse_transform(y_pred_label)[0]

    # Map to human-readable label
    row = label_data[label_data['encoded_part'] == decoded]
    return row['extracted_part'].iloc[0] if not row.empty else decoded
