import numpy as np
import joblib # Import joblib
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd

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
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image)
    features = features.squeeze(0)
    sample_data_list = features.flatten().tolist()
    sample_data_list = np.array(sample_data_list)
    return sample_data_list

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_logistic_regression(image_path):
    # Load the trained model
    all_theta = joblib.load('trained_model_logistic.joblib')
    # Extract features from the image
    features = extract_features(image_path, resnet)
    scaler = joblib.load('scaler_logistic.pkl')
    features = scaler.transform(features.reshape(1, -1))
    features_bias = np.c_[np.ones(features.shape[0]), features]

    # Prediction
    y_pred_probs = []
    # Get the number of classes (number of elements in all_theta)
    num_classes = len(all_theta)
    for i in range(num_classes):  # Iterate using the number of classes
        # Make sure all_theta[i] is a 1D array and has the correct shape for matrix multiplication
        theta = all_theta[i].reshape(-1, 1) if all_theta[i].ndim == 1 else all_theta[i]
        z = features_bias @ theta
        y_pred_probs.append(sigmoid(z)[0][0])  # Get the probability value from sigmoid output

    # Get the class with the highest probability
    predicted_class = np.argmax(y_pred_probs)  # Use np.argmax to find the index of max probability
    label_data = pd.read_csv('data_codes.csv')
    row = label_data[label_data['encoded_part'] == predicted_class]
    return row['extracted_part'].iloc[0] if not row.empty else str(predicted_class)
