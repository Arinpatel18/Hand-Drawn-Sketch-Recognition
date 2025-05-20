import numpy as np
import gdown
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib  # For saving the model
from tqdm import tqdm

class NeuralNetwork:
    def __init__(self, layers, activation='relu', learning_rate=0.01):
        self.layers = layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i+1], layers[i]) * np.sqrt(1. / layers[i]))
            self.biases.append(np.zeros((layers[i+1])))

    def sigmoid(self,Z):
      return 1 / (1 + np.exp(-np.array(Z)))

    def sigmoid_derivative(self,Z):
      return self.sigmoid(Z) * (1 - self.sigmoid(Z))

    def ReLU(self,Z):
      return np.maximum(0, np.array(Z))

    def ReLU_derivative(self,Z):
        return Z > 0

    def softmax(self,Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # stability fix
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def linear(self,Z):
      return Z

    def linear_derivative(self,Z):
      return np.ones(Z.shape)

    def mse(self,y_true, y_pred):
      return np.mean(np.power(y_true - y_pred, 2))

    def cross_entropy_loss(self,y_true, y_pred):
        m = y_true.shape[0]
        # Avoid log(0) by adding small value
        log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)] + 1e-9)
        return np.sum(log_likelihood) / m

    def accuracy(self,y_true, y_pred):
        return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

    def forward(self, x):
        activations = [np.array(x)]
        z_values = []

        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i].T) + self.biases[i]
            z_values.append(z)

            if i == len(self.weights) - 1:
                activations.append(self.softmax(z))  # Final layer softmax
            elif self.activation == 'sigmoid':
                activations.append(self.sigmoid(z))
            elif self.activation == 'relu':
                activations.append(self.ReLU(z))
            elif self.activation == 'linear':
                activations.append(self.linear(z))

        return activations, z_values

    def backward(self, x, y, activations, z_values):
        m = y.shape[0] if y.ndim > 1 else 1
        delta = activations[-1] - y  # Cross-entropy derivative

        for i in range(len(self.weights) - 1, -1, -1):
            if i < len(self.weights) - 1:
                if self.activation == 'sigmoid':
                    delta *= self.sigmoid_derivative(z_values[i])
                elif self.activation == 'relu':
                    delta *= self.ReLU_derivative(z_values[i])
                elif self.activation == 'linear':
                    delta *= self.linear_derivative(z_values[i])

            a_prev = activations[i] if activations[i].ndim > 1 else activations[i].reshape(1, -1)
            delta_mat = delta if delta.ndim > 1 else delta.reshape(1, -1)

            self.weights[i] -= self.learning_rate * (delta_mat.T @ a_prev) / m
            self.biases[i] -= self.learning_rate * delta_mat.sum(axis=0) / m

            delta = delta @ self.weights[i]

    def train(self, x_train, y_train, epochs=20, batch_size=64):
        for epoch in range(epochs):
            indices = np.arange(len(x_train))
            np.random.shuffle(indices)
            x_train, y_train = x_train[indices], y_train[indices]

            for start in range(0, len(x_train), batch_size):
                end = start + batch_size
                x_batch = x_train[start:end]
                y_batch = y_train[start:end]

                activations, z_values = self.forward(x_batch)
                self.backward(x_batch, y_batch, activations, z_values)

            # Evaluate
            train_pred = self.predict(x_train)
            train_loss = self.cross_entropy_loss(y_train, train_pred)
            train_acc = self.accuracy(y_train, train_pred)

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

    def predict(self, x_test):
        activations, _ = self.forward(x_test)
        return activations[-1]

    def save(self, filepath):
      model_data = {
          'layers': self.layers,
          'activation': self.activation,
          'learning_rate': self.learning_rate,
          'weights': self.weights,
          'biases': self.biases
      }
      joblib.dump(model_data, filepath)

    @staticmethod
    def load(filepath):
      model_data = joblib.load(filepath)
      model = NeuralNetwork(model_data['layers'], model_data['activation'], model_data['learning_rate'])
      model.weights = model_data['weights']
      model.biases = model_data['biases']
      return model

if __name__ == "__main__":
    file_id = "1QfzguI680h2Od7VXoPTypVr2h-DWwvkk"
    download_url = f"https://drive.google.com/uc?id={file_id}"

    # Download the file
    output = "data.csv"
    gdown.download(download_url, output, quiet=False)

    # Read the CSV file
    data = pd.read_csv(output)
    data = data.dropna()


    X = data.drop(data.columns[0],axis = 1).drop(columns = ['encoded_part','extracted_part'])
    y = data['encoded_part']

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y_onehot = np.eye(250)[y_encoded]  # One-hot encoding

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(X, y_onehot, test_size=0.1, random_state=42, stratify=y_encoded)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    Layers = np.array([X_train.shape[1],1024,512,250])
    ANN = NeuralNetwork(Layers,activation = 'relu',learning_rate=0.005)
    ANN.train(X_train,y_train, epochs=60, batch_size=64) # epochs=60, batch_size=64 (or any desired batch_size)

    y_pred = ANN.predict(X_val)

    ANN.save("ann_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(encoder, "label_encoder.pkl")

    y_pred_labels = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
    y_true_labels = np.argmax(y_val, axis=1)  # Convert one-hot to class labels
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    print(f"Validation Accuracy: {100*accuracy:.4f}%")