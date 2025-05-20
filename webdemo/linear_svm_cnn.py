import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import warnings
from sklearn.exceptions import ConvergenceWarning
import joblib

class OneVsRestSVM:
    def __init__(self, C, max_iter):
        self.C = C
        self.max_iter = max_iter
        self.classifiers = {}

    def fit(self, X, y):
        self.unique_classes = np.unique(y)
        for cls in tqdm(self.unique_classes, desc="Training"):
            binary_y = np.where(y == cls, 1, -1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                clf = LinearSVC(C=self.C, max_iter=self.max_iter)
                clf.fit(X, binary_y)
            self.classifiers[cls] = clf

    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = []
        for row in X:
            scores = {cls: clf.decision_function([row])[0] for cls, clf in self.classifiers.items()}
            predictions.append(max(scores, key=scores.get))
        return np.array(predictions)

    def save_model(self, filename):
        joblib.dump({
            'C': self.C,
            'max_iter': self.max_iter,
            'unique_classes': self.unique_classes,
            'classifiers': self.classifiers
        }, filename)

    @classmethod
    def load_model(cls, filename):
        data = joblib.load(filename)
        model = cls(C=data['C'], max_iter=data['max_iter'])
        model.unique_classes = data['unique_classes']
        model.classifiers = data['classifiers']
        return model

if __name__ == "__main__":
    data = pd.read_csv('/content/drive/MyDrive/prml/cnn_features_train.csv')
    X = data.drop(data.columns[0], axis=1).drop(["extracted_part", "encoded_part"], axis=1)
    y = data["encoded_part"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model = OneVsRestSVM(C=1, max_iter=1000)
    model.fit(X_train.values, y_train.values)
    model.save_model('onevsrest_svm_model.pkl')

    y_pred = model.predict(X_test.values)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")