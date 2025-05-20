*Hand-Drawn Sketch Recognition

Overview:

Hand-drawn sketch recognition bridges the gap between human creativity and machine interpretation by enabling systems to understand and classify rough, freehand drawings. This project focuses on developing a deep learning-based approach to recognize and categorize simple hand-drawn sketches into predefined classes.

Abstract:

Utilizing the Quick, Draw! dataset from Google, a Convolutional Neural Network (CNN) model was trained and evaluated for its accuracy and efficiency. The model achieved promising performance, demonstrating the capability of AI in interpreting abstract and minimalistic visual inputs. This project has practical applications in human-computer interaction, education, and assistive technologies.

Table of Contents:

1. Introduction

2. Dataset

3. Feature Extraction

4. Approaches Tried

5. Experiments and Results

6. Contributions

7. Installation

8. Usage

Introduction:

Hand-drawn sketches are significantly different from natural images, requiring robust feature extraction techniques for accurate classification. Our goal is to classify sketches into one of several object categories using various Machine Learning models, compare the accuracies, and choose the best one among them.

Dataset:

The dataset consists of images of 250 different objects, each with 80-90 image samples, totaling 20,180 images. Each image is sized at 1111x1111 pixels. The features are extracted through pixels, and the labels are defined from the parent folder of the image.

Feature Extraction:

We employed several methods for feature extraction:

a. Histogram Oriented Gradient (HOG): Captures the distribution of gradient orientations in small image regions.

b. CNN ResNet-based Feature Extraction: Utilizes ResNet filters in CNN layers to extract low-dimensional feature vectors.

c. Principal Component Analysis (PCA): A dimensionality reduction technique to optimize feature vector size for model performance.

*Approaches Tried:

We explored various machine learning models, including:

1. Decision Trees

2. Logistic Regression

3. K-Nearest Neighbors (KNN)

4. Support Vector Machines (SVM)

5. Bayesian Classification

6. Artificial Neural Networks (ANN)

Experiments and Results:

The project includes detailed experiments and results for each model, showcasing their performance metrics and accuracy rates.

Contributions:

Vivekananda Mudelli : Implemented KNN and ANN models, frontend development, CNN feature extraction, and final report.

Yashwanth : Implemented Multivariate Gaussian bias and Linear SVM models, backend development, video creation, and report writing.

Jagadeesh : Implemented Multivariate Gaussian bias and Logistic Regression models, backend development, and project webpage development.

Arin : Implemented Decision Tree and K-Means Clustering, contributed to the final report.

Akshay : Implemented KNN and ANN models, created PowerPoint presentation, and contributed to the final report and video creation.

Ganesh : Implemented Decision Tree and K-Means Clustering, contributed to the final report.

Installation:

To run this project, clone the repository and install the required dependencies:

bash

git clone https://github.com/VivekanandaMudelli/hand_drawn_sketch_recognition.git

cd hand_drawn_sketch_recognition

pip install -r requirements.txt

Usage:
To train the model, run the following command:

bash

python app.py
