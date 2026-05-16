SmartHeart: Intelligent Heart Disease Risk Prediction System 

--> SmartHeart is a machine learning-based decision support tool designed to approximate the likelihood of heart disease by utilizing clinical features. This project implements a comprehensive pipeline, from data ingestion and preprocessing to multi-model evaluation, to provide reliable, data-driven diagnostic insights.


Key Performance Highlights:-

--> The Random Forest classifier was selected as the optimal model for the system. Achieved a peak predictive accuracy of 97%. Maintained high reliability with 0.96 precision and 0.98 recall. Demonstrated near-perfect classification ability with an AUC score of 0.9967.


System Architecture:-

Data Ingestion: Loads and validates the cardiovascular dataset extracted from Kaggle.

Preprocessing: Includes median/mode imputation for missing data, one-hot encoding for categorical variables, and feature scaling.

Model Training: Evaluates four supervised techniques: Logistic Regression, Decision Tree, Random Forest, and Gaussian Naïve Bayes.

Evaluation: Metrics include Confusion Matrix interpretation and ROC-AUC analysis.


Results:-

Model: Random Forest

Accuracy = 0.9700, Precision = 0.9631, Recall = 0.9795

Model: Decision Tree

Accuracy = 0.8801, Precision = 0.8713, Recall = 0.9010

Model: Logistic Regression

Accuracy = 0.7125, Precision = 0.6935, Recall = 0.7952

Model: Naive Bayes

Accuracy = 0.6914, Precision = 0.6735, Recall = 0.7816


Ethical & Social Impact:-

Checked data distribution to prevent demographic bias in predictions. Ensured clinical data was anonymized for confidentiality. Supports SDG 3 (Good Health and Well-being) by encouraging preventive medicine through early detection.


Tech Stack:-

Python was the primary language used in the building & training of this model. The librairies that were utilized were Scikit-learn (Model Training), Pandas (Data Management), Matplotlib (Visualization). The model ran on a Command Line Interface (CLI).
