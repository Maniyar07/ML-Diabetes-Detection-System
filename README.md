# ML-Diabetes-Detection-System
End-to-end machine learning Flask web app for diabetes risk prediction using preprocessing pipelines, feature engineering, model comparison, threshold tuning, and XGBoost.
# ML Diabetes Detection System

A machine learning web application that predicts whether a person has a low or high risk of diabetes based on medical and lifestyle-related input features.

The project uses a complete ML pipeline including preprocessing, feature engineering, outlier handling, model comparison, threshold tuning, and Flask-based web deployment.

---

## Project Overview

This project predicts diabetes risk using patient information such as:

- Gender
- Age
- Hypertension
- Heart disease
- Smoking history
- BMI
- HbA1c level
- Blood glucose level

The system returns:

- Diabetes risk prediction
- Confidence score
- Clean web-based result page

---

## Tech Stack

### Machine Learning
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Seaborn

### Web Development
- Flask
- HTML
- CSS

### Deployment
- Render
- Gunicorn

---

## Features

- End-to-end machine learning pipeline
- Custom feature engineering
- IQR-based outlier handling
- StandardScaler for numerical features
- OneHotEncoder for categorical features
- RandomForest and XGBoost model comparison
- ROC-AUC based model selection
- Threshold tuning for better recall
- Confidence score display
- Flask web interface
- Modular project structure
- Production-ready deployment setup

---

## Machine Learning Pipeline

The pipeline follows this flow:

```text
Raw Input Data
        ↓
Outlier Handling
        ↓
Feature Engineering
        ↓
Missing Value Handling
        ↓
Scaling + Encoding
        ↓
ML Model Prediction
