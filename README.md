# 🩺 ML Diabetes Risk Prediction System

An end-to-end Machine Learning web application that predicts whether a person has **Low Risk** or **High Risk** of diabetes based on clinical and lifestyle-related inputs.

This project follows a complete data science lifecycle: **EDA, preprocessing, feature engineering, model training, model comparison, evaluation, deployment, and containerization using Docker**.

---

## 🚀 Live Demo

🔗 **Deployed App:**  
`https://your-render-url.onrender.com`

> Replace the above URL with your actual Render deployment link.

---

## 📌 Project Overview

Diabetes is one of the most common chronic health conditions worldwide. Early risk prediction can help individuals take preventive action through lifestyle changes, medical consultation, and regular monitoring.

This project uses machine learning to predict diabetes risk using patient data such as:

- Gender
- Age
- Hypertension
- Heart disease history
- Smoking history
- BMI
- HbA1c level
- Blood glucose level

The application provides:

- Risk prediction: **Low Risk / High Risk**
- Confidence score
- Clean Flask-based web interface
- Dockerized deployment support
- Render cloud deployment support

---

## 🎯 Key Features

- ✅ End-to-end Machine Learning pipeline
- ✅ Custom preprocessing pipeline using Scikit-learn
- ✅ IQR-based outlier handling
- ✅ Feature engineering using custom transformer
- ✅ OneHotEncoding for categorical features
- ✅ StandardScaler for numerical features
- ✅ RandomForest and XGBoost model comparison
- ✅ ROC-AUC based model selection
- ✅ Threshold tuning for recall optimization
- ✅ Confidence score display
- ✅ Flask web application
- ✅ Render deployment
- ✅ Docker containerization
- ✅ Modular and production-style project structure

---

## 🧠 Machine Learning Workflow

```text
Raw Dataset
    ↓
Data Cleaning
    ↓
Exploratory Data Analysis
    ↓
Outlier Handling
    ↓
Feature Engineering
    ↓
Preprocessing Pipeline
    ↓
Model Training
    ↓
Model Comparison
    ↓
Evaluation
    ↓
Model Serialization
    ↓
Flask Deployment
    ↓
Docker Containerization
