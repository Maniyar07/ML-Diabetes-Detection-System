from flask import Flask, request, render_template
import pandas as pd
import pickle
import os
import config

# 🔥 IMPORTANT FIX
from custom_transformers import OutlierHandler, FeatureEngineer

app = Flask(__name__)

# LOAD MODEL
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "best_model.pkl")

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("✅ Model Loaded")
except Exception as e:
    print("❌ Model load error:", e)
    model = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return "Model not loaded"

    try:
        data = pd.DataFrame([{
            'gender': request.form['gender'],
            'age': float(request.form['age']),
            'hypertension': int(request.form['hypertension']),
            'heart_disease': int(request.form['heart_disease']),
            'smoking_history': request.form['smoking_history'],
            'bmi': float(request.form['bmi']),
            'HbA1c_level': float(request.form['HbA1c_level']),
            'blood_glucose_level': float(request.form['blood_glucose_level'])
        }])

        pred = model.predict(data)[0]
        prob = model.predict_proba(data)[0]

        if pred == 1:
            result = "High Risk"
            confidence = prob[1]
            result_class = "danger"
        else:
            result = "Low Risk"
            confidence = prob[0]
            result_class = "success"

        return render_template(
            "result.html",
            prediction=result,
            probability=f"{confidence * 100:.4f}%",
            result_class=result_class
        )

    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG)