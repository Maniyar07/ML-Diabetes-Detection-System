from flask import Flask, request, render_template
import pandas as pd
import pickle
import config

# IMPORTANT: required for loading pipeline custom transformers
from custom_transformers import OutlierHandler, FeatureEngineer

app = Flask(__name__)

# LOAD MODEL

try:
    print("Loading model from:", config.MODEL_PATH)

    with open(config.MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print("✅ Model Loaded Successfully")

except Exception as e:
    print("❌ Model load error:", e)
    model = None

# HOME ROUTE

@app.route("/")
def home():
    return render_template("index.html")

# PREDICT ROUTE

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return "❌ Model not loaded. Check Render logs."

    try:
        data = pd.DataFrame([{
            "gender": request.form["gender"],
            "age": float(request.form["age"]),
            "hypertension": int(request.form["hypertension"]),
            "heart_disease": int(request.form["heart_disease"]),
            "smoking_history": request.form["smoking_history"],
            "bmi": float(request.form["bmi"]),
            "HbA1c_level": float(request.form["HbA1c_level"]),
            "blood_glucose_level": float(request.form["blood_glucose_level"])
        }])

        # Input validation
        age = data.loc[0, "age"]
        bmi = data.loc[0, "bmi"]
        hba1c = data.loc[0, "HbA1c_level"]
        glucose = data.loc[0, "blood_glucose_level"]

        if age < 0 or age > 120:
            return "❌ Invalid age value"

        if bmi <= 0 or bmi > 100:
            return "❌ Invalid BMI value"

        if hba1c <= 0:
            return "❌ Invalid HbA1c value"

        if glucose <= 0:
            return "❌ Invalid blood glucose value"

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
        return f"❌ Prediction error: {str(e)}"

# RUN APP

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG)