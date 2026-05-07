import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "diabetes_prediction_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")

DEBUG = False

PORT = int(os.environ.get("PORT", 8000))

SHAP_SAMPLE_SIZE = 500