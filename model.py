import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import config

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from custom_transformers import OutlierHandler, FeatureEngineer


# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(config.DATA_PATH)
df = df.drop_duplicates()
df = df[df['gender'] != 'Other']

X = df.drop("diabetes", axis=1)
y = df["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ==============================
# PREPROCESSOR
# ==============================
numeric_features = [
    'age', 'bmi', 'HbA1c_level',
    'blood_glucose_level', 'glucose_high_bp',
    'hypertension', 'heart_disease', 'is_senior'
]

categorical_features = ['gender', 'smoking_history']

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_features),

    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_features)
])


# ==============================
# MODEL COMPARISON
# ==============================
models = {
    "RandomForest": RandomForestClassifier(random_state=42, class_weight='balanced'),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

best_model = None
best_score = 0

for name, model in models.items():
    print(f"\nTraining {name}")

    pipe = Pipeline([
        ('outliers', OutlierHandler(['bmi', 'blood_glucose_level', 'HbA1c_level'])),
        ('feature_engineering', FeatureEngineer()),
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_proba)

    print(f"{name} ROC-AUC: {score:.4f}")

    if score > best_score:
        best_score = score
        best_model = pipe

print("\nBest Model Selected")


# ==============================
# EVALUATION
# ==============================
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)


# ==============================
# THRESHOLD
# ==============================
for t in np.arange(0.1, 0.9, 0.1):
    preds = (y_proba > t).astype(int)
    recall = classification_report(y_test, preds, output_dict=True)['1']['recall']
    print(f"Threshold {t:.1f} → Recall: {recall:.4f}")


# ==============================
# DASHBOARD
# ==============================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 0])
axes[0, 0].set_title("Confusion Matrix")

RocCurveDisplay.from_estimator(best_model, X_test, y_test, ax=axes[0, 1])
axes[0, 1].set_title("ROC Curve")

PrecisionRecallDisplay.from_estimator(best_model, X_test, y_test, ax=axes[1, 0])
axes[1, 0].set_title("Precision-Recall Curve")

try:
    importances = best_model.named_steps['model'].feature_importances_
    axes[1, 1].bar(range(len(importances)), importances)
    axes[1, 1].set_title("Feature Importance")
except:
    axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig("dashboard.png")
plt.show()


# ==============================
# SHAP (100% SAFE)
# ==============================
try:
    X_processed = best_model[:-1].transform(X_test)

    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    X_df = pd.DataFrame(X_processed, columns=feature_names)

    X_sample = X_df.sample(config.SHAP_SAMPLE_SIZE, random_state=42)

    explainer = shap.Explainer(best_model.named_steps['model'])
    shap_values = explainer(X_sample)

    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig("shap.png")
    plt.close()

    print("SHAP generated")

except Exception as e:
    print("SHAP skipped:", e)


# ==============================
# SAVE MODEL
# ==============================
with open(config.MODEL_PATH, "wb") as f:
    pickle.dump(best_model, f)

print("\nModel saved successfully!")