import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("car_data.csv")
df.columns = df.columns.str.strip().str.lstrip('\ufeff')

print("Dataset shape:", df.shape)
print(df.head())
print("\nPurchased value counts:\n", df["Purchased"].value_counts())

# ── Feature engineering ────────────────────────────────────────────────────────
le = LabelEncoder()
df["Gender_enc"] = le.fit_transform(df["Gender"])   # Female=0, Male=1

X = df[["Gender_enc", "Age", "AnnualSalary"]].values
y = df["Purchased"].values

# ── Train / test split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Scale features ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Train model ────────────────────────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=4,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train_sc, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_sc)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importances
feature_names = ["Gender", "Age", "Annual Salary"]
importances = model.feature_importances_
print("\nFeature Importances:")
for name, imp in zip(feature_names, importances):
    print(f"  {name}: {imp:.4f}")

# ── Save artifacts ─────────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)

with open("model/rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Save accuracy for display in app
import json
stats = {
    "accuracy": round(acc * 100, 2),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "feature_importances": {
        name: round(float(imp), 4)
        for name, imp in zip(feature_names, importances)
    }
}
with open("model/stats.json", "w") as f:
    json.dump(stats, f)

print("\nModel, scaler, and stats saved to ./model/")
