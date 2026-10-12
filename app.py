import os
from flask import Flask, render_template, request, jsonify
import pickle
import json
import numpy as np

app = Flask(__name__)

# ── Load model artifacts ───────────────────────────────────────────────────────
with open("model/rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

with open("model/stats.json", "r") as f:
    stats = json.load(f)


@app.route("/")
def index():
    return render_template("index.html", stats=stats)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        gender = data.get("gender", "").strip()
        age = float(data.get("age", 0))
        salary = float(data.get("salary", 0))

        # Validate
        if gender not in ("Male", "Female"):
            return jsonify({"error": "Gender must be Male or Female"}), 400
        if not (18 <= age <= 100):
            return jsonify({"error": "Age must be between 18 and 100"}), 400
        if salary < 0:
            return jsonify({"error": "Salary must be a positive number"}), 400

        # Encode gender
        gender_enc = 1 if gender == "Male" else 0

        # Scale & predict
        features = np.array([[gender_enc, age, salary]])
        features_sc = scaler.transform(features)
        prediction = int(model.predict(features_sc)[0])
        proba = model.predict_proba(features_sc)[0]
        confidence = round(float(max(proba)) * 100, 1)
        buy_probability = round(float(proba[1]) * 100, 1)

        result = {
            "prediction": prediction,
            "label": "Should Buy" if prediction == 1 else "Should Not Buy",
            "confidence": confidence,
            "buy_probability": buy_probability,
            "no_buy_probability": round(float(proba[0]) * 100, 1),
        }
        return jsonify(result)

    except (ValueError, KeyError) as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400


if __name__ == "__main__":
    app.run(debug=False, port=5000, use_reloader=False)
