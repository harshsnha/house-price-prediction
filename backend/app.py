import json
import os
import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND = os.path.join(BASE, "frontend")

app = Flask(__name__, static_folder=FRONTEND)
bundle = joblib.load(os.path.join(BASE, "model", "house_price_model.pkl"))
model = bundle["model"]
num_cols, cat_cols, cities = bundle["num_cols"], bundle["cat_cols"], bundle["cities"]

with open(os.path.join(BASE, "model", "metrics.json")) as f:
    metrics = json.load(f)

LIMITS = {
    "area_sqft": (100, 20000),
    "bedrooms": (1, 10),
    "bathrooms": (1, 10),
    "age_years": (0, 100),
    "location_score": (1, 10),
    "parking": (0, 10),
}


@app.route("/")
def home():
    return send_from_directory(FRONTEND, "index.html")


@app.route("/meta")
def meta():
    best = next(r for r in metrics["results"] if r["model"] == metrics["best_model"])
    return jsonify({"cities": cities, "model": metrics["best_model"], "r2": best["r2"]})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    row = {}
    for col in num_cols:
        if col not in data:
            return jsonify({"error": f"{col} is required"}), 400
        try:
            val = float(data[col])
        except (TypeError, ValueError):
            return jsonify({"error": f"{col} must be a number"}), 400
        lo, hi = LIMITS[col]
        if not lo <= val <= hi:
            return jsonify({"error": f"{col} should be between {lo} and {hi}"}), 400
        row[col] = val

    city = data.get("city")
    if city not in cities:
        return jsonify({"error": "pick a valid city"}), 400
    row["city"] = city

    price = float(model.predict(pd.DataFrame([row]))[0])
    return jsonify({"predicted_price": int(round(max(price, 0), -3))})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
