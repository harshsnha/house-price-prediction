import json
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(BASE, "report_images")
os.makedirs(IMG, exist_ok=True)
os.makedirs(os.path.join(BASE, "model"), exist_ok=True)

TARGET = "price"
NUM_COLS = ["area_sqft", "bedrooms", "bathrooms", "age_years", "location_score", "parking"]
CAT_COLS = ["city"]

df = pd.read_csv(os.path.join(BASE, "data", "house_price.csv")).drop_duplicates()
df[NUM_COLS] = df[NUM_COLS].fillna(df[NUM_COLS].median())
df[CAT_COLS] = df[CAT_COLS].fillna(df[CAT_COLS].mode().iloc[0])

X, y = df[NUM_COLS + CAT_COLS], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def build(reg):
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    return Pipeline([("prep", prep), ("reg", reg)])


candidates = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

rows, fitted = [], {}
for name, reg in candidates.items():
    pipe = build(reg).fit(X_train, y_train)
    pred = pipe.predict(X_test)
    cv = cross_val_score(build(reg), X_train, y_train, cv=5, scoring="r2")
    rows.append({
        "model": name,
        "r2": round(r2_score(y_test, pred), 4),
        "cv_r2": round(cv.mean(), 4),
        "mae": int(mean_absolute_error(y_test, pred)),
        "rmse": int(mean_squared_error(y_test, pred) ** 0.5),
    })
    fitted[name] = pipe

results = pd.DataFrame(rows).sort_values("cv_r2", ascending=False).reset_index(drop=True)
print(results.to_string(index=False))

best_name = results.loc[0, "model"]
best = fitted[best_name]
print("\nbest model:", best_name)

# charts
plt.figure(figsize=(7, 4))
plt.barh(results["model"][::-1], results["r2"][::-1], color="#4c72b0")
plt.xlim(max(0, results["r2"].min() - 0.05), 1)
plt.xlabel("R2 score (test set)")
plt.title("Model comparison")
plt.tight_layout()
plt.savefig(os.path.join(IMG, "model_comparison.png"), dpi=130)
plt.close()

pred = best.predict(X_test)
plt.figure(figsize=(5.5, 5.5))
plt.scatter(y_test, pred, alpha=0.4, color="#4c72b0")
lims = [y.min(), y.max()]
plt.plot(lims, lims, "r--")
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title(f"Actual vs predicted ({best_name})")
plt.tight_layout()
plt.savefig(os.path.join(IMG, "actual_vs_predicted.png"), dpi=130)
plt.close()

plt.figure(figsize=(7, 4))
plt.hist(y_test - pred, bins=40, color="#c44e52", edgecolor="white")
plt.axvline(0, color="black", linewidth=1)
plt.xlabel("Residual (actual - predicted)")
plt.title("Residual distribution")
plt.tight_layout()
plt.savefig(os.path.join(IMG, "residuals.png"), dpi=130)
plt.close()

imp = permutation_importance(best, X_test, y_test, n_repeats=5, random_state=42)
imp_s = pd.Series(imp.importances_mean, index=X_test.columns).sort_values()
plt.figure(figsize=(7, 4))
imp_s.plot(kind="barh", color="#55a868")
plt.xlabel("Importance (drop in R2 when shuffled)")
plt.title("Feature importance")
plt.tight_layout()
plt.savefig(os.path.join(IMG, "feature_importance.png"), dpi=130)
plt.close()

joblib.dump({
    "model": best,
    "name": best_name,
    "num_cols": NUM_COLS,
    "cat_cols": CAT_COLS,
    "cities": sorted(df["city"].unique().tolist()),
}, os.path.join(BASE, "model", "house_price_model.pkl"))

with open(os.path.join(BASE, "model", "metrics.json"), "w") as f:
    json.dump({
        "best_model": best_name,
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "results": results.to_dict(orient="records"),
        "importance": {k: round(float(v), 4) for k, v in imp_s.sort_values(ascending=False).items()},
    }, f, indent=2)
