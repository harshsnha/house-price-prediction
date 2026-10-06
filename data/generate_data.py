"""Builds a sample housing dataset (data/house_price.csv). Skip if you have your own."""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1500

city_factor = {"Mumbai": 1.9, "Delhi": 1.6, "Bengaluru": 1.5, "Pune": 1.2, "Lucknow": 0.9, "Patna": 0.75}
cities = rng.choice(list(city_factor), n, p=[0.18, 0.18, 0.18, 0.16, 0.15, 0.15])

df = pd.DataFrame({
    "city": cities,
    "area_sqft": rng.integers(500, 4000, n),
    "bedrooms": rng.integers(1, 6, n),
    "bathrooms": rng.integers(1, 5, n),
    "age_years": rng.integers(0, 40, n),
    "location_score": rng.integers(1, 11, n),
    "parking": rng.integers(0, 3, n),
})

base = (
    df["area_sqft"] * 2800
    + df["bedrooms"] * 200000
    + df["bathrooms"] * 120000
    + df["location_score"] * 300000
    + df["parking"] * 150000
    - df["age_years"] * 25000
)
df["price"] = (base * df["city"].map(city_factor) + rng.normal(0, 350000, n)).clip(lower=500000).round(-3)

# sprinkle some missing values so the cleaning step has something to do
for col in ["age_years", "bathrooms", "city"]:
    df.loc[rng.choice(n, 20, replace=False), col] = np.nan

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "house_price.csv")
df.to_csv(out, index=False)
print(f"saved {len(df)} rows -> {out}")
