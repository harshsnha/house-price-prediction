import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(BASE, "report_images")
os.makedirs(IMG, exist_ok=True)

df = pd.read_csv(os.path.join(BASE, "data", "house_price.csv"))
print("shape:", df.shape)
print(df.isnull().sum())
print(df.describe().round(2))

plt.figure(figsize=(7, 4))
df["price"].plot(kind="hist", bins=40, color="#4c72b0", edgecolor="white")
plt.title("Price distribution")
plt.xlabel("Price (INR)")
plt.tight_layout()
plt.savefig(os.path.join(IMG, "price_distribution.png"), dpi=130)
plt.close()

city_avg = df.groupby("city")["price"].mean().sort_values()
plt.figure(figsize=(7, 4))
(city_avg / 1e6).plot(kind="barh", color="#55a868")
plt.xlabel("Average price (million INR)")
plt.title("Average price by city")
plt.tight_layout()
plt.savefig(os.path.join(IMG, "price_by_city.png"), dpi=130)
plt.close()

corr = df.select_dtypes("number").corr()
plt.figure(figsize=(7, 6))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.title("Correlation heatmap")
plt.tight_layout()
plt.savefig(os.path.join(IMG, "correlation_heatmap.png"), dpi=130)
plt.close()

print("charts saved in report_images/")
