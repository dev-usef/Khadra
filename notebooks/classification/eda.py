# eda.py
# Exploratory Data Analysis (EDA) only

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data\\crop_recommendation.csv")

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS - CROP RECOMMENDATION")
print("=" * 60)

# 1. First rows
print("\n1) First 5 Rows:")
print(df.head())

# 2. Dataset shape
print("\n2) Dataset Shape:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 3. Column names
print("\n3) Columns:")
print(df.columns.tolist())

# 4. Data types
print("\n4) Data Types:")
print(df.dtypes)

# 5. Missing values
print("\n5) Missing Values:")
print(df.isnull().sum())

# 6. Duplicate rows
print("\n6) Duplicate Rows:")
print(df.duplicated().sum())

# 7. Statistical summary
print("\n7) Statistical Summary:")
print(df.describe())

# 8. Number of crop classes
print("\n8) Number of Crop Classes:")
print(df["label"].nunique())

# 9. Crop distribution
print("\n9) Crop Distribution:")
print(df["label"].value_counts())

# 10. Correlation matrix
print("\n10) Correlation Matrix:")
print(df.drop("label", axis=1).corr().round(2))

# 11. Crop distribution
plt.figure(figsize=(12, 6))
df["label"].value_counts().plot(kind="bar")
plt.title("Crop Distribution")
plt.xlabel("Crop")
plt.ylabel("Number of Samples")
plt.xticks(rotation=75)
plt.tight_layout()
plt.show()

# 12. Histograms for numerical features
df.drop("label", axis=1).hist(figsize=(12, 8), bins=20)
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.show()

# 13. Boxplots
plt.figure(figsize=(12, 6))
df.drop("label", axis=1).boxplot()
plt.title("Feature Boxplots")
plt.ylabel("Value")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# 14. Correlation heatmap without seaborn
corr = df.drop("label", axis=1).corr()

plt.figure(figsize=(9, 7))
plt.imshow(corr, interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# 15. Average feature values by crop
grouped = df.groupby("label").mean(numeric_only=True)

print("\n11) Average Feature Values by Crop:")
print(grouped.round(2))

print("\nEDA completed successfully.")
