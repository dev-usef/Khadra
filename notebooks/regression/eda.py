import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/crop_yield.csv")

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS - CROP YIELD")
print("=" * 60)

# 1. First 5 rows
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

# 8. Number of crop types
print("\n8) Number of Crop Types:")
print(df["Crop"].nunique())

# 9. Crop distribution
print("\n9) Crop Distribution:")
print(df["Crop"].value_counts())

# 10. Correlation matrix
numeric_df = df.select_dtypes(include="number")

print("\n10) Correlation Matrix:")
print(numeric_df.corr().round(2))

# 11. Yield distribution
plt.figure(figsize=(10, 6))
df["Yield_tons_per_hectare"].hist(bins=30)
plt.title("Yield Distribution")
plt.xlabel("Yield (tons/hectare)")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.show()

# 12. Histograms for numerical features
numeric_df.hist(figsize=(12, 8), bins=20)
plt.suptitle("Numerical Feature Distributions")
plt.tight_layout()
plt.show()

# 13. Boxplots
plt.figure(figsize=(14, 7))
numeric_df.boxplot()
plt.title("Numerical Feature Boxplots")
plt.ylabel("Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 14. Correlation matrix visualization
corr = numeric_df.corr()

plt.figure(figsize=(10, 8))
plt.imshow(corr, interpolation="nearest")
plt.colorbar()

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=60
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns
)

plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# 15. Numerical Features vs Yield
features = [
    col for col in numeric_df.columns
    if col != "Yield_tons_per_hectare"
]

for feature in features:

    plt.figure(figsize=(7, 5))

    plt.scatter(
        df[feature],
        df["Yield_tons_per_hectare"],
        alpha=0.5
    )

    plt.title(f"{feature} vs Yield")
    plt.xlabel(feature)
    plt.ylabel("Yield (tons/hectare)")

    plt.tight_layout()
    plt.show()

# 16. Average yield by crop
average_yield = (
    df.groupby("Crop")["Yield_tons_per_hectare"]
    .mean()
    .sort_values(ascending=False)
)

print("\n11) Average Yield by Crop:")
print(average_yield.round(2))

plt.figure(figsize=(12, 6))

average_yield.plot(kind="bar")

plt.title("Average Yield by Crop")
plt.xlabel("Crop")
plt.ylabel("Average Yield (tons/hectare)")
plt.xticks(rotation=75)

plt.tight_layout()
plt.show()

print("\nEDA completed successfully.")