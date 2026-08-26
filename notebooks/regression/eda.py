import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load data
df = pd.read_csv("data\\crop_yield.csv")

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicates:", df.duplicated().sum())

print("\nStatistics:")
print(df.describe())


# =========================
# EDA
# =========================

numerical_features = [
    "Area_hectares",
    "Temperature_C",
    "Rainfall_mm",
    "N_kg_ha",
    "P_kg_ha",
    "K_kg_ha",
    "Fertilizer_kg_ha",
    "Irrigation_percent",
    "Yield_tons_per_hectare"
]

# Yield distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Yield_tons_per_hectare"], kde=True)
plt.title("Yield Distribution")
plt.show()


# Numerical distributions
df[numerical_features].hist(figsize=(14, 10), bins=30)
plt.tight_layout()
plt.show()


# Correlation
plt.figure(figsize=(10, 8))
sns.heatmap(
    df[numerical_features].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)
plt.title("Feature Correlation")
plt.tight_layout()
plt.show()


# Average NPK by crop
avg_npk = df.groupby("Crop")[["N_kg_ha", "P_kg_ha", "K_kg_ha"]].mean()

avg_npk.plot(kind="bar", figsize=(14, 6))
plt.title("Average NPK by Crop")
plt.ylabel("Average Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Average yield by crop
avg_yield = df.groupby("Crop")["Yield_tons_per_hectare"].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
avg_yield.plot(kind="bar")
plt.title("Average Yield by Crop")
plt.ylabel("Yield (tons/hectare)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Top 10 crops boxplot
top_crops = df["Crop"].value_counts().head(10).index
df_top = df[df["Crop"].isin(top_crops)]

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_top, x="Crop", y="Yield_tons_per_hectare")
plt.title("Yield Distribution - Top 10 Crops")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# =========================
# Model Comparison
# =========================

def remove_outliers_iqr(data, columns):
    data = data.copy()

    for col in columns:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        data = data[
            (data[col] >= lower) &
            (data[col] <= upper)
        ]

    return data


model_df = remove_outliers_iqr(
    df,
    df.select_dtypes(include=np.number).columns
)

numerical_features_model = [
    "Area_hectares",
    "Temperature_C",
    "Rainfall_mm",
    "N_kg_ha",
    "P_kg_ha",
    "K_kg_ha",
    "Fertilizer_kg_ha",
    "Irrigation_percent"
]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features_model),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["Crop"])
])

X = model_df.drop("Yield_tons_per_hectare", axis=1)
y = model_df["Yield_tons_per_hectare"]

X = preprocessor.fit_transform(X)

feature_names = (
    numerical_features_model +
    list(preprocessor.named_transformers_["cat"].get_feature_names_out(["Crop"]))
)

X = pd.DataFrame(X, columns=feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


models = {
    "SVM": SVR(kernel="poly", degree=3, C=1.0),
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "KNN": KNeighborsRegressor(),
    "Ensemble": VotingRegressor([
        ("lr", LinearRegression()),
        ("knn", KNeighborsRegressor())
    ])
}


results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, prediction),
        "MSE": mean_squared_error(y_test, prediction),
        "R2": r2_score(y_test, prediction)
    })


results_df = pd.DataFrame(results).sort_values(
    "R2",
    ascending=False
)

print("\nModel Comparison:")
print(results_df.to_string(index=False))


# R2 comparison
plt.figure(figsize=(9, 5))
sns.barplot(data=results_df, x="R2", y="Model")
plt.title("Model Comparison - R2")
plt.tight_layout()
plt.show()


# MAE and MSE comparison
errors = results_df.melt(
    id_vars="Model",
    value_vars=["MAE", "MSE"],
    var_name="Metric",
    value_name="Value"
)

plt.figure(figsize=(10, 5))
sns.barplot(data=errors, x="Model", y="Value", hue="Metric")
plt.title("Model Errors")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# Decision Tree feature importance
dt_model = models["Decision Tree"]

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": dt_model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance, x="Importance", y="Feature")
plt.title("Decision Tree Feature Importance")
plt.tight_layout()
plt.show()
