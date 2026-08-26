import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# Load data
df = pd.read_csv("data\\crop_recommendation.csv")

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

features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Feature distributions
df[features].hist(figsize=(14, 8), bins=25)
plt.tight_layout()
plt.show()


# Target distribution
plt.figure(figsize=(12, 6))
df["label"].value_counts().plot(kind="bar")
plt.title("Crop Distribution")
plt.xlabel("Crop")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Correlation
plt.figure(figsize=(9, 7))
sns.heatmap(
    df[features].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)
plt.title("Feature Correlation")
plt.tight_layout()
plt.show()


# Average NPK by crop
avg_npk = df.groupby("label")[["N", "P", "K"]].mean()

avg_npk.plot(kind="bar", figsize=(14, 6))
plt.title("Average NPK by Crop")
plt.ylabel("Average Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Average environmental conditions by crop
avg_environment = df.groupby("label")[
    ["temperature", "humidity", "ph", "rainfall"]
].mean()

avg_environment.plot(kind="bar", subplots=True, figsize=(14, 10))
plt.tight_layout()
plt.show()


# =========================
# Model Comparison
# =========================

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
}


results = []

for name, model in models.items():

    if name == "Random Forest":
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    results.append({
        "Model": name,
        "Train Accuracy": train_acc,
        "Test Accuracy": test_acc,
        "Difference": train_acc - test_acc
    })


results_df = pd.DataFrame(results).sort_values(
    "Test Accuracy",
    ascending=False
)

print("\nModel Comparison:")
print(results_df.to_string(index=False))


# Accuracy comparison
plt.figure(figsize=(9, 5))
sns.barplot(
    data=results_df,
    x="Test Accuracy",
    y="Model"
)
plt.title("Model Accuracy Comparison")
plt.xlim(0, 1)
plt.tight_layout()
plt.show()


# Confusion matrix for the best model
best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]

if best_model_name == "Random Forest":
    best_pred = best_model.predict(X_test)
else:
    best_pred = best_model.predict(X_test_scaled)

cm = confusion_matrix(y_test, best_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=False,
    cmap="Blues",
    xticklabels=sorted(y.unique()),
    yticklabels=sorted(y.unique())
)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# Random Forest feature importance
rf_model = models["Random Forest"]

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nRandom Forest Feature Importance:")
print(importance)

plt.figure(figsize=(9, 5))
sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()
