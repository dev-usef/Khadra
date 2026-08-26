import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data\\crop_recommendation.csv")

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("\n" + "-" * 60)
print(" MODEL: Random Forest ")
print("-" * 60)

rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)

rf_train_acc = accuracy_score(y_train, rf_train_pred)
rf_test_acc = accuracy_score(y_test, rf_test_pred)
rf_diff = rf_train_acc - rf_test_acc

print("\nOVERFITTING CHECK")
print(f"Training Accuracy : {rf_train_acc * 100:.2f}%")
print(f"Testing Accuracy  : {rf_test_acc * 100:.2f}%")
print(f"Difference        : {rf_diff * 100:.2f}%")

if rf_diff < 0.03:
    print(" No significant overfitting detected.")
elif rf_diff < 0.08:
    print(" Slight overfitting may exist.")
else:
    print(" Possible overfitting detected.")

print("\nMODEL EVALUATION")
print(classification_report(y_test, rf_test_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_test_pred))

joblib.dump(rf_model, "models\\random_forest_classifier.pkl")

print("\n" + "-" * 60)
print(" MODEL: K-Nearest Neighbors (KNN) ")
print("-" * 60)

knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_model.fit(X_train_scaled, y_train)

knn_train_pred = knn_model.predict(X_train_scaled)
knn_test_pred = knn_model.predict(X_test_scaled)

knn_train_acc = accuracy_score(y_train, knn_train_pred)
knn_test_acc = accuracy_score(y_test, knn_test_pred)
knn_diff = knn_train_acc - knn_test_acc

print("\nOVERFITTING CHECK")
print(f"Training Accuracy : {knn_train_acc * 100:.2f}%")
print(f"Testing Accuracy  : {knn_test_acc * 100:.2f}%")
print(f"Difference        : {knn_diff * 100:.2f}%")

if knn_diff < 0.03:
    print(" No significant overfitting detected.")
elif knn_diff < 0.08:
    print(" Slight overfitting may exist.")
else:
    print(" Possible overfitting detected.")

print("\nMODEL EVALUATION")
print(classification_report(y_test, knn_test_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, knn_test_pred))

joblib.dump(knn_model, "models\\knn_classifier.pkl")

print("\n" + "-" * 60)
print(" MODEL: Logistic Regression ")
print("-" * 60)

lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr_model.fit(X_train_scaled, y_train)

lr_train_pred = lr_model.predict(X_train_scaled)
lr_test_pred = lr_model.predict(X_test_scaled)

lr_train_acc = accuracy_score(y_train, lr_train_pred)
lr_test_acc = accuracy_score(y_test, lr_test_pred)
lr_diff = lr_train_acc - lr_test_acc

print("\nOVERFITTING CHECK")
print(f"Training Accuracy : {lr_train_acc * 100:.2f}%")
print(f"Testing Accuracy  : {lr_test_acc * 100:.2f}%")
print(f"Difference        : {lr_diff * 100:.2f}%")

if lr_diff < 0.03:
    print(" No significant overfitting detected.")
elif lr_diff < 0.08:
    print(" Slight overfitting may exist.")
else:
    print(" Possible overfitting detected.")

print("\nMODEL EVALUATION")
print(classification_report(y_test, lr_test_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, lr_test_pred))

joblib.dump(lr_model, "models\\logistic_regression_classifier.pkl")

joblib.dump(scaler, "models\\scaler.pkl")

comparison_data = {
    "Model": ["Random Forest", "KNN", "Logistic Regression"],
    "Train Accuracy": [
        f"{rf_train_acc * 100:.2f}%",
        f"{knn_train_acc * 100:.2f}%",
        f"{lr_train_acc * 100:.2f}%",
    ],
    "Test Accuracy": [
        f"{rf_test_acc * 100:.2f}%",
        f"{knn_test_acc * 100:.2f}%",
        f"{lr_test_acc * 100:.2f}%",
    ],
    "Difference": [
        f"{rf_diff * 100:.2f}%",
        f"{knn_diff * 100:.2f}%",
        f"{lr_diff * 100:.2f}%",
    ],
}

print("\n" + "=" * 60)
print("FINAL MODELS ACCURACY COMPARISON")
print("=" * 60)
df_compare = pd.DataFrame(comparison_data)
print(df_compare.to_string(index=False))
