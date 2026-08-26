import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("data\\crop_recommendation.csv")

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# Train the model
model.fit(X_train, y_train)


# ==================================================
# OVERFITTING CHECK
# ==================================================

# Prediction on training data
train_pred = model.predict(X_train)

# Training accuracy
train_accuracy = accuracy_score(y_train, train_pred)


# Prediction on testing data
y_pred = model.predict(X_test)

# Testing accuracy
test_accuracy = accuracy_score(y_test, y_pred)


# Difference between training and testing accuracy
accuracy_difference = train_accuracy - test_accuracy


print("\n" + "=" * 60)
print("OVERFITTING CHECK")
print("=" * 60)

print(f"Training Accuracy : {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy  : {test_accuracy * 100:.2f}%")
print(f"Difference        : {accuracy_difference * 100:.2f}%")


# Simple interpretation
if accuracy_difference < 0.03:
    print(" No significant overfitting detected.")

elif accuracy_difference < 0.08:
    print(" Slight overfitting may exist.")

else:
    print(" Possible overfitting detected.")


# ==================================================
# MODEL EVALUATION
# ==================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==================================================
# FEATURE IMPORTANCE
# ==================================================

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)


# ==================================================
# SAVE MODEL
# ==================================================

import joblib

joblib.dump(model, "models\\crop_classifier.pkl")
print("\nModel saved as: crop_classifier.pkl")