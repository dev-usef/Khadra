
import joblib
import pandas as pd

model = joblib.load("crop_classifier.pkl")

print("=" * 50)
print("CROP PREDICTION")
print("=" * 50)

N = float(input("Nitrogen (N): "))
P = float(input("Phosphorus (P): "))
K = float(input("Potassium (K): "))
temperature = float(input("Temperature (°C): "))
humidity = float(input("Humidity (%): "))
ph = float(input("Soil pH: "))
rainfall = float(input("Rainfall (mm): "))

new_data = pd.DataFrame([{
    "N": N,
    "P": P,
    "K": K,
    "temperature": temperature,
    "humidity": humidity,
    "ph": ph,
    "rainfall": rainfall
}])

prediction = model.predict(new_data)[0]

print("\n🌱 Recommended Crop:", prediction)

probabilities = model.predict_proba(new_data)[0]
classes = model.classes_

results = pd.DataFrame({
    "Crop": classes,
    "Probability": probabilities
}).sort_values("Probability", ascending=False)

print("\nTop 5 Predictions:")
print(results.head(5).to_string(index=False))
