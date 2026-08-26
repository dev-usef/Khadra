# predictions.py
# Crop Yield Prediction

import joblib
import pandas as pd

# The trained Pipeline contains both preprocessing and regression model.
MODEL_PATH = "yield_regressor.pkl"

model = joblib.load(MODEL_PATH)

print("=" * 60)
print("KHADRA - CROP YIELD PREDICTION")
print("=" * 60)

# User inputs
crop = input("Crop: ").strip()

area = float(input("Area (hectares): "))
temperature = float(input("Temperature (°C): "))
rainfall = float(input("Rainfall (mm): "))

n = float(input("Nitrogen (N) kg/ha: "))
p = float(input("Phosphorus (P) kg/ha: "))
k = float(input("Potassium (K) kg/ha: "))

fertilizer = float(input("Fertilizer (kg/ha): "))
irrigation = float(input("Irrigation (%): "))

# Create one-row DataFrame with the exact training feature names
new_data = pd.DataFrame([{
    "Crop": crop,
    "Area_hectares": area,
    "Temperature_C": temperature,
    "Rainfall_mm": rainfall,
    "N_kg_ha": n,
    "P_kg_ha": p,
    "K_kg_ha": k,
    "Fertilizer_kg_ha": fertilizer,
    "Irrigation_percent": irrigation
}])

# Predict
# No manual encoding is needed because the trained Pipeline
# already contains the preprocessing step.
prediction = model.predict(new_data)[0]

print("\n" + "=" * 60)
print(f"Predicted Yield: {prediction:.2f} tons/hectare")
print("=" * 60)
