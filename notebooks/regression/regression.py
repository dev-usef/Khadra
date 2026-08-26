

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('data\\crop_yield.csv')
df.head()

df.info()

# Target: Yield
# Keep Crop as a feature, but encode it because it is categorical.
x = df.drop(columns=['Yield_tons_per_hectare'],axis=1)
y = df['Yield_tons_per_hectare']

df.info()

categorical_features = ['Crop']
numeric_features = [col for col in x.columns if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features),
       ('num', 'passthrough', numeric_features)
    ]
)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=42
)

print('x_train:', x_train.shape)
print('x_test:', x_test.shape)
print('y_train:', y_train.shape)
print('y_test:', y_test.shape)

# Pipeline: encode Crop, keep numeric features, then train Linear Regression.
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())])

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print('First 10 predictions:')
print(y_pred[:10])

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print('MAE:', mae)
print('MSE:', mse)
print('RMSE:', rmse)
print('R2 Score:', r2)

new_data = pd.DataFrame({
    'Crop': ['rice'],
    'Area_hectares': [20],
    'Temperature_C': [28],
    'Rainfall_mm': [300],
    'N_kg_ha': [80],
    'P_kg_ha': [40],
    'K_kg_ha': [40],
    'Fertilizer_kg_ha': [100],
    'Irrigation_percent': [70]
})

prediction = model.predict(new_data)

print("Predicted Yield:", prediction[0], "tons/hectare")

import joblib

joblib.dump(model, "models\\yield_regressor.pkl")

print("Model saved successfully!")
