import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# 1. Load & Clean Data
# =========================

df = pd.read_csv('data\\crop_yield.csv')


def remove_outliers_iqr(data, columns):
    df_out = data.copy()

    for col in columns:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_out = df_out[
            (df_out[col] >= lower_bound) &
            (df_out[col] <= upper_bound)
        ]

    return df_out


numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

print(f"Original DataFrame shape: {df.shape}")

df = remove_outliers_iqr(df, numerical_cols)

print(f"DataFrame shape after outlier removal: {df.shape}")


# =========================
# 2. Preprocessing
# =========================

numerical_features = [
    'Area_hectares',
    'Temperature_C',
    'Rainfall_mm',
    'N_kg_ha',
    'P_kg_ha',
    'K_kg_ha',
    'Fertilizer_kg_ha',
    'Irrigation_percent'
]

categorical_features = ['Crop']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

X = df.drop('Yield_tons_per_hectare', axis=1)
y = df['Yield_tons_per_hectare']

X_transformed = preprocessor.fit_transform(X)

cat_feature_names = (
    preprocessor
    .named_transformers_['cat']
    .get_feature_names_out(categorical_features)
)

all_feature_names = numerical_features + list(cat_feature_names)

X_transformed_df = pd.DataFrame(
    X_transformed,
    columns=all_feature_names
)

X_train, X_test, y_train, y_test = train_test_split(
    X_transformed_df,
    y,
    test_size=0.2,
    random_state=42
)

print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")


# =========================
# 3. Train Models
# =========================

# SVM
svm_regressor = SVR(kernel="linear", C=20.0)
svm_regressor.fit(X_train, y_train)

y_pred_svm = svm_regressor.predict(X_test)

mae_svm = mean_absolute_error(y_test, y_pred_svm)
mse_svm = mean_squared_error(y_test, y_pred_svm)
r2_svm = r2_score(y_test, y_pred_svm)


# Linear Regression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

y_pred_lr = linear_regressor.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)


# Decision Tree
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train, y_train)

y_pred_dt = dt_regressor.predict(X_test)

mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)


# KNN
knn_regressor = KNeighborsRegressor()
knn_regressor.fit(X_train, y_train)

y_pred_knn = knn_regressor.predict(X_test)

mae_knn = mean_absolute_error(y_test, y_pred_knn)
mse_knn = mean_squared_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)


# Ensemble
en_linear_regressor = LinearRegression()
en_knn_regressor = KNeighborsRegressor()

en_regressor = VotingRegressor(
    estimators=[
        ('lr', en_linear_regressor),
        ('knn', en_knn_regressor)
    ]
)

en_regressor.fit(X_train, y_train)

y_pred_ensemble = en_regressor.predict(X_test)

mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
mse_ensemble = mean_squared_error(y_test, y_pred_ensemble)
r2_ensemble = r2_score(y_test, y_pred_ensemble)


# =========================
# 4. Select Best Model
# =========================

r2_scores = {
    'SVM Regressor': r2_svm,
    'Linear Regression': r2_lr,
    'Decision Tree Regressor': r2_dt,
    'KNN Regressor': r2_knn,
    'Ensemble (LR + KNN)': r2_ensemble
}

model_map = {
    'SVM Regressor': svm_regressor,
    'Linear Regression': linear_regressor,
    'Decision Tree Regressor': dt_regressor,
    'KNN Regressor': knn_regressor,
    'Ensemble (LR + KNN)': en_regressor
}

best_model_name = max(r2_scores, key=r2_scores.get)
best_model = model_map[best_model_name]

print(f"\nBest Model: {best_model_name}")
print(f"Best R-squared: {r2_scores[best_model_name]:.4f}")


# =========================
# 5. Export Models
# =========================

os.makedirs("models", exist_ok=True)

# Complete deployment pipeline:
# raw input -> preprocessing -> trained model
deployment_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', best_model)
    ]
)

joblib.dump(
    deployment_pipeline,
    "models/yield_regressor.pkl"
)

joblib.dump(
    preprocessor,
    "models/preprocessor.pkl"
)

input_schema = {
    'numerical_features': numerical_features,
    'categorical_features': categorical_features,
    'target': 'Yield_tons_per_hectare'
}

joblib.dump(
    input_schema,
    "models/yield_input_schema.pkl"
)

print("\nExport completed:")
print(" - models/yield_regressor.pkl")
print(" - models/preprocessor.pkl")
print(" - models/yield_input_schema.pkl")
