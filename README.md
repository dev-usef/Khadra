# Khadra 🌱

Khadra is a Streamlit smart-agriculture dashboard built around the project’s existing trained machine-learning models. It helps users explore crop suitability and estimate crop yield from agricultural conditions.

## Features

- Crop recommendation from soil nutrients, pH, temperature, humidity, and rainfall
- Crop-yield prediction from crop, area, climate, nutrients, fertilizer, and irrigation inputs
- Dataset-derived input limits, friendly validation, and clear prediction results
- A models page showing evaluation metrics reproduced from the supplied training splits

## Project structure

```text
Khadra/
├── app/
│   └── app.py
├── data/
│   ├── crop_recommendation.csv
│   └── crop_yield.csv
├── models/
│   ├── crop_classifier.pkl
│   └── yield_regressor.pkl
├── notebooks/
│   ├── classification/
│   └── regression/
├── requirements.txt
└── README.md
```

## Install and run

Create and activate a Python virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Run the command from the project root so the application can locate `data/` and `models/`.

## ML tasks

**Crop Recommendation** uses the saved `RandomForestClassifier` to predict a crop label from the exact seven features used during training.

**Crop Yield Prediction** uses the saved preprocessing-and-linear-regression pipeline. The pipeline handles crop encoding itself, so the application sends the original feature names directly to it.

The `crop_yield.csv` data is a synthetic dataset. Both tools provide estimates for learning and decision-support purposes; they do not guarantee agricultural outcomes.
