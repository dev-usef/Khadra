"""Khadra Streamlit application — built around the saved project models only."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent
CLASSIFIER_PATH = ROOT / "models" / "crop_classifier.pkl"
REGRESSOR_PATH = ROOT / "models" / "yield_regressor.pkl"
CLASSIFICATION_DATA = ROOT / "data" / "crop_recommendation.csv"
YIELD_DATA = ROOT / "data" / "crop_yield.csv"

CLASSIFICATION_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
YIELD_FEATURES = ["Crop", "Area_hectares", "Temperature_C", "Rainfall_mm", "N_kg_ha", "P_kg_ha", "K_kg_ha", "Fertilizer_kg_ha", "Irrigation_percent"]
CROP_ICONS = {"rice": "🌾", "maize": "🌽", "banana": "🍌", "mango": "🥭", "apple": "🍎", "orange": "🍊", "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈", "coconut": "🥥", "coffee": "☕", "cotton": "🌿", "jute": "🌱"}


def icon(crop: str) -> str:
    return CROP_ICONS.get(crop.lower(), "🌱")


@st.cache_resource(show_spinner=False)
def load_models() -> tuple[Any, Any]:
    return joblib.load(CLASSIFIER_PATH), joblib.load(REGRESSOR_PATH)


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(CLASSIFICATION_DATA), pd.read_csv(YIELD_DATA)


def training_range(data: pd.DataFrame, column: str) -> tuple[float, float]:
    return float(data[column].min()), float(data[column].max())


def range_note(data: pd.DataFrame, column: str, unit: str = "") -> None:
    low, high = training_range(data, column)
    st.caption(f"Model training range: {low:g}–{high:g}{f' {unit}' if unit else ''}")


def warn_outside_range(data: pd.DataFrame, values: dict[str, float]) -> None:
    if any(not training_range(data, name)[0] <= value <= training_range(data, name)[1] for name, value in values.items()):
        st.warning("⚠️ One or more values are outside the model's training data range. The prediction may be less reliable.")


def inject_css() -> None:
    st.markdown("""
    <style>
      .stApp { background: linear-gradient(135deg,#06130d 0%,#092116 45%,#103b25 100%); color:#f3fff7; }
      .block-container { max-width:1250px; padding-top:2rem; padding-bottom:3rem; }
      .hero { padding:32px 38px; border-radius:26px; background:linear-gradient(135deg,rgba(34,197,94,.22),rgba(6,78,59,.32)); border:1px solid rgba(167,243,208,.22); box-shadow:0 15px 45px rgba(0,0,0,.22); margin-bottom:28px; }
      .hero h1 { font-size:clamp(2.3rem,5vw,3rem); margin:0; color:#dcfce7; font-weight:800; }.hero p { font-size:1.1rem; color:#bbf7d0; margin:6px 0; }.small { color:#a7f3d0; }
      .result { padding:32px; border-radius:24px; text-align:center; background:linear-gradient(135deg,rgba(34,197,94,.26),rgba(20,83,45,.50)); border:1px solid rgba(134,239,172,.32); box-shadow:0 15px 45px rgba(0,0,0,.18); margin:20px 0; }.result h1 { font-size:clamp(2.8rem,7vw,4rem); color:#dcfce7; margin:8px 0; }
      .metric-card { padding:18px; border-radius:18px; background:rgba(255,255,255,.055); border:1px solid rgba(255,255,255,.12); text-align:center; color:#f3fff7; }
      div[data-testid="stSidebar"] { background:linear-gradient(180deg,#04100a,#071b11); }
      [data-testid="stForm"] { border:1px solid rgba(167,243,208,.18); background:rgba(4,28,15,.58); border-radius:20px; padding:1.25rem; }
      [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span { color:#f3fff7 !important; font-weight:700 !important; }
      [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color:#b9d7c0 !important; }
      [data-testid="stNumberInput"] input, [data-baseweb="select"] > div, [data-baseweb="select"] span { color:#102c19 !important; font-weight:600; }
      [data-baseweb="select"] > div, [data-testid="stNumberInput"] input { background:#f7fff8 !important; border-radius:10px !important; }
      [role="listbox"] li { color:#102c19 !important; background:#fff !important; }
      [data-testid="stForm"] h3 { color:#dcfce7; font-weight:800; border-bottom:1px solid rgba(167,243,208,.2); padding-bottom:.55rem; }
      [data-testid="stFormSubmitButton"] { display:flex; justify-content:center; padding-top:.8rem; }
      [data-testid="stFormSubmitButton"] button { width:auto !important; min-width:210px; border-radius:14px !important; background:#22a55a !important; color:#fff !important; border:1px solid #86efac !important; padding:.8rem 1.4rem !important; font-weight:800 !important; cursor:pointer !important; box-shadow:0 8px 20px rgba(0,0,0,.23); }
      [data-testid="stFormSubmitButton"] button:hover { background:#15803d !important; color:#fff !important; transform:translateY(-1px); }
      [data-testid="stFormSubmitButton"] button:focus-visible { outline:3px solid #dcfce7; outline-offset:3px; }
      .stButton>button { border-radius:14px; font-weight:700; min-height:48px; background:#22a55a; color:#fff; border-color:#86efac; }
      @media(max-width:700px) { .hero { padding:24px; } [data-testid="stForm"] { padding:.9rem; } }
    </style>
    """, unsafe_allow_html=True)


def header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def show_home() -> None:
    header("🌱 Khadra", "Smart Machine Learning for Better Agricultural Decisions")
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="metric-card"><h2>🌾 Crop Recommendation</h2><p class="small">Match soil and environmental conditions to a crop with the saved classification model.</p></div>', unsafe_allow_html=True)
        if st.button("Explore Crop Recommendation", key="home_crop", use_container_width=True): st.session_state.page = "Crop Recommendation"; st.rerun()
    with right:
        st.markdown('<div class="metric-card"><h2>📈 Yield Prediction</h2><p class="small">Estimate yield per hectare using the saved regression pipeline.</p></div>', unsafe_allow_html=True)
        if st.button("Explore Yield Prediction", key="home_yield", use_container_width=True): st.session_state.page = "Yield Prediction"; st.rerun()


def show_crop_recommendation(classifier: Any, data: pd.DataFrame) -> None:
    header("🌾 Crop Recommendation", "Enter soil and climate conditions to get a model-based crop recommendation.")
    if list(classifier.feature_names_in_) != CLASSIFICATION_FEATURES:
        st.error("The saved classifier does not match the expected feature schema."); return
    with st.form("crop_form"):
        soil, climate = st.columns(2, gap="large"); values = {}
        with soil:
            st.subheader("Soil Conditions")
            for key, label, step in [("N", "Nitrogen (N)", 1.0), ("P", "Phosphorus (P)", 1.0), ("K", "Potassium (K)", 1.0), ("ph", "Soil pH", .1)]:
                kwargs = {"min_value":0.0, "value":float(data[key].median()), "step":step}
                if key == "ph": kwargs["max_value"] = 14.0
                values[key] = st.number_input(label, **kwargs); range_note(data, key)
        with climate:
            st.subheader("Environmental Conditions")
            values["temperature"] = st.number_input("Temperature (°C)", value=float(data.temperature.median()), step=.1); range_note(data, "temperature", "°C")
            values["humidity"] = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=float(data.humidity.median()), step=.1); range_note(data, "humidity", "%")
            values["rainfall"] = st.number_input("Rainfall (mm)", min_value=0.0, value=float(data.rainfall.median()), step=1.0); range_note(data, "rainfall", "mm")
        submitted = st.form_submit_button("🌾 Recommend Crop")
    if submitted:
        warn_outside_range(data, values)
        try:
            frame = pd.DataFrame([values], columns=CLASSIFICATION_FEATURES); crop = classifier.predict(frame)[0]
            st.markdown(f'<div class="result"><div class="small">RECOMMENDED CROP</div><h1>{icon(str(crop))} {str(crop).title()}</h1><p>Based on the soil and climate values you entered.</p></div>', unsafe_allow_html=True)
            if hasattr(classifier, "predict_proba"):
                st.subheader("Top 5 Predictions")
                for name, probability in sorted(zip(classifier.classes_, classifier.predict_proba(frame)[0]), key=lambda pair:pair[1], reverse=True)[:5]: st.progress(float(probability), text=f"{icon(str(name))} {str(name).title()} — {probability:.1%}")
        except Exception: st.error("We could not generate a recommendation. Please review the values and try again.")


def show_yield_prediction(regressor: Any, data: pd.DataFrame) -> None:
    header("📈 Crop Yield Prediction", "Estimate yield from crop, growing conditions, nutrients, and management inputs.")
    if list(regressor.feature_names_in_) != YIELD_FEATURES:
        st.error("The saved regression pipeline does not match the expected feature schema."); return
    with st.form("yield_form"):
        land, nutrients, management = st.columns(3, gap="large"); values = {}
        with land:
            st.subheader("Crop & Growing Area"); crops = sorted(data.Crop.dropna().astype(str).unique())
            values["Crop"] = st.selectbox("Crop", crops, format_func=lambda crop:f"{icon(crop)} {crop.title()}")
            for key, label, unit, minimum in [("Area_hectares","Area (hectares)","hectares",.01),("Temperature_C","Temperature (°C)","°C",None),("Rainfall_mm","Rainfall (mm)","mm",0.0)]:
                args={"value":float(data[key].median()), "step":.1 if key != "Rainfall_mm" else 1.0};
                if minimum is not None: args["min_value"] = minimum
                values[key]=st.number_input(label, **args); range_note(data,key,unit)
        with nutrients:
            st.subheader("Nutrients")
            for key,label in [("N_kg_ha","Nitrogen (kg/ha)"),("P_kg_ha","Phosphorus (kg/ha)"),("K_kg_ha","Potassium (kg/ha)")]: values[key]=st.number_input(label,min_value=0.0,value=float(data[key].median()),step=.1); range_note(data,key,"kg/ha")
        with management:
            st.subheader("Management"); values["Fertilizer_kg_ha"]=st.number_input("Fertilizer (kg/ha)",min_value=0.0,value=float(data.Fertilizer_kg_ha.median()),step=.1); range_note(data,"Fertilizer_kg_ha","kg/ha")
            values["Irrigation_percent"]=st.number_input("Irrigation (%)",min_value=0.0,max_value=100.0,value=float(data.Irrigation_percent.median()),step=.1); range_note(data,"Irrigation_percent","%")
        submitted=st.form_submit_button("🌱 Predict Yield")
    if submitted:
        warn_outside_range(data,{key:value for key,value in values.items() if key != "Crop"})
        try:
            result=float(regressor.predict(pd.DataFrame([values],columns=YIELD_FEATURES))[0])
            st.markdown(f'<div class="result"><div class="small">PREDICTED YIELD</div><h1>{result:.2f}</h1><h3>tons/hectare</h3><p>Estimated yield based on the provided conditions.</p></div>',unsafe_allow_html=True)
        except Exception: st.error("We could not calculate a yield estimate. Please review the values and try again.")


def show_models(classifier: Any, classification: pd.DataFrame, regression: pd.DataFrame) -> None:
    header("🤖 Models", "The saved models that power Khadra.")
    X=classification.drop(columns="label"); y=classification.label; _, xt, _, yt=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    accuracy=accuracy_score(yt,classifier.predict(xt))
    left,right=st.columns(2)
    with left: st.markdown(f'<div class="metric-card"><h2>🌾 {type(classifier).__name__}</h2><p class="small">Crop recommendation</p><h2>{accuracy:.2%}</h2><p class="small">Evaluation accuracy</p></div>',unsafe_allow_html=True)
    with right: st.markdown(f'<div class="metric-card"><h2>📈 LinearRegression Pipeline</h2><p class="small">Yield prediction with built-in crop encoding</p><h2>{len(regression):,}</h2><p class="small">Yield-dataset rows</p></div>',unsafe_allow_html=True)


def show_about() -> None:
    header("ℹ️ About Khadra", "Agricultural decision support powered by the project's trained ML models.")
    st.markdown('<div class="metric-card"><h3>Guidance, not a guarantee</h3><p class="small">Khadra provides model estimates from the supplied datasets. Use them alongside local field knowledge and professional agricultural advice.</p></div>',unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Khadra | Smart Agriculture",page_icon="🌱",layout="wide",initial_sidebar_state="expanded"); inject_css()
    try: classifier,regressor=load_models(); classification,regression=load_data()
    except FileNotFoundError: st.error("A required model or dataset file is missing. Please restore the models/ and data/ folders."); return
    except Exception: st.error("Khadra could not load its saved models. Please verify the project dependencies and files."); return
    pages=["Home","Crop Recommendation","Yield Prediction","Models","About"]
    if "page" not in st.session_state: st.session_state.page="Home"
    with st.sidebar:
        st.markdown("## 🌱 Khadra\n*Smart Agriculture*"); page=st.radio("Navigate",pages,index=pages.index(st.session_state.page))
    st.session_state.page=page
    {"Home":show_home,"Crop Recommendation":lambda:show_crop_recommendation(classifier,classification),"Yield Prediction":lambda:show_yield_prediction(regressor,regression),"Models":lambda:show_models(classifier,classification,regression),"About":show_about}[page]()


if __name__ == "__main__": main()
