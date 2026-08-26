"""Khadra Streamlit application — built around the saved project models only."""

from pathlib import Path
from typing import Any
from html import escape

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent
CLASSIFIER_PATH = ROOT / "models" / "random_forest_classifier.pkl"
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
      .stApp { background: radial-gradient(circle at 8% 5%,rgba(134,239,172,.10),transparent 28%), radial-gradient(circle at 92% 18%,rgba(34,197,94,.12),transparent 25%), linear-gradient(135deg,#03130b 0%,#062016 46%,#0b3b24 100%); color:#f5fff7; }
      .block-container { max-width:1320px; padding-top:1.4rem; padding-bottom:4rem; }
      div[data-testid="stSidebar"] { background:linear-gradient(180deg,#021008,#062117 60%,#0a2e1c); border-right:1px solid rgba(167,243,208,.12); }
      .hero { position:relative; overflow:hidden; padding:38px 42px; border-radius:30px; background:linear-gradient(135deg,rgba(34,197,94,.24),rgba(5,60,34,.68)); border:1px solid rgba(187,247,208,.22); box-shadow:0 24px 65px rgba(0,0,0,.28); margin-bottom:25px; }
      .hero:after { content:'🌿  🍃  🌱'; position:absolute; right:28px; bottom:13px; font-size:34px; opacity:.65; letter-spacing:10px; }
      .hero h1 { font-size:clamp(2.3rem,5vw,3.5rem); margin:0; color:#ecfff2; font-weight:900; letter-spacing:-1px; }.hero p { font-size:1.08rem; color:#b9f6ca; margin:8px 0 0; max-width:780px; }.small { color:#a7f3c0; }
      .metric-card { padding:24px; min-height:160px; border-radius:24px; background:linear-gradient(145deg,rgba(255,255,255,.075),rgba(255,255,255,.025)); border:1px solid rgba(187,247,208,.14); box-shadow:0 16px 35px rgba(0,0,0,.16); text-align:center; color:#f3fff7; }
      .result { padding:30px; border-radius:28px; text-align:center; background:linear-gradient(135deg,rgba(34,197,94,.30),rgba(20,83,45,.58)); border:1px solid rgba(134,239,172,.36); box-shadow:0 18px 55px rgba(0,0,0,.24); margin:24px 0; }.result h1 { font-size:clamp(2.8rem,7vw,4.5rem); color:#e9ffef; margin:8px 0; font-weight:900; }
      .section-title { color:#dcfce7; font-weight:900; font-size:1.15rem; margin:1rem 0 .5rem; }
      .leaf-divider { display:flex; justify-content:center; gap:18px; font-size:22px; margin:3px 0 20px; opacity:.75; }
      [data-testid="stForm"] { border:1px solid rgba(167,243,208,.18); background:linear-gradient(145deg,rgba(4,36,20,.75),rgba(4,24,14,.58)); border-radius:25px; padding:1.35rem; box-shadow:0 16px 40px rgba(0,0,0,.18); }
      [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span { color:#f3fff7 !important; font-weight:700 !important; }
      [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color:#a9cfb2 !important; }
      [data-testid="stNumberInput"] input, [data-baseweb="select"] > div, [data-baseweb="select"] span { color:#102c19 !important; font-weight:650; }
      [data-baseweb="select"] > div, [data-testid="stNumberInput"] input { background:#f7fff8 !important; border-radius:12px !important; }
      [role="listbox"] li { color:#102c19 !important; background:#fff !important; }
      [data-testid="stForm"] h3 { color:#dcfce7; font-weight:850; border-bottom:1px solid rgba(167,243,208,.2); padding-bottom:.55rem; }
      [data-testid="stFormSubmitButton"] { display:flex; justify-content:center; padding-top:.8rem; }
      [data-testid="stFormSubmitButton"] button, .stButton>button { border-radius:15px !important; background:linear-gradient(135deg,#22a55a,#15803d) !important; color:#fff !important; border:1px solid #86efac !important; padding:.8rem 1.4rem !important; font-weight:850 !important; box-shadow:0 10px 25px rgba(0,0,0,.24); }
      [data-testid="stFormSubmitButton"] button:hover, .stButton>button:hover { transform:translateY(-2px); box-shadow:0 14px 30px rgba(34,197,94,.22); }
      [data-testid="stMetric"] { background:rgba(255,255,255,.045); border:1px solid rgba(187,247,208,.12); padding:15px 18px; border-radius:18px; }
      [data-testid="stMetricValue"] { color:#dcfce7 !important; }

      .chart-shell { margin:18px 0 24px; padding:22px; border-radius:28px; background:linear-gradient(145deg,rgba(255,255,255,.075),rgba(255,255,255,.018)); border:1px solid rgba(187,247,208,.15); box-shadow:0 18px 50px rgba(0,0,0,.22); }
      .chart-head { display:flex; justify-content:space-between; align-items:flex-end; gap:16px; margin-bottom:18px; }
      .chart-title { color:#ecfff2; font-size:1.08rem; font-weight:900; letter-spacing:.2px; }
      .chart-sub { color:#8fcaa0; font-size:.82rem; margin-top:3px; }
      .chart-badge { padding:7px 12px; border-radius:999px; background:rgba(34,197,94,.12); border:1px solid rgba(134,239,172,.25); color:#b9f6ca; font-weight:800; font-size:.78rem; white-space:nowrap; }
      .pred-layout { display:grid; grid-template-columns:220px 1fr; gap:28px; align-items:center; }
      .donut-wrap { display:flex; justify-content:center; align-items:center; }
      .donut { width:178px; height:178px; border-radius:50%; display:flex; align-items:center; justify-content:center; box-shadow:0 0 45px rgba(34,197,94,.16); position:relative; }
      .donut:after { content:''; width:118px; height:118px; border-radius:50%; background:#071b10; border:1px solid rgba(187,247,208,.12); position:absolute; }
      .donut-center { position:relative; z-index:2; text-align:center; }
      .donut-value { color:#ecfff2; font-size:1.65rem; font-weight:950; }
      .donut-label { color:#8fcaa0; font-size:.72rem; margin-top:2px; }
      .bars { display:flex; flex-direction:column; gap:12px; }
      .bar-row { display:grid; grid-template-columns:120px 1fr 62px; gap:12px; align-items:center; }
      .bar-label { color:#e9fff0; font-weight:750; font-size:.86rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
      .bar-track { height:13px; border-radius:999px; background:rgba(255,255,255,.07); overflow:hidden; box-shadow:inset 0 1px 4px rgba(0,0,0,.25); }
      .bar-fill { height:100%; border-radius:999px; background:linear-gradient(90deg,#0f7a3a,#34d477,#b8f7c8); box-shadow:0 0 16px rgba(52,212,119,.24); }
      .bar-value { text-align:right; color:#b9f6ca; font-weight:850; font-size:.82rem; }
      .yield-layout { display:grid; grid-template-columns:1fr 230px; gap:28px; align-items:center; }
      .yield-chart { position:relative; height:235px; padding:10px 8px 0; }
      .yield-axis { position:absolute; left:0; right:0; bottom:25px; height:1px; background:rgba(187,247,208,.14); }
      .yield-bars { height:185px; display:flex; align-items:flex-end; justify-content:space-around; gap:18px; padding:0 10px 25px; }
      .ybar-wrap { height:100%; flex:1; display:flex; flex-direction:column; align-items:center; justify-content:flex-end; }
      .ybar-value { color:#dfffea; font-size:.78rem; font-weight:850; margin-bottom:7px; }
      .ybar { width:min(62px,70%); min-height:10px; border-radius:14px 14px 5px 5px; background:linear-gradient(180deg,#74e6a0,#159447); box-shadow:0 0 22px rgba(34,197,94,.17); }
      .ybar.pred { background:linear-gradient(180deg,#d5ffe1,#22c55e,#0b6e34); box-shadow:0 0 28px rgba(74,222,128,.35); }
      .ybar-label { margin-top:9px; color:#8fcaa0; font-size:.7rem; text-align:center; white-space:nowrap; }
      .gauge-card { text-align:center; padding:18px; border-radius:24px; background:rgba(0,0,0,.12); border:1px solid rgba(187,247,208,.1); }
      .gauge { width:190px; height:95px; margin:10px auto 0; border-radius:190px 190px 0 0; background:conic-gradient(from 270deg at 50% 100%, #0d6f36 0deg, #22c55e 120deg, #b8f7c8 180deg, transparent 180deg); position:relative; overflow:hidden; }
      .gauge:after { content:''; position:absolute; width:138px; height:69px; left:26px; bottom:0; background:#071b10; border-radius:138px 138px 0 0; border:1px solid rgba(187,247,208,.1); }
      .gauge-needle { position:absolute; width:3px; height:77px; left:calc(50% - 1px); bottom:0; background:#ecfff2; transform-origin:bottom center; z-index:3; border-radius:5px; box-shadow:0 0 12px rgba(255,255,255,.5); }
      .gauge-dot { position:absolute; width:12px; height:12px; border-radius:50%; background:#ecfff2; left:calc(50% - 6px); bottom:-1px; z-index:4; }
      .gauge-number { margin-top:10px; color:#ecfff2; font-size:1.9rem; font-weight:950; }
      .gauge-note { color:#8fcaa0; font-size:.76rem; }
      .importance-list { display:flex; flex-direction:column; gap:14px; }
      .imp-row { display:grid; grid-template-columns:130px 1fr 58px; gap:12px; align-items:center; }
      .imp-label { color:#e9fff0; font-weight:750; font-size:.84rem; }
      .imp-track { height:12px; border-radius:999px; background:rgba(255,255,255,.06); overflow:hidden; }
      .imp-fill { height:100%; border-radius:999px; background:linear-gradient(90deg,#0b6e34,#22c55e,#d5ffe1); }
      .imp-value { color:#b9f6ca; text-align:right; font-size:.8rem; font-weight:850; }
      @media(max-width:850px) { .pred-layout,.yield-layout { grid-template-columns:1fr; } .bar-row { grid-template-columns:105px 1fr 52px; } .gauge-card { max-width:330px; margin:auto; } }
      @media(max-width:700px) { .hero { padding:25px; }.hero:after { font-size:22px; }.metric-card { min-height:140px; } }
    </style>
    """, unsafe_allow_html=True)


def header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)



def leaf_divider() -> None:
    st.markdown('<div class="leaf-divider"><span>🌿</span><span>🍃</span><span>🌱</span><span>🍃</span><span>🌿</span></div>', unsafe_allow_html=True)


def prediction_bar_chart(names, probabilities) -> None:
    pairs = sorted([(str(n).title(), float(v)) for n, v in zip(names, probabilities)], key=lambda x: x[1], reverse=True)[:5]
    if not pairs:
        return
    top_prob = pairs[0][1]
    deg = max(2.0, min(360.0, top_prob * 360.0))
    stops = f"#0d6f36 0deg, #22c55e {deg*.68:.1f}deg, #b8f7c8 {deg:.1f}deg, rgba(255,255,255,.05) {deg:.1f}deg 360deg"
    bars = []
    for name, prob in pairs:
        width = max(1.5, prob * 100)
        bars.append(f'<div class="bar-row"><div class="bar-label">{escape(icon(name))} {escape(name)}</div><div class="bar-track"><div class="bar-fill" style="width:{width:.2f}%"></div></div><div class="bar-value">{prob:.1%}</div></div>')
    bars_html=''.join(bars)
    html = ('<div class="chart-shell"><div class="chart-head"><div><div class="chart-title">Prediction Confidence</div><div class="chart-sub">Top 5 model probabilities</div></div><div class="chart-badge">Random Forest</div></div>'
            + f'<div class="pred-layout"><div class="donut-wrap"><div class="donut" style="background:conic-gradient({stops})"><div class="donut-center"><div class="donut-value">{top_prob:.1%}</div><div class="donut-label">top confidence</div></div></div></div><div class="bars">{bars_html}</div></div></div>')
    st.markdown(html, unsafe_allow_html=True)


def yield_chart(predicted: float, data: pd.DataFrame) -> None:
    if "Yield_tons_per_hectare" not in data.columns:
        return
    vals = {"Prediction": float(predicted), "Dataset median": float(data["Yield_tons_per_hectare"].median()), "Dataset minimum": float(data["Yield_tons_per_hectare"].min()), "Dataset maximum": float(data["Yield_tons_per_hectare"].max())}
    lo, hi = vals["Dataset minimum"], vals["Dataset maximum"]
    span = max(hi-lo, 1e-9)
    heights = {k:max(8,min(100,((v-lo)/span)*100)) for k,v in vals.items()}
    bars=[]
    for k in ["Dataset minimum","Dataset median","Prediction","Dataset maximum"]:
        cls='ybar pred' if k=='Prediction' else 'ybar'
        bars.append(f'<div class="ybar-wrap"><div class="ybar-value">{vals[k]:.2f}</div><div class="{cls}" style="height:{heights[k]:.1f}%"></div><div class="ybar-label">{escape(k.replace("Dataset ",""))}</div></div>')
    bars_html=''.join(bars)
    pos=max(0,min(1,(predicted-lo)/span))
    angle=-90+pos*180
    html = ('<div class="chart-shell"><div class="chart-head"><div><div class="chart-title">Yield Performance View</div><div class="chart-sub">Prediction compared with the training dataset range</div></div><div class="chart-badge">tons / hectare</div></div>'
            + f'<div class="yield-layout"><div class="yield-chart"><div class="yield-bars">{bars_html}</div><div class="yield-axis"></div></div><div class="gauge-card"><div class="chart-title">Position in Range</div><div class="gauge"><div class="gauge-needle" style="transform:rotate({angle:.1f}deg)"></div><div class="gauge-dot"></div></div><div class="gauge-number">{predicted:.2f}</div><div class="gauge-note">relative to min → max</div></div></div></div>')
    st.markdown(html, unsafe_allow_html=True)


def show_home() -> None:
    header("🌱 Khadra", "Smart Machine Learning for Better Agricultural Decisions")
    leaf_divider()
    a,b,c=st.columns(3)
    with a: st.metric("🌾 Crop Classes", "22")
    with b: st.metric("📊 Classification", "99.55%")
    with c: st.metric("📈 Yield R²", "97.62%")
    st.markdown("<br>",unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="metric-card"><div style="font-size:42px">🌾</div><h2>Crop Recommendation</h2><p class="small">Match soil and environmental conditions to the most suitable crop, then explore the model probabilities visually.</p></div>', unsafe_allow_html=True)
        if st.button("Explore Crop Recommendation →", key="home_crop", use_container_width=True): st.session_state.page = "Crop Recommendation"; st.rerun()
    with right:
        st.markdown('<div class="metric-card"><div style="font-size:42px">📈</div><h2>Yield Prediction</h2><p class="small">Estimate tons per hectare and compare the prediction with the dataset distribution.</p></div>', unsafe_allow_html=True)
        if st.button("Explore Yield Prediction →", key="home_yield", use_container_width=True): st.session_state.page = "Yield Prediction"; st.rerun()
    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown('<div class="metric-card"><h3>🌿 How Khadra works</h3><p class="small">Data → EDA → Machine Learning → Prediction → Visual Insight</p></div>',unsafe_allow_html=True)

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
                st.markdown('<div class="section-title">📊 Prediction probabilities</div>', unsafe_allow_html=True)
                pairs=sorted(zip(classifier.classes_, classifier.predict_proba(frame)[0]), key=lambda pair:pair[1], reverse=True)[:5]
                names=[p[0] for p in pairs]; probs=[p[1] for p in pairs]
                prediction_bar_chart(names, probs)
                for name, probability in pairs:
                    st.progress(float(probability), text=f"{icon(str(name))} {str(name).title()} — {probability:.1%}")
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
            st.markdown('<div class="section-title">📊 Prediction vs dataset</div>', unsafe_allow_html=True)
            yield_chart(result, data)
        except Exception: st.error("We could not calculate a yield estimate. Please review the values and try again.")


def show_models(classifier: Any, classification: pd.DataFrame, regression: pd.DataFrame) -> None:
    header("🤖 Models", "The saved models that power Khadra.")
    X=classification.drop(columns="label"); y=classification.label; _, xt, _, yt=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    accuracy=accuracy_score(yt,classifier.predict(xt))
    left,right=st.columns(2)
    with left: st.markdown(f'<div class="metric-card"><h2>🌾 {type(classifier).__name__}</h2><p class="small">Crop recommendation</p><h2>{accuracy:.2%}</h2><p class="small">Evaluation accuracy</p></div>',unsafe_allow_html=True)
    with right: st.markdown(f'<div class="metric-card"><h2>📈 LinearRegression Pipeline</h2><p class="small">Yield prediction with built-in crop encoding</p><h2>{len(regression):,}</h2><p class="small">Yield-dataset rows</p></div>',unsafe_allow_html=True)
    if hasattr(classifier, "feature_importances_"):
        st.markdown('<div class="section-title">🌿 Classification feature importance</div>', unsafe_allow_html=True)
        fi=pd.DataFrame({"Feature":CLASSIFICATION_FEATURES,"Importance":classifier.feature_importances_}).sort_values("Importance",ascending=False)
        max_imp=float(fi["Importance"].max()) if len(fi) else 1.0
        rows=[]
        for _, r in fi.iterrows():
            pct=float(r["Importance"])/max_imp*100 if max_imp else 0
            rows.append(f'<div class="imp-row"><div class="imp-label">{escape(str(r["Feature"]))}</div><div class="imp-track"><div class="imp-fill" style="width:{pct:.2f}%"></div></div><div class="imp-value">{float(r["Importance"]):.3f}</div></div>')
        rows_html=''.join(rows)
        st.markdown(f'<div class="chart-shell"><div class="chart-head"><div><div class="chart-title">What drives the recommendation?</div><div class="chart-sub">Relative feature importance from the trained Random Forest</div></div><div class="chart-badge">Model Insight</div></div><div class="importance-list">{rows_html}</div></div>', unsafe_allow_html=True)


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
