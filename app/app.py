import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="CropVision AI", page_icon="🌱", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp{background:linear-gradient(135deg,#06130d 0%,#092116 45%,#103b25 100%);color:#f3fff7}
.block-container{max-width:1250px;padding-top:2rem;padding-bottom:3rem}
.hero{padding:32px 38px;border-radius:26px;background:linear-gradient(135deg,rgba(34,197,94,.22),rgba(6,78,59,.32));border:1px solid rgba(167,243,208,.22);box-shadow:0 15px 45px rgba(0,0,0,.22);margin-bottom:28px}
.hero h1{font-size:48px;margin:0;color:#dcfce7;font-weight:800}.hero p{font-size:18px;color:#bbf7d0;margin:6px 0}.small{color:#a7f3d0}
.result{padding:32px;border-radius:24px;text-align:center;background:linear-gradient(135deg,rgba(34,197,94,.26),rgba(20,83,45,.50));border:1px solid rgba(134,239,172,.32);box-shadow:0 15px 45px rgba(0,0,0,.18);margin:20px 0}.result h1{font-size:44px;color:#bbf7d0;margin:5px}
.metric-card{padding:18px;border-radius:18px;background:rgba(255,255,255,.055);border:1px solid rgba(255,255,255,.09);text-align:center}
div[data-testid="stSidebar"]{background:linear-gradient(180deg,#04100a,#071b11)}
.stButton>button{border-radius:14px;font-weight:700;min-height:48px}
</style>
""", unsafe_allow_html=True)

if not os.path.exists("crop_classifier.pkl"):
    st.error("Model file not found. Run `python classification.py` first, then `streamlit run app.py`.")
    st.stop()

model = joblib.load("crop_classifier.pkl")
df = pd.read_csv("Crop_recommendation.csv")

X = df.drop("label", axis=1)
y = df["label"]
_, X_test, _, y_test = train_test_split(X, y, test_size=.20, random_state=42, stratify=y)
y_test_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_test_pred)

st.markdown("""
<div class="hero">
<h1>🌱 CropVision AI</h1>
<p>Smart Crop Recommendation System powered by Machine Learning</p>
<p class="small">Enter soil & climate conditions and let the Classification model recommend the most suitable crop.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 🌿 CropVision")
st.sidebar.caption("Random Forest Classification")
page = st.sidebar.radio("Navigate", ["🌱 Prediction", "📊 Model Insights", "📁 Dataset"])
st.sidebar.divider()
st.sidebar.metric("Model Accuracy", f"{accuracy*100:.2f}%")
st.sidebar.metric("Crop Classes", df["label"].nunique())
st.sidebar.metric("Dataset Rows", len(df))

if page == "🌱 Prediction":
    st.subheader("🔮 Crop Prediction")
    st.write("Enter the seven input features. The model predicts the crop from the `label` column.")
    c1,c2,c3=st.columns(3)
    with c1:
        N=st.number_input("Nitrogen (N)",0.0,200.0,90.0,0.1)
        P=st.number_input("Phosphorus (P)",0.0,200.0,42.0,0.1)
        K=st.number_input("Potassium (K)",0.0,250.0,43.0,0.1)
    with c2:
        temperature=st.number_input("Temperature (°C)",-10.0,60.0,20.8,0.1)
        humidity=st.number_input("Humidity (%)",0.0,100.0,82.0,0.1)
    with c3:
        ph=st.number_input("Soil pH",0.0,14.0,6.5,0.01)
        rainfall=st.number_input("Rainfall (mm)",0.0,5000.0,202.9,0.1)

    new_data=pd.DataFrame([{"N":N,"P":P,"K":K,"temperature":temperature,"humidity":humidity,"ph":ph,"rainfall":rainfall}])

    if st.button("🚀 Predict Best Crop",use_container_width=True,type="primary"):
        prediction=model.predict(new_data)[0]
        probabilities=model.predict_proba(new_data)[0]
        classes=model.classes_
        order=np.argsort(probabilities)[::-1][:5]
        top=[(classes[i],probabilities[i]) for i in order]
        confidence=float(probabilities[list(classes).index(prediction)])

        st.balloons()
        st.markdown(f'<div class="result"><div class="small">Recommended Crop</div><h1>🌾 {prediction.replace("_"," ").title()}</h1><p>Based on the soil and climate values you entered.</p></div>',unsafe_allow_html=True)
        st.subheader("🎯 Prediction Confidence")
        st.progress(confidence)
        st.write(f"**{confidence*100:.2f}%** model probability")
        st.subheader("🏆 Top 5 Predictions")
        cols=st.columns(5)
        for col,(crop,prob) in zip(cols,top):
            with col:
                st.markdown(f'<div class="metric-card"><b>🌱 {crop.title()}</b><br><span class="small">{prob*100:.2f}%</span></div>',unsafe_allow_html=True)
        chart=pd.DataFrame(top,columns=["Crop","Probability"])
        fig,ax=plt.subplots(figsize=(8,4)); ax.barh(chart["Crop"].str.title()[::-1],chart["Probability"][::-1]*100); ax.set_xlabel("Probability (%)"); ax.set_title("Top Crop Predictions"); plt.tight_layout(); st.pyplot(fig)

elif page == "📊 Model Insights":
    st.subheader("📊 Model Insights")
    a,b,c=st.columns(3)
    with a: st.markdown(f'<div class="metric-card"><h2>{accuracy*100:.2f}%</h2><span class="small">Accuracy</span></div>',unsafe_allow_html=True)
    with b: st.markdown(f'<div class="metric-card"><h2>{df["label"].nunique()}</h2><span class="small">Crop Classes</span></div>',unsafe_allow_html=True)
    with c: st.markdown(f'<div class="metric-card"><h2>{len(df)}</h2><span class="small">Samples</span></div>',unsafe_allow_html=True)

    st.subheader("🌟 Feature Importance")
    imp=pd.Series(model.feature_importances_,index=X.columns).sort_values(ascending=False)
    fig,ax=plt.subplots(figsize=(9,4)); ax.bar(imp.index,imp.values); ax.set_ylabel("Importance"); ax.set_title("Random Forest Feature Importance"); plt.xticks(rotation=30); plt.tight_layout(); st.pyplot(fig)

    st.subheader("🧩 Confusion Matrix")
    labels=model.classes_; cm=confusion_matrix(y_test,y_test_pred,labels=labels)
    fig,ax=plt.subplots(figsize=(12,9)); im=ax.imshow(cm); ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels))); ax.set_xticklabels(labels,rotation=90); ax.set_yticklabels(labels); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix"); fig.colorbar(im,ax=ax); plt.tight_layout(); st.pyplot(fig)

    st.subheader("📋 Classification Report")
    report=classification_report(y_test,y_test_pred,output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(3),use_container_width=True)

else:
    st.subheader("📁 Dataset Overview")
    st.dataframe(df.head(20),use_container_width=True)
    st.subheader("📊 Class Distribution")
    counts=df["label"].value_counts(); fig,ax=plt.subplots(figsize=(10,5)); ax.bar(counts.index,counts.values); ax.set_ylabel("Samples"); ax.set_xlabel("Crop"); ax.set_title("Number of Samples per Crop"); plt.xticks(rotation=75); plt.tight_layout(); st.pyplot(fig)
    st.subheader("📈 Basic Statistics")
    st.dataframe(df.describe().round(2),use_container_width=True)

st.divider()
st.caption("CropVision AI • Random Forest Classification • Python + Streamlit")
