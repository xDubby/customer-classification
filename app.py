import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

import traceback
import sys


    
# ── Configurazione pagina ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Classification",
    page_icon="🏦",
    layout="wide"
)

# ── Caricamento modelli ──────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    xgb    = joblib.load("src/model_xgb.pkl")
    rf     = joblib.load("src/model_rf.pkl")
    scaler = joblib.load("src/scaler.pkl")
    return xgb, rf, scaler

xgb_model, rf_model, scaler = load_models()

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🏦 Customer Credit Risk Classification")
st.markdown("Inserisci i dati del cliente per ottenere una previsione del rischio di credito.")
st.divider()

# ── Sidebar — input utente ───────────────────────────────────────────────────
st.sidebar.header("📋 Dati Cliente")

checking    = st.sidebar.selectbox("Conto corrente", ["<0 DM", "0-200 DM", ">200 DM", "no account"])
savings     = st.sidebar.selectbox("Risparmi", ["<100 DM", "100-500 DM", "500-1000 DM", ">1000 DM", "unknown/none"])
employment  = st.sidebar.selectbox("Anni di impiego", ["unemployed", "<1 year", "1-4 years", "4-7 years", ">7 years"])
duration    = st.sidebar.slider("Durata prestito (mesi)", 4, 72, 24)
credit_amount = st.sidebar.number_input("Importo credito (DM)", 100, 20000, 3000, step=100)
age         = st.sidebar.slider("Età", 18, 80, 35)
purpose     = st.sidebar.selectbox("Scopo prestito", ["car (new)", "car (used)", "furniture/equipment", "radio/TV", "education", "business", "repairs", "others"])
credit_history = st.sidebar.selectbox("Storia creditizia", ["no credits taken", "all credits paid back duly", "existing credits paid back duly", "delay in paying off", "critical account"])
housing     = st.sidebar.selectbox("Abitazione", ["own", "rent", "free"])
job         = st.sidebar.selectbox("Tipo di lavoro", ["skilled", "highly skilled", "unskilled resident", "unskilled non-resident"])

st.sidebar.divider()
model_choice = st.sidebar.radio("Modello", ["XGBoost (recall ottimizzato)", "Random Forest (AUC ottimizzato)"])

# ── Preprocessing input ──────────────────────────────────────────────────────
checking_map   = {"<0 DM": 0, "0-200 DM": 1, ">200 DM": 2, "no account": 3}
savings_map    = {"<100 DM": 0, "100-500 DM": 1, "500-1000 DM": 2, ">1000 DM": 3, "unknown/none": 4}
employment_map = {"unemployed": 0, "<1 year": 1, "1-4 years": 2, "4-7 years": 3, ">7 years": 4}
purpose_map    = {"car (new)": 0, "car (used)": 1, "furniture/equipment": 2, "radio/TV": 3, "education": 4, "business": 5, "repairs": 6, "others": 7}
history_map    = {"no credits taken": 0, "all credits paid back duly": 1, "existing credits paid back duly": 2, "delay in paying off": 3, "critical account": 4}
housing_map    = {"own": 0, "rent": 1, "free": 2}
job_map        = {"skilled": 0, "highly skilled": 1, "unskilled resident": 2, "unskilled non-resident": 3}

input_data = pd.DataFrame([{
    "checking":           checking_map[checking],
    "duration":           duration,
    "credit_history":     history_map[credit_history],
    "purpose":            purpose_map[purpose],
    "credit_amount":      credit_amount,
    "savings":            savings_map[savings],
    "employment":         employment_map[employment],
    "installment_rate":   2,
    "personal_status":    2,
    "guarantors":         0,
    "residence":          2,
    "property":           0,
    "age":                age,
    "other_installments": 2,
    "housing":            housing_map[housing],
    "existing_credits":   1,
    "job":                job_map[job],
    "dependents":         1,
    "phone":              0,
    "foreign":            0,
}])

# Scaling variabili numeriche
num_cols = ["duration", "credit_amount", "installment_rate", "residence", "age", "existing_credits", "dependents"]
input_scaled = input_data.copy()
input_scaled[num_cols] = scaler.transform(input_data[num_cols])

# ── Predizione ───────────────────────────────────────────────────────────────
model = xgb_model if "XGBoost" in model_choice else rf_model
proba = model.predict_proba(input_scaled)[0]
pred  = model.predict(input_scaled)[0]

prob_good = proba[0]
prob_bad  = proba[1]

# ── Output ───────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    if pred == 0:
        st.success("✅ Cliente: **GOOD**")
    else:
        st.error("⚠️ Cliente: **BAD**")

with col2:
    st.metric("Probabilità Good", f"{prob_good:.1%}")

with col3:
    st.metric("Probabilità Bad", f"{prob_bad:.1%}", delta=f"{prob_bad - 0.3:.1%} vs baseline")

st.divider()

# ── Grafico probabilità ───────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📊 Probabilità di rischio")
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.barh(["Good (0)", "Bad (1)"], [prob_good, prob_bad],
                   color=["steelblue", "tomato"], alpha=0.8)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    for bar, val in zip(bars, [prob_good, prob_bad]):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f"{val:.1%}", va="center", fontweight="bold")
    ax.set_xlabel("Probabilità")
    st.pyplot(fig)

with col_b:
    st.subheader("📋 Riepilogo input")
    st.dataframe(pd.DataFrame({
        "Feature": ["Conto corrente", "Risparmi", "Durata", "Importo", "Età", "Scopo", "Lavoro"],
        "Valore":  [checking, savings, f"{duration} mesi", f"{credit_amount} DM", f"{age} anni", purpose, job]
    }), hide_index=True, use_container_width=True)