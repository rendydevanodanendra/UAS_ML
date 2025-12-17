import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import OrdinalEncoder

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Adult Income Predictor", layout="wide")

# --- LOAD DATA UNTUK VISUALISASI ---
@st.cache_data
def load_data():
    df = pd.read_csv("adult.csv") # Pastikan file adult.csv ada di GitHub Anda
    df.replace("?", "unknown", inplace=True)
    return df

df_org = load_data()

# --- SIDEBAR: INPUT USER ---
st.sidebar.header("Input Data Sensus")

def user_input_features():
    age = st.sidebar.slider("Usia", 17, 90, 30)
    workclass = st.sidebar.selectbox("Status Kerja", df_org['workclass'].unique())
    fnlwgt = st.sidebar.number_input("fnlwgt", value=77516)
    education = st.sidebar.selectbox("Pendidikan", df_org['education'].unique())
    education_num = st.sidebar.slider("Education Num", 1, 16, 13)
    marital_status = st.sidebar.selectbox("Status Pernikahan", df_org['marital.status'].unique())
    occupation = st.sidebar.selectbox("Pekerjaan", df_org['occupation'].unique())
    relationship = st.sidebar.selectbox("Hubungan", df_org['relationship'].unique())
    race = st.sidebar.selectbox("Ras", df_org['race'].unique())
    sex = st.sidebar.selectbox("Jenis Kelamin", df_org['sex'].unique())
    cap_gain = st.sidebar.number_input("Capital Gain", value=0)
    cap_loss = st.sidebar.number_input("Capital Loss", value=0)
    hours_per_week = st.sidebar.slider("Jam Kerja per Minggu", 1, 99, 40)
    native_country = st.sidebar.selectbox("Negara Asal", df_org['native.country'].unique())

    data = {
        'age': age, 'workclass': workclass, 'fnlwgt': fnlwgt, 'education': education,
        'education.num': education_num, 'marital.status': marital_status,
        'occupation': occupation, 'relationship': relationship, 'race': race,
        'sex': sex, 'capital.gain': cap_gain, 'capital.loss': cap_loss,
        'hours.per.week': hours_per_week, 'native.country': native_country
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- MAIN PAGE ---
st.title("Aplikasi Prediksi Pendapatan Sensus")
st.markdown("""
Aplikasi ini memprediksi apakah pendapatan seseorang **>50K** atau **<=50K** berdasarkan data sensus menggunakan model *Gradient Boosting*.
""")

# --- BAGIAN VISUALISASI (Perbaikan Error Count) ---
st.subheader("Analisis Distribusi Data")
col1, col2 = st.columns(2)

with col1:
    # Perbaikan error 'count' untuk Education vs Sex
    df_plot1 = df_org.groupby(["education", "sex"]).size().reset_index(name='count')
    fig1, ax1 = plt.subplots()
    sns.barplot(data=df_plot1, x="education", y="count", hue="sex", ax=ax1)
    plt.xticks(rotation=90)
    st.pyplot(fig1)

with col2:
    # Perbaikan error 'count' untuk Hours vs Sex
    df_plot2 = df_org.groupby(["hours.per.week", "sex"]).size().reset_index(name='count')
    fig2, ax2 = plt.subplots()
    sns.barplot(data=df_plot2, x="hours.per.week", y="count", hue="sex", ax=ax2)
    st.pyplot(fig2)

# --- PROSES PREDIKSI ---
st.subheader("Hasil Prediksi")

# 1. Gabungkan input user dengan data asli untuk encoding yang konsisten
df_combined = pd.concat([input_df, df_org.drop('income', axis=1)], axis=0)

# 2. Encoding
cat_cols = df_org.select_dtypes(exclude='number').columns.drop('income')
encoder = OrdinalEncoder()
df_combined[cat_cols] = encoder.fit_transform(df_combined[cat_cols])

# 3. Ambil baris pertama (input user)
input_encoded = df_combined[:1]

# 4. Load Model (Pastikan file .pkl sudah diupload ke GitHub)
try:
    with open('model_income.pkl', 'rb') as f:
        model = pickle.load(f)
    
    prediction = model.predict(input_encoded)
    prediction_proba = model.predict_proba(input_encoded)

    if prediction[0] == 1:
        st.success("Prediksi Pendapatan: **>50K**")
    else:
        st.info("Prediksi Pendapatan: **<=50K**")

    st.write(f"Probabilitas: {np.max(prediction_proba)*100:.2f}%")

except FileNotFoundError:
    st.error("File 'model_income.pkl' tidak ditemukan. Harap upload model ke repositori GitHub Anda.")

# --- FOOTER ---
st.write("---")
st.write("Dikembangkan oleh Kelompok 9 Teknik Informatika - Universitas Trunojoyo Madura [cite: 11]")