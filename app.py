import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Konfigurasi Halaman
st.set_page_config(page_title="Income Predictor", layout="centered")

# Fungsi untuk memuat model dan encoder
@st.cache_resource
def load_artifacts():
    # Pastikan file-file ini sudah ada di repositori GitHub yang sama
    model_path = 'model_income.pkl'
    encoder_path = 'encoder.pkl'
    label_enc_path = 'label_encoder.pkl'
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    with open(label_enc_path, 'rb') as f:
        label_enc = pickle.load(f)
        
    return model, encoder, label_enc

    st.title("💰 Prediksi Pendapatan Sensus")
    st.write("Masukkan data di bawah ini untuk memprediksi kategori pendapatan.")

    # Form Input User (Fokus pada 14 fitur sesuai dataset)
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Usia", min_value=17, max_value=90, value=30)
            workclass = st.selectbox("Status Kerja", ['Private', 'Self-emp-not-inc', 'Local-gov', 'unknown', 'State-gov', 'Self-emp-inc', 'Federal-gov', 'Without-pay', 'Never-worked'])
            fnlwgt = st.number_input("fnlwgt (Final Weight)", value=77516)
            education = st.selectbox("Pendidikan", ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school', '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])
            edu_num = st.slider("Education Num", 1, 16, 13)
            marital = st.selectbox("Status Pernikahan", ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
            occupation = st.selectbox("Pekerjaan", ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair', 'Transport-moving', 'Farming-fishing', 'Machine-op-inspct', 'Tech-support', 'unknown', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv'])

        with col2:
            relationship = st.selectbox("Hubungan", ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'])
            race = st.selectbox("Ras", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
            sex = st.selectbox("Jenis Kelamin", ['Male', 'Female'])
            cap_gain = st.number_input("Capital Gain", value=0)
            cap_loss = st.number_input("Capital Loss", value=0)
            hours = st.number_input("Jam Kerja per Minggu", min_value=1, max_value=99, value=40)
            country = st.selectbox("Negara Asal", ['United-States', 'Cuba', 'Jamaica', 'India', 'unknown', 'Mexico', 'South', 'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran', 'Philippines', 'Italy', 'Poland', 'Columbia', 'Cambodia', 'Thailand', 'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal', 'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala', 'China', 'Japan', 'Yugoslavia', 'Peru', 'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago', 'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary', 'Holand-Netherlands'])

        submit = st.form_submit_button("Prediksi")

    if submit:
        # 1. Kumpulkan data input
        input_data = pd.DataFrame([{
            'age': age, 'workclass': workclass, 'fnlwgt': fnlwgt, 'education': education,
            'education.num': edu_num, 'marital.status': marital, 'occupation': occupation,
            'relationship': relationship, 'race': race, 'sex': sex,
            'capital.gain': cap_gain, 'capital.loss': cap_loss,
            'hours.per.week': hours, 'native.country': country
        }])

        # 2. Transformasi kategori menggunakan encoder yang sudah disimpan
        cat_cols = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
        input_data[cat_cols] = encoder.transform(input_data[cat_cols])

        # 3. Lakukan Prediksi
        pred_numeric = model.predict(input_data)
        result = label_enc.inverse_transform(pred_numeric)

        # 4. Tampilkan Hasil
        st.write("---")
        if result[0] == '>50K':
            st.success(f"Hasil Prediksi: **{result[0]}** (Penghasilan Tinggi)")
        else:
            st.info(f"Hasil Prediksi: **{result[0]}** (Penghasilan Rendah)")
            
except FileNotFoundError:
    st.error("Error: File model atau encoder tidak ditemukan. Pastikan sudah menjalankan tahap export di notebook.")


