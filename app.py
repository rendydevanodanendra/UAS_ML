import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Income Predictor", layout="centered")

@st.cache_resource
def load_artifacts():
    # Load model dan encoder yang sudah Anda upload ke GitHub
    with open('model_income.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_enc = pickle.load(f)
    return model, encoder, label_enc

st.title("💰 Sistem Prediksi Pendapatan")

try:
    model, encoder, label_enc = load_artifacts()

    with st.form("my_form"):
        st.write("Masukkan Data Sensus:")
        age = st.number_input("Usia", 17, 90, 30)
        workclass = st.selectbox("Status Kerja", ['Private', 'Local-gov', 'State-gov', 'Self-emp-not-inc', 'Federal-gov', 'Without-pay', 'Never-worked'])
        fnlwgt = st.number_input("fnlwgt", value=77516)
        edu_num = st.slider("Education Num", 1, 16, 13)
        marital = st.selectbox("Status Pernikahan", ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
        occupation = st.selectbox("Pekerjaan", ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair', 'Transport-moving', 'Farming-fishing', 'Machine-op-inspct', 'Tech-support', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv'])
        relationship = st.selectbox("Hubungan", ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'])
        race = st.selectbox("Ras", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
        sex = st.selectbox("Jenis Kelamin", ['Male', 'Female'])
        cap_gain = st.number_input("Capital Gain", value=0)
        cap_loss = st.number_input("Capital Loss", value=0)
        hours = st.number_input("Jam Kerja per Minggu", 1, 99, 40)
        country = st.selectbox("Negara Asal", ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'Puerto-Rico', 'El-Salvador', 'India', 'Cuba', 'England', 'China', 'South', 'Jamaica', 'Italy', 'Japan', 'Guatemala', 'Columbia', 'Vietnam', 'Dominican-Republic', 'Poland', 'Haiti', 'Taiwan', 'Iran', 'Portugal', 'Nicaragua', 'Peru', 'Greece', 'France', 'Ecuador', 'Ireland', 'Hong', 'Trinadad&Tobago', 'Cambodia', 'Thailand', 'Laos', 'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Hungary', 'Honduras', 'Scotland', 'Holand-Netherlands'])
        
        submit = st.form_submit_button("Prediksi Sekarang")

    if submit:
        # Siapkan data untuk prediksi (pastikan urutan kolom sesuai dataset)
        # Catatan: Kolom 'education' dan 'native.country' diikutkan jika model memerlukannya
        input_data = pd.DataFrame([{
            'age': age, 'workclass': workclass, 'fnlwgt': fnlwgt, 
            'education': 'Bachelors', # Dummy karena ada edu_num
            'education.num': edu_num, 'marital.status': marital, 
            'occupation': occupation, 'relationship': relationship, 
            'race': race, 'sex': sex, 'capital.gain': cap_gain, 
            'capital.loss': cap_loss, 'hours.per.week': hours, 
            'native.country': country
        }])

        # Encoding kategori
        cat_cols = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
        input_data[cat_cols] = encoder.transform(input_data[cat_cols])

        # Prediksi
        res = model.predict(input_data)
        label = label_enc.inverse_transform(res)[0]
        
        st.divider()
        st.subheader(f"Hasil Prediksi: {label}")

except Exception as e:
    st.error(f"Gagal memuat sistem: {e}")
