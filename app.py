import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Income Predictor", layout="centered", page_icon="💰")

@st.cache_resource
def load_artifacts():
    # Load model dan encoder
    with open('model_income.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_enc = pickle.load(f)
    return model, encoder, label_enc

st.title("💰 Sistem Prediksi Pendapatan")
st.write("Aplikasi ini memprediksi apakah pendapatan seseorang di atas atau di bawah $50K berdasarkan data sensus.")

# --- SIDEBAR: Info Model ---
st.sidebar.header("Statistik Model")
# Nilai akurasi ini adalah performa model saat testing (bisa kamu sesuaikan nilainya)
st.sidebar.metric(label="Akurasi Model (Global)", value="86.2%") 
st.sidebar.write("Model: Gradient Boosting Classifier")
st.sidebar.divider()
st.sidebar.write("Dibuat oleh: Rendy Devano Danendra")

try:
    model, encoder, label_enc = load_artifacts()

    with st.form("my_form"):
        st.subheader("📝 Masukkan Data Sensus")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Usia", 17, 90, 30)
            workclass = st.selectbox("Status Kerja", ['Private', 'Local-gov', 'State-gov', 'Self-emp-not-inc', 'Federal-gov', 'Without-pay', 'Never-worked'])
            fnlwgt = st.number_input("fnlwgt (Final Weight)", value=77516)
            edu_num = st.slider("Education Num", 1, 16, 13)
            marital = st.selectbox("Status Pernikahan", ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
            
        with col2:
            occupation = st.selectbox("Pekerjaan", ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair', 'Transport-moving', 'Farming-fishing', 'Machine-op-inspct', 'Tech-support', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv'])
            relationship = st.selectbox("Hubungan", ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'])
            race = st.selectbox("Ras", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
            sex = st.selectbox("Jenis Kelamin", ['Male', 'Female'])
            hours = st.number_input("Jam Kerja per Minggu", 1, 99, 40)

        cap_gain = st.number_input("Capital Gain", value=0)
        cap_loss = st.number_input("Capital Loss", value=0)
        country = st.selectbox("Negara Asal", ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'Puerto-Rico', 'El-Salvador', 'India', 'Cuba', 'England', 'China', 'South', 'Jamaica', 'Italy', 'Japan', 'Guatemala', 'Columbia', 'Vietnam', 'Dominican-Republic', 'Poland', 'Haiti', 'Taiwan', 'Iran', 'Portugal', 'Nicaragua', 'Peru', 'Greece', 'France', 'Ecuador', 'Ireland', 'Hong', 'Trinadad&Tobago', 'Cambodia', 'Thailand', 'Laos', 'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Hungary', 'Honduras', 'Scotland', 'Holand-Netherlands'])
        
        submit = st.form_submit_button("Prediksi Sekarang")

    if submit:
        # Menyiapkan DataFrame
        input_data = pd.DataFrame([{
            'age': age, 'workclass': workclass, 'fnlwgt': fnlwgt, 
            'education': 'Bachelors', 
            'education.num': edu_num, 'marital.status': marital, 
            'occupation': occupation, 'relationship': relationship, 
            'race': race, 'sex': sex, 'capital.gain': cap_gain, 
            'capital.loss': cap_loss, 'hours.per.week': hours, 
            'native.country': country
        }])

        # Encoding
        cat_cols = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
        input_data[cat_cols] = encoder.transform(input_data[cat_cols])

        # --- LOGIKA PREDIKSI & AKURASI PREDIKSI ---
        res = model.predict(input_data)
        prob = model.predict_proba(input_data) # Probabilitas hasil
        confidence = max(prob[0]) * 100 # Persentase keyakinan model
        
        raw_label = label_enc.inverse_transform(res)[0]
        
        # Penentuan label Kaya/Gak Kaya
        if ">50K" in raw_label:
            keterangan = "Kaya"
            deskripsi = "Pendapatan > $50.000 / tahun"
            warna = "success"
        else:
            keterangan = "Gak Kaya"
            deskripsi = "Pendapatan <= $50.000 / tahun"
            warna = "error"

        st.divider()
        st.subheader("📊 Hasil Analisis Real-time")
        
        # Tampilan Grid untuk Hasil dan Akurasi
        col_res, col_acc = st.columns(2)
        
        with col_res:
            if warna == "success":
                st.success(f"### Status: {keterangan}")
            else:
                st.error(f"### Status: {keterangan}")
            st.info(f"**Keterangan:** {deskripsi}")
            
        with col_acc:
            st.metric(label="Akurasi Prediksi (Confidence)", value=f"{confidence:.2f}%")
            st.progress(confidence / 100)
            st.write("Seberapa yakin model dengan hasil ini.")

except Exception as e:
    st.error(f"Gagal memuat sistem: {e}")
