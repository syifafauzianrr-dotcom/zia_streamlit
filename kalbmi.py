import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# =============== Konfigurasi Halaman ===============
st.set_page_config(page_title="Aplikasi Kesehatan — Obesitas", layout="wide")

# =============== CSS Tema ==========================
health_mint_css = """
<style>
[data-testid="stSidebar"] {
    background-color: #E8FFF4;
}
[data-testid="stSidebar"] * {
    color: #1E5631 !important;
}
div[role="radiogroup"] label {
    color: #1E5631 !important;
    font-weight: 600;
}
h1 {
    color: #1E5631 !important;
}
h2, h3, h4 {
    color: #2ECC71 !important;
}
[data-testid="stMetricLabel"] {
    color: #1E5631 !important;
}
.stButton>button {
    background-color: #2ECC71;
    color: white;
    border-radius: 8px;
    border: none;
}
.stButton>button:hover {
    background-color: #27AE60;
}
</style>
"""
st.markdown(health_mint_css, unsafe_allow_html=True)

# ========= Fungsi Perhitungan ===============
def calculate_bmi(weight, height_cm):
    height_m = height_cm / 100
    return weight / (height_m ** 2) if height_m > 0 else None

def bmi_category(bmi):
    if bmi < 18.5:
        return "Kurus (Underweight)"
    elif 18.5 <= bmi < 25:
        return "Normal"
    elif 25 <= bmi < 30:
        return "Gemuk (Overweight)"
    return "Obesitas"

def waist_risk(sex, waist_cm):
    if sex == "Laki-laki":
        return "Tinggi" if waist_cm > 94 else "Normal"
    return "Tinggi" if waist_cm > 80 else "Normal"

# ============= Dataset Sintetis ============
def generate_synthetic_dataset():
    np.random.seed(42)
    n = 300
    heights = np.random.normal(165, 10, n).clip(140, 200)
    weights = np.random.normal(75, 15, n).clip(40, 160)
    ages = np.random.randint(18, 70, n)
    sexes = np.random.choice(["Laki-laki", "Perempuan"], n)
    waists = (weights / (heights/100) * 0.5 * np.random.uniform(0.9, 1.1, n)).clip(60, 150)

    return pd.DataFrame({
        "weight_kg": weights,
        "height_cm": heights,
        "age": ages,
        "sex": sexes,
        "waist_cm": waists
    })

#========== Model Logistic Regression ============
def train_obesity_model(df):
    df = df.copy()
    df["is_obese"] = (df["bmi"] >= 30).astype(int)

    X = df[["weight_kg", "height_cm", "age"]].values
    y = df["is_obese"].values

    if len(df) < 30:
        return None, None, None, "Dataset terlalu sedikit (minimal 30 baris)."

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train_s, y_train)

    score = clf.score(X_test_s, y_test)
    return clf, scaler, score, None

# ========= Plot Histogram BMI ==========
def plot_bmi_histogram(df):
    fig, ax = plt.subplots()
    ax.hist(df["bmi"], bins=20)
    ax.set_xlabel("BMI")
    ax.set_ylabel("Jumlah individu")
    ax.set_title("Distribusi BMI")
    return fig

# =========== MENU ======================
menu = st.sidebar.radio(
    "Menu",
    ["🏠 Home", "📊 Kalkulator BMI", "📈 Analisis Data", "🧩 Model Prediksi", "💻 Tentang Aplikasi"]
)

# ============= HOME ====================
if menu == "🏠 Home":
    st.title("🏠 Aplikasi Sederhana Edukasi & Kalkulator Obesitas")
    st.write("""
    Selamat datang di aplikasi edukasi obesitas.  
    Gunakan menu di sidebar untuk mengakses fitur:
    - Kalkulator BMI  
    - Analisis Dataset  
    - Model Prediksi Obesitas  
    - Tentang Aplikasi  
    """)

# ============ KALKULATOR BMI ==============
elif menu == "📊 Kalkulator BMI":
    st.title("📊 Kalkulator BMI")

    st.sidebar.header("Input Data BMI")
    weight = st.sidebar.number_input("Berat (kg)", 20.0, 300.0, 70.0)
    height_cm = st.sidebar.number_input("Tinggi (cm)", 100.0, 250.0, 170.0)
    age = st.sidebar.number_input("Usia (tahun)", 1, 120, 30)
    sex = st.sidebar.selectbox("Jenis Kelamin", ("Laki-laki", "Perempuan"))
    waist = st.sidebar.number_input("Lingkar Pinggang (cm)", 40.0, 200.0, 85.0)

    bmi = calculate_bmi(weight, height_cm)

    col1, col2 = st.columns(2)

    with col1:
        st.header("Hasil Kalkulator")
        st.metric("BMI", f"{bmi:.1f}" if bmi else "—")

        if bmi:
            st.write("Kategori:", bmi_category(bmi))
            st.write("Risiko lingkar pinggang:", waist_risk(sex, waist))

            st.subheader("Saran Singkat")
            cat = bmi_category(bmi)

            if cat == "Normal":
                st.write("Pertahankan pola makan sehat dan aktivitas fisik teratur.")
            elif cat == "Kurus (Underweight)":
                st.write("Pertimbangkan peningkatan asupan kalori sehat dan konsultasi ahli gizi.")
            elif cat == "Gemuk (Overweight)":
                st.write("Turunkan berat badan bertahap melalui pengaturan porsi dan olahraga.")
            else:
                st.write("Obesitas meningkatkan risiko penyakit kronis — konsultasikan dengan tenaga kesehatan.")

    with col2:
        st.header("Ringkasan Kesehatan")
        st.write(f"Usia: {age} tahun")
        st.write(f"Jenis kelamin: {sex}")
        st.write(f"Lingkar pinggang: {waist} cm")

# ========== ANALISIS DATASET ===========
elif menu == "📈 Analisis Data":
    st.title("📈 Analisis Dataset")

    uploaded = st.file_uploader("Unggah CSV", type=["csv"])

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.success("CSV berhasil dimuat.")
        except Exception as e:
            st.error("Gagal membaca CSV: " + str(e))
            df = None
    else:
        df = generate_synthetic_dataset()
        st.info("Menggunakan dataset sintetis untuk contoh.")

    if df is not None:
        if not all(col in df.columns for col in ["weight_kg", "height_cm"]):
            st.warning("CSV wajib memiliki kolom weight_kg dan height_cm.")
        else:
            df["bmi"] = df["weight_kg"] / ((df["height_cm"] / 100) ** 2)
            df["category"] = df["bmi"].apply(bmi_category)

            st.subheader("Statistik Singkat")
            st.write(df.describe())

            st.pyplot(plot_bmi_histogram(df))

            st.subheader("Jumlah per Kategori BMI")
            st.bar_chart(df["category"].value_counts())

# ========== MODEL PREDIKSI ===========
elif menu == "🧩 Model Prediksi":
    st.title("🧩 Model Prediksi Obesitas")

    st.markdown("""
    ### 🧠 Tentang Model
    Model Logistic Regression digunakan untuk:
    - Mengklasifikasikan obesitas  
    - Menghitung probabilitas risiko  
    - Menggunakan fitur: berat, tinggi, usia  
    """)

    st.markdown("---")
    st.markdown("### 📥 Cara Menggunakan Model")
    st.markdown("""
    1. Unggah dataset CSV dengan kolom:
       - weight_kg  
       - height_cm  
       - age  
    2. Model akan otomatis menghitung BMI dan melatih model.  
    3. Setelah itu, kamu bisa mencoba prediksi individu.  
    """)

    st.markdown("---")

    uploaded = st.file_uploader("Unggah CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        df["bmi"] = df["weight_kg"] / ((df["height_cm"] / 100) ** 2)
    else:
        st.info("Unggah dataset untuk melatih model.")
        df = None

    if df is not None:
        clf, scaler, score, err = train_obesity_model(df)

        if err:
            st.warning(err)
        else:
            st.subheader("✅ Model Berhasil Dilatih")
            st.write(f"Akurasi model: {score:.2f}")

            # Koefisien
            st.subheader("📊 Pengaruh Fitur Prediksi")
            coef = clf.coef_[0]
            features = ["Berat (kg)", "Tinggi (cm)", "Usia"]

            coef_df = pd.DataFrame({
                "Fitur": features,
                "Koefisien": coef
            })

            st.bar_chart(coef_df.set_index("Fitur"))

            st.markdown("---")
            st.subheader("🔍 Coba Prediksi Individu")

            w_in = st.number_input("Berat (kg)", 20.0, 300.0, 70.0)
            h_in = st.number_input("Tinggi (cm)", 100.0, 250.0, 170.0)
            age_in = st.number_input("Usia", 1, 120, 30)

            x = np.array([[w_in, h_in, age_in]])
            x_s = scaler.transform(x)

            prob = clf.predict_proba(x_s)[0, 1]
            pred = clf.predict(x_s)[0]

            st.write(f"Peluang obesitas: {prob:.2f}")
            st.write("Kategori model:", "Obesitas" if pred == 1 else "Tidak Obesitas")

            st.markdown("### 📝 Interpretasi Hasil")
            if pred == 1:
                st.success("✅ Individu diprediksi berisiko obesitas.")
            else:
                st.info("ℹ Individu diprediksi tidak obesitas.")

            st.markdown("---")
            st.subheader("📌 Ringkasan Model")
            st.write(f"""
            - Akurasi model: {score:.2f}  
            - Jumlah data latih: {len(df)} baris  
            - Fitur: berat, tinggi, usia  
            - Target: obesitas (BMI ≥ 30)  
            """)

# ========== TENTANG APLIKASI =========
elif menu == "💻 Tentang Aplikasi":
    st.title("💻 Tentang Aplikasi")
    st.write("""
    Aplikasi ini dibuat untuk edukasi mengenai:
    - Perhitungan BMI  
    - Risiko obesitas  
    - Analisis dataset kesehatan  
    - Model prediksi sederhana menggunakan Logistic Regression  

    Dibangun menggunakan:
    - Python  
    - Streamlit  
    - Scikit-learn  
    - Matplotlib  
    """)