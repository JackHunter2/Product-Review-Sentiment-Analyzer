import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Analisis Sentimen Ulasan Produk",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load("sentiment_nb_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_artifacts()
label_map = {0: "Negatif", 1: "Positif"}
examples = {
    "Contoh positif": "This product is amazing and works perfectly, very satisfied",
    "Contoh negatif": "This product is terrible and useless, very disappointed",
}

if "review_input" not in st.session_state:
    st.session_state["review_input"] = ""

st.title("Analisis Sentimen Ulasan Produk")
st.write(
    "Prediksi sentimen ulasan produk menggunakan model Multinomial Naive Bayes."
)
st.markdown(
    "- Tulis ulasan singkat dan jelas\n"
    "- Gunakan bahasa yang konsisten (Inggris)\n"
    "- Untuk banyak ulasan, gunakan unggah CSV di bagian bawah"
)

left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Ulasan tunggal")
    example_cols = st.columns(len(examples))
    for (label, text), col in zip(examples.items(), example_cols):
        if col.button(label, key=f"btn_{label}"):
            st.session_state["review_input"] = text
            st.rerun()

    # Gunakan key agar teks dapat diubah dari tombol contoh
    st.text_area(
        "Masukkan teks ulasan produk:",
        height=150,
        key="review_input",
        placeholder="Ceritakan pengalaman Anda dengan produk...",
        value=st.session_state.get("review_input", ""),
    )

    predict_clicked = st.button("Prediksi Sentimen")

    if predict_clicked:
        review_text = st.session_state.get("review_input", "").strip()
        if not review_text:
            st.warning("Teks ulasan tidak boleh kosong.")
        else:
            with st.spinner("Memproses ulasan..."):
                text_vector = vectorizer.transform([review_text])
                prediction = model.predict(text_vector)[0]
                probability = model.predict_proba(text_vector)[0]

            st.subheader("Hasil Prediksi")
            if prediction == 1:
                st.success(f"Sentimen: {label_map[prediction]}")
            else:
                st.error(f"Sentimen: {label_map[prediction]}")

            prob_neg, prob_pos = float(probability[0]), float(probability[1])
            prob_cols = st.columns(2)
            prob_cols[0].metric("Probabilitas Negatif", f"{prob_neg*100:.1f}%")
            prob_cols[1].metric("Probabilitas Positif", f"{prob_pos*100:.1f}%")

with right_col:
    st.subheader("Tentang model")
    st.info(
        "Model Multinomial Naive Bayes dilatih dari ulasan produk. "
        "Vektorisasi teks menggunakan TF-IDF yang disimpan di `vectorizer.pkl`."
    )
    st.caption("Tips: hindari teks kosong, gunakan kalimat lengkap.")

st.markdown("---")
st.subheader("Prediksi banyak ulasan (CSV)")
uploaded_file = st.file_uploader(
    "Unggah CSV dengan kolom teks ulasan", type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as err:
        st.error(f"Gagal membaca CSV: {err}")
        df = None

    if df is not None and not df.empty:
        text_column = st.selectbox(
            "Pilih kolom yang berisi teks ulasan",
            options=df.columns.tolist(),
        )

        if st.button("Jalankan batch prediksi"):
            with st.spinner("Memproses seluruh ulasan..."):
                texts = df[text_column].fillna("").astype(str)
                vectors = vectorizer.transform(texts)
                predictions = model.predict(vectors)
                probabilities = model.predict_proba(vectors)

                result_df = df.copy()
                result_df["sentimen"] = [label_map[p] for p in predictions]
                result_df["prob_negatif"] = probabilities[:, 0]
                result_df["prob_positif"] = probabilities[:, 1]

            st.success("Batch prediksi selesai.")
            st.dataframe(result_df.head(20))

            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Unduh hasil CSV",
                data=csv_bytes,
                file_name="hasil_sentimen.csv",
                mime="text/csv",
            )
    elif df is not None:
        st.warning("File CSV kosong. Pastikan ada data ulasan.")
