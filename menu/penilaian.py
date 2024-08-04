import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from functions import *
import matplotlib.pyplot as plt

indosbert_path = "denaya/indoSBERT-large"

def show():
    st.subheader("Unggah _File_", divider="red")
    st.write("Pastikan _file_ memiliki format **Excel (.xlsx)** dan memiliki kolom **NIM**, **Soal**, dan **Jawaban**.")
    uploaded_file = st.file_uploader("Unggah File", label_visibility="collapsed", type=["xlsx"])

    if "data_labeled" not in st.session_state:
        st.session_state.data_labeled = 0

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)

        with st.spinner("Prapemrosesan data sedang dilakukan ...."):
            df_preprocced = preprocess(df)
            q_vec, a_vec = sentence_embedding(indosbert_path, df_preprocced)
            df_preprocced['Soal_Embed'] = q_vec.tolist()
            df_preprocced['Jawaban_Embed'] = a_vec.tolist()
            df_selected, df_test = split(df_preprocced)
            df_selected['Nilai'] = [0.0] * len(df_selected)
            st.toast("Prapemrosesan data selesai!", icon="✅")
        labeling(df_selected)
        grading(df_test)

@st.fragment
def labeling(df_selected):
    st.subheader("Nilai Sampel Data", divider="red")
    st.write("Isikan nilai beberapa sampel data untuk pelatihan model _machine learning_.")
    df_labeled = st.data_editor(
        df_selected,
        disabled=("Soal", "Jawaban"),
        column_config={
            "Soal_Embed": None,
            "Jawaban_Embed": None,
            "Cosine": None,
            "Nilai": st.column_config.NumberColumn(
                "Nilai",
                min_value=0,
                max_value=10,
                step = 0.01,
                format="%0.2f"
            )
        },
        hide_index=True,
    )
    st.session_state.data_labeled = df_labeled

@st.fragment
def grading(df_test):
    if st.button("Mulai Penilaian"):
        success_placeholder = st.empty()
        with st.spinner("Penilaian sedang dilakukan ...."):
            df_train, df_val = train_test_split(st.session_state.data_labeled, random_state = 42, train_size=0.7)
            x_train = df_train[['Soal_Embed','Jawaban_Embed']]
            y_train = df_train['Nilai']

            x_val = df_val[['Soal_Embed','Jawaban_Embed']]
            y_val = df_val['Nilai']

            x_test = df_test[['Soal_Embed','Jawaban_Embed']]

            x_train = match_matrix(x_train)
            x_val = match_matrix(x_val)
            x_test = match_matrix(x_test)

            model = model_builder()
            stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_SMAPE', mode='min', baseline=2, start_from_epoch=85)
            history = model.fit(x_train, y_train, validation_data = [x_val,y_val], epochs=100, batch_size=32,callbacks=[stop_early])

            y_predict = model.predict(x_test)
            df_test['Nilai'] = y_predict

            df_result = pd.concat([df_test[['Soal','Jawaban','Nilai']], st.session_state.data_labeled[['Soal','Jawaban','Nilai']]],axis=0)
            df_result['Nilai'] = df_result['Nilai'].astype(float).round(2)
            
            csv_result = convert_df(df_result)
            
            st.toast("Penilaian selesai!", icon="✅")

            st.download_button(
                label="Unduh CSV",
                data= csv_result,
                file_name="hasil_penilaian.csv",
                mime="text/csv"
            )

            # Panggil fungsi baru untuk menampilkan kualitas penilaian
            quality_controling(history, y_predict)

@st.fragment
def quality_controling(history, y_predict):
    st.subheader("Kualitas Penilaian", divider="red")

    # Tampilkan metrik menggunakan st.metric
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Loss", f"{history.history['val_loss'][-1]:.2f}")
    col2.metric("RMSE", f"{history.history['val_root_mean_squared_error'][-1]:.2f}")
    col3.metric("MAE", f"{history.history['val_mean_absolute_error'][-1]:.2f}")
    col4.metric("SMAPE", f"{history.history['val_SMAPE'][-1]:.2f}")

    # Buat dua kolom: satu untuk histogram, satu untuk metrik min, median, max
    col_hist, col_stats = st.columns([0.75, 0.25])

    with col_hist:
        # Buat histogram dari y_predict dengan ukuran yang lebih kecil
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(y_predict, bins=20, edgecolor='black')
        ax.set_title('Histogram of Predicted Values')
        ax.set_xlabel('Predicted Value')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

    with col_stats:
        # Tampilkan nilai min, median, max menggunakan st.metric
        st.metric("Nilai Minimum", f"{np.min(y_predict):.2f}")
        st.metric("Nilai Median", f"{np.median(y_predict):.2f}")
        st.metric("Nilai Maksimum", f"{np.max(y_predict):.2f}")

@st.cache_data
def convert_df(df):
    return df.to_csv().encode("utf-8")