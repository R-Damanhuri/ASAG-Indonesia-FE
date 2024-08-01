from functions import *
import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

indosbert_path = "denaya/indoSBERT-large"

st.set_page_config(page_title="Indonesian ASAG",
                   layout="wide",
                   page_icon="📝")

st.title("Indonesian ASAG")

tab1, tab2 = st.tabs(["Tentang", "Penilaian"])

with tab1:
  st.subheader("Selamat Datang!")
  st.markdown(
    """
    Indonesian ASAG merupakan sistem penilaian otomatis jawaban singkat berbahasa Indonesia
    menggunakan teknologi _natural language processing_ (NLP) dan _machine learning_. Sistem ini dapat membantu Anda
    untuk menilai ujian jawaban singkat dengan topik apa pun secara otomatis.
    """
  )

  st.subheader("Cara Penggunaan")
  st.markdown(
    """
    1. Persiapkan file Excel dengan kolom NIM, Soal, dan Jawaban.
    2. Pilih _tab_ Penilaian pada _navigation bar_.
    3. Unggah file Excel.
    4. Nilai sampel data yang ditampilkan.
    5. Klik tombol Mulai Penilaian untuk memulai proses penilaian otomatis.
    6. Klik tombol Unduh CSV untuk mengunduh hasil penilaian otomatis dalam format CSV.
    """
  )

with tab2:
  st.subheader("Unggah File Excel")
  st.write("Pastikan terdapat kolom **NIM**, **Soal**, dan **Jawaban**.")
  uploaded_file = st.file_uploader("Unggah File", label_visibility="collapsed", type=["xlsx"])

  if "data_labeled" not in st.session_state:
    st.session_state.data_labeled = 0

  if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    with st.spinner("Sedang melakukan prapemrosesan data..."):
      df_preprocced = preprocess(df)

      q_vec, a_vec = sentence_embedding(indosbert_path, df_preprocced)
      
      df_preprocced['Soal_Embed'] = q_vec.tolist()
      df_preprocced['Jawaban_Embed'] = a_vec.tolist()

      df_selected, df_test = split(df_preprocced)
      df_selected['Nilai'] = [0.0] * len(df_selected)
      st.success("Prapemrosesan data selesai!")
    
    @st.fragment
    def labeling():
      st.divider()
      st.subheader("Nilai Sampel Data")
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
    labeling()

    @st.fragment
    def grading():
      if st.button("Mulai Penilaian"):
        with st.spinner("Sedang melakukan penilaian..."):
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
          model.fit(x_train, y_train, validation_data = [x_val,y_val], epochs=100, batch_size=32,callbacks=[stop_early])

          y_predict = model.predict(x_test)
          df_test['Nilai'] = y_predict

          df_result = pd.concat([df_test[['Soal','Jawaban','Nilai']], st.session_state.data_labeled[['Soal','Jawaban','Nilai']]],axis=0)
          df_result['Nilai'] = df_result['Nilai'].astype(float).round(2)
          
          @st.cache_data
          def convert_df(df):
            return df.to_csv().encode("utf-8")
          
          csv_result = convert_df(df_result)
          
          st.success("Penilaian selesai!")

          st.download_button(
            label="Unduh CSV",
            data= csv_result,
            file_name="hasil_penilaian.csv",
            mime="text/csv"
          )
    grading()