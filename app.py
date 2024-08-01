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
    menggunakan teknologi _natural language processing_ (NLP). Sistem ini dapat membantu Anda
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
  st.write("Unggah File Excel. Pastikan terdapat kolom **NIM**, **Soal**, dan **Jawaban**.")
  uploaded_file = st.file_uploader("Unggah File", label_visibility="collapsed", type=["xlsx"])

  if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df_preprocced = preprocess(df)

    q_vec, a_vec = sentence_embedding(indosbert_path, df_preprocced)
    
    df_preprocced['Soal_Embed'] = q_vec.tolist()
    df_preprocced['Jawaban_Embed'] = a_vec.tolist()

    df_selected, df_test = split(df_preprocced)
    df_selected['Nilai'] = [0.0] * len(df_selected)
    
  if "data_labeled" not in st.session_state:
    st.session_state.data_labeled = 0      
    labeling()        
    grading()