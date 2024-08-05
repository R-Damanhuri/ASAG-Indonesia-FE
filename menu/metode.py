import streamlit as st
import pandas as pd

def show():
    st.subheader("Alur Penilaian", divider='red')
    st.image("./images/alur_penilaian.png", width=600, caption="Alur Penilaian Indonesian ASAG")

    st.subheader("Praproses", divider='red')
    st.markdown(
        """
        - Pemilihan kolom Soal dan Jawaban
        - Penyeragaman huruf kecil
        - Penghapusan karakter selain huruf dan angka
        - Penghapusan spasi berlebih
        """
    )

    st.subheader("Representasi Teks", divider='red')
    st.markdown(
        """
        - Teknik: *Fine-tuning Sentence Embedding Model*
        - Model: [IndoSBERT](https://huggingface.co/denaya/indoSBERT-large)
        - *Fine-tune*: [SimCSE](https://www.sbert.net/examples/unsupervised_learning/SimCSE/README.html)
        """
    )

    st.subheader("Pembagian Data", divider='red')
    st.image("./images/pembagian_data.png", width=550, caption="Alur Pembagian Data Indonesian ASAG")

    st.subheader("Pelatihan Model", divider='red')
    data = {
            'Parameter': [
                'Model','BiLSTM units', 'Dropout rate', 'Loss Function', 'Learning Rate', 'Optimizer', 'Epoch', 'Callback', 'Arsitektur'
            ],
            'Nilai': [
                'Matching Matrix BiLSTM','128', '0.4', 'Mean Squared Error (MSE)', '1e-3', 'Adam', '100', "EarlyStopping(monitor='val_SMAPE', mode='min', baseline=2, start_from_epoch=85)", "./images/arsitektur_model.png"
            ]
        }

    df = pd.DataFrame(data)
    st.dataframe(
        df,
        column_config={
            "Arsitektur": st.column_config.ImageColumn(
                "Arsitektur"
            )
        },
        hide_index=True,
    )