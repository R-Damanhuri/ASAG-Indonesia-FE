import streamlit as st
import pandas as pd

def show():
    st.subheader("Alur Penilaian", divider='red')
    st.image("./images/alur_penilaian.png", width=600, caption="Alur Penilaian")

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
    st.image("./images/pembagian_data.png", width=550, caption="Pembagian Data")

    st.subheader("Pembangunan dan Pelatihan Model", divider='red')
    data = {
            'Parameter': [
                'Model','BiLSTM units', 'Dropout rate', 'Loss Function', 'Learning Rate', 'Optimizer', 'Epoch', 'Callback',
            ],
            'Nilai': [
                'Matching Matrix BiLSTM','128', '0.4', 'Mean Squared Error (MSE)', '1e-3', 'Adam', '100', "EarlyStopping(monitor='val_SMAPE', mode='min', baseline=2, start_from_epoch=85)"
            ]
        }

    df = pd.DataFrame(data)
    show_st_table(df)

    col1, col2 = st.columns([0.4,0.6])
    with col1:
        st.image("./images/matching_matrix.png", width=225, caption="Pembentukan Matching Matrix")
    with col2:
        st.image("./images/arsitektur_model.png", width=425, caption="Arsitektur Model")    

def show_st_table(df, st_col=None, hide_index=True):

    if hide_index:
        hide_table_row_index = """
                <style>
                tbody th {display:none}
                .blank {display:none}
                </style>
                """
        st.markdown(hide_table_row_index, unsafe_allow_html=True)

    if st_col is None:
        st.table(df)
    else:
        st_col.table(df)