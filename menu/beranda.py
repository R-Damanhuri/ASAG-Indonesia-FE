import streamlit as st

def show():
    st.title("Indonesian ASAG")
    st.subheader("Selamat Datang!")
    st.markdown(
        """
        Indonesian ASAG merupakan sistem penilaian otomatis jawaban singkat berbahasa Indonesia
        menggunakan teknologi _natural language processing_ (NLP) dan _machine learning_. Sistem ini dapat membantu Anda
        untuk menilai ujian jawaban singkat dengan topik apa pun secara otomatis.
        """
    )