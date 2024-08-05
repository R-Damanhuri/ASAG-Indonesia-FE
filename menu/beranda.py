import streamlit as st

def show():
    st.subheader("Ringkasan", divider='red')
    st.markdown(
        """
        Indonesian ASAG merupakan sistem penilaian otomatis jawaban singkat berbahasa Indonesia
        menggunakan teknologi _natural language processing_ (NLP) dan _machine learning_. Sistem ini dapat membantu Anda
        untuk menilai ujian jawaban singkat dengan topik apa pun secara otomatis. Metode penilaian dapat dipelajari pada menu Metode.
        """
    )
    
    st.subheader("Petunjuk Penggunaan", divider='red')
    st.markdown(
        """
        1. Persiapkan _file_ Excel dengan kolom NIM, Soal, dan Jawaban.
        2. Pilih menu Penilaian pada _sidebar_.
        3. Unggah _file_ Excel.
        4. Nilai sampel data yang ditampilkan.
        5. Klik tombol Mulai Penilaian untuk memulai proses penilaian otomatis.
        6. Klik tombol Unduh CSV untuk mengunduh hasil penilaian otomatis dalam format CSV.
        """
    )