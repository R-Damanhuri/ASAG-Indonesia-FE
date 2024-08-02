import streamlit as st

def show():
    st.title("Tentang Indonesian ASAG")
    st.subheader("Cara Penggunaan")
    st.markdown(
        """
        1. Persiapkan file Excel dengan kolom NIM, Soal, dan Jawaban.
        2. Pilih halaman Penilaian pada sidebar.
        3. Unggah file Excel.
        4. Nilai sampel data yang ditampilkan.
        5. Klik tombol Mulai Penilaian untuk memulai proses penilaian otomatis.
        6. Klik tombol Unduh CSV untuk mengunduh hasil penilaian otomatis dalam format CSV.
        """
    )