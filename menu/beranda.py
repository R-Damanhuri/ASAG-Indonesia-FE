import streamlit as st

def show():
    import streamlit as st

    st.markdown("""
    <style>
    .header {
        display: flex;
        align-items: center;
        padding: 20px;
        background-color: #f0f2f6;
        margin-top: -15px;
        margin-bottom: 20px;
        border-radius: 10px;
    }
    .title-container {
        display: flex;
        flex-direction: column;
    }
    .title-container h1 {
        font-size: 42px;
        margin: 0;
        line-height: 0.5;
        color: #333;
    }
    .title-container p {
        font-size: 14px;
        margin: 0;
        padding-left: 12px;
    }
    </style>
    
    <div class="header">
        <div class="title-container">
            <h1>📝 Indonesian ASAG</h1>
            <p>Sistem Penilaian Otomatis Jawaban Singkat Bahasa Indonesia</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Ringkasan", divider='red')
    st.markdown(
        """
        Indonesian ASAG merupakan sistem penilaian otomatis jawaban singkat berbahasa Indonesia
        menggunakan teknologi _natural language processing_ (NLP) dan _machine learning_. Sistem ini dapat membantu Anda
        untuk menilai ujian jawaban singkat dengan topik apa pun secara otomatis.
        """
    )
    
    st.subheader("Petunjuk Penggunaan", divider='red')
    st.markdown(
        """
        1. Persiapkan _file_ Excel dengan kolom NIM, Soal, dan Jawaban.
        2. Pilih halaman Penilaian pada _sidebar_.
        3. Unggah _file_ Excel.
        4. Nilai sampel data yang ditampilkan.
        5. Klik tombol Mulai Penilaian untuk memulai proses penilaian otomatis.
        6. Klik tombol Unduh CSV untuk mengunduh hasil penilaian otomatis dalam format CSV.
        """
    )