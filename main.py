import streamlit as st
from streamlit_option_menu import option_menu
from menu import beranda, metode, penilaian

st.set_page_config(page_title="Indonesian ASAG", layout="wide", page_icon="📝")

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

with st.sidebar:
    selected = option_menu(
        menu_title="📝 Indonesian ASAG",
        options=["Beranda", "Metode", "Penilaian"],
        icons=["house", "gear" , "pencil-square"],
        menu_icon="cast",
        default_index=0,
        styles={
          "menu-icon":{"display":"none"},
          "menu-title":{
            "font-size":"1.25rem",
            "font-weight":"bold"
            }
        }
    )
    # Footer
    st.markdown("---")
    st.write("© 2024 | Made with love by [R. Damanhuri](https://github.com/r-Damanhuri/)")

if selected == "Beranda":
    beranda.show()
elif selected == "Metode":
    metode.show()
elif selected == "Penilaian":
    penilaian.show()