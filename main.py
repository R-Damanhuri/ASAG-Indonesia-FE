import streamlit as st
from streamlit_option_menu import option_menu
from menu import beranda, tentang, penilaian

st.set_page_config(page_title="Indonesian ASAG", layout="wide", page_icon="📝")

with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Beranda", "Tentang", "Penilaian"],
        icons=["house", "info-circle", "pencil-square"],
        menu_icon="cast",
        default_index=0,
    )

if selected == "Beranda":
    beranda.show()
elif selected == "Tentang":
    tentang.show()
elif selected == "Penilaian":
    penilaian.show()