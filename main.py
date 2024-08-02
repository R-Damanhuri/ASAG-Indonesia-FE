import streamlit as st
from streamlit_option_menu import option_menu
from menu import beranda, penilaian

st.set_page_config(page_title="Indonesian ASAG", layout="wide", page_icon="📝")

with st.sidebar:
    selected = option_menu(
        menu_title="📝 Indonesian ASAG",
        options=["Beranda", "Penilaian"],
        icons=["house", "pencil-square"],
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
    st.write("© 2024 Indonesian ASAG. Hubungi kami di support@indonesian-asag.com")

if selected == "Beranda":
    beranda.show()
elif selected == "Penilaian":
    penilaian.show()