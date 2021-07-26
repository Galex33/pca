import utils.display as udisp
import streamlit as st
import views.cours
import views.viz
import views.application

MENU = {
    "Cours théorique" : views.cours,
    "Visualisation" : views.viz,
    "Application" : views.application,
}

def main():
    st.sidebar.title("Explorer la PCA")
    menu_selection = st.sidebar.radio("",list(MENU.keys()))
    menu = MENU[menu_selection]

    with st.spinner(f"Chargement {menu_selection} ..."):
        udisp.render_page(menu)

if __name__ == "__main__":
    main()