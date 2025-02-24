import streamlit as st


if "pdf_pages" in st.session_state:
        zoom = st.slider(
                    "Zoom Level", 100, 1000, 500, 50, key="zoom_pdf")
        for img in st.session_state["pdf_pages"]:
            st.image(img, width=zoom)