#app/ui.py
import streamlit as st

def header():
    st.title("Receipt OCR → CSV")
    st.caption("Fast pipeline: preprocess → OCR → parse → CSV")

def footer():
    st.markdown("Made with PaddleOCR + OpenCV + Streamlit")




