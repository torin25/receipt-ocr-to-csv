# Make repo root importable when running app/app.py
import sys, os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)



import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.receipt_ocr_to_csv.preprocessing import preprocess_image
from src.receipt_ocr_to_csv.ocr import get_ocr, run_ocr
from src.receipt_ocr_to_csv.parsing import parse_receipt
from src.receipt_ocr_to_csv.export import df_to_csv_bytes

st.set_page_config(page_title="Receipt OCR → CSV", layout="wide")

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def load_sample():
    candidates = list(DATA_DIR.glob("*.jpg")) + list(DATA_DIR.glob("*.png"))
    return candidates[0] if candidates else None

@st.cache_resource
def ocr_model():
    return get_ocr()

@st.cache_data(show_spinner=False)
def pipeline(image_bytes: bytes):
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    proc_bgr, disp_rgb = preprocess_image(pil)
    lines = run_ocr(ocr_model(), proc_bgr)
    meta, items_df = parse_receipt(lines)
    tidy = assemble_dataframe(meta, items_df)
    return disp_rgb, lines, meta, tidy

def assemble_dataframe(meta, items_df):
    # Ensure minimal tidy schema
    df = items_df.copy()
    df["merchant"] = meta.get("merchant")
    df["date"] = meta.get("date")
    df["currency"] = meta.get("currency")
    # Order columns for export
    cols = ["merchant", "date", "item", "qty", "unit_price", "line_total", "currency"]
    for c in cols:
        if c not in df.columns: df[c] = np.nan
    return df[cols]

def main():
    st.title("Receipt OCR → CSV")
    st.caption("Upload a receipt image (JPG/PNG). The app preprocesses, runs OCR, parses items, and exports CSV.")

    c1, c2 = st.columns([1,1])
    with c1:
        uploaded = st.file_uploader("Upload receipt (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if st.button("Use sample") and uploaded is None:
            sample = load_sample()
            if sample:
                uploaded = st.file_uploader("Re-upload to override", type=[], disabled=True)
                image_bytes = sample.read_bytes()
            else:
                st.warning("Place an example image in data/ to use sample.")
                image_bytes = None
        else:
            image_bytes = uploaded.read() if uploaded else None

    if image_bytes:
        disp_rgb, lines, meta, tidy = pipeline(image_bytes)
        with c1:
            st.subheader("Preview")
            st.image(disp_rgb, channels="RGB", use_column_width=True)
        with c2:
            st.subheader("Detected text (top 20)")
            st.write(pd.DataFrame(lines)[:20])

        st.subheader("Parsed result")
        edited = st.data_editor(tidy, use_container_width=True, num_rows="dynamic")
        csv_bytes = df_to_csv_bytes(edited)
        st.download_button("Download CSV", data=csv_bytes, file_name="receipt.csv", mime="text/csv")
    else:
        st.info("Upload a receipt to begin.")

if __name__ == "__main__":
    main()
