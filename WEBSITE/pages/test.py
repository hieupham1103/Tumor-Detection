import streamlit as st
import pandas
import package


image_files = st.file_uploader("Tải ảnh lên", type=["png", "jpg", "jpeg"])


if image_files is not None:
    img = package.predict.DetectTumor(image_files)
    st.image(img)
