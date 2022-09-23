import streamlit as st
import package


st.markdown("# Dự đoán một bênh nhân")
st.sidebar.markdown("# Dự đoán một bênh nhân")

image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
if image_file is not None:
    if st.button('PREDICT'):
        res = package.predict.getPredictImg(image_file)
        st.write(res)