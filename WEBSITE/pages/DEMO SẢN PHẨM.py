import streamlit as st
import pandas
import package


st.markdown("# DỰ ĐOÁN")

 
image_files = st.file_uploader("Tải ảnh lên", type=["png","jpg","jpeg"], accept_multiple_files = True)

if image_files is not None:
    if st.button('Kết Quả'):
        countRow = 0
        df = pandas.DataFrame(columns =  ["Tên", "loại bệnh", "Độ chính xác (%)"])
        listIMG = []
        for file in image_files:
            resType, resRatio, res = package.predict.getPredictImg(file)
            df.loc[countRow] =  [file.name, resType, resRatio * 100]
            # st.write(f"{file.name}: {resType} {resRatio}")
            if (resType == "Não có khối u"):
                img = package.predict.DetectTumor(file)
                title = file.name
                listIMG.append((img, title))
                # st.write(package.predict.DetectTumor(image_files))
            countRow += 1

        st.dataframe(df)
        for img, title in listIMG:
            st.write(title)
            st.image(img)