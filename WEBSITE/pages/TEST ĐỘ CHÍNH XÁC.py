import streamlit as st
import pandas
import package


st.markdown("# BÁO CÁO QUÁ TRÌNH TRAINNING")
st.markdown("## TIẾN HÀNH CHẠY THỬ TRÊN BỘ DỮ LIỆU")


num = st.slider('Số lượng file của bộ dữ liệu:', 1, 260)

option = st.selectbox(
    'TIẾN HÀNH CHẠY THỬ TRÊN BỘ DỮ LIỆU',
    ('All', 'NÃO BÌNH THƯỜNG', 'NÃO CÓ KHỐI U', 'PHỔI BÌNH THƯỜNG', 'PHỔI CÓ KHỐI U'))

if st.button('CHẠY THỬ'):
    if option == 'NÃO BÌNH THƯỜNG':
        package.predict.runAllDataSet(2, num)
    if option == 'NÃO CÓ KHỐI U':
        package.predict.runAllDataSet(3, num)
    if option == 'PHỔI BÌNH THƯỜNG':
        package.predict.runAllDataSet(0, num)
    if option == 'PHỔI CÓ KHỐI U':
        package.predict.runAllDataSet(1, num)
    if option == 'All':
        st.write('# NÃO BÌNH THƯỜNG')
        package.predict.runAllDataSet(2, num)
        st.write('# NÃO CÓ KHỐI U')
        package.predict.runAllDataSet(3, num)
        st.write('# PHỔI BÌNH THƯỜNG')
        package.predict.runAllDataSet(0, num)
        st.write('# PHỔI CÓ KHỐI U')
        package.predict.runAllDataSet(1, num)



# col1, col2, col3, col4 = st.columns(4)

# with col1:
#     if st.button('NÃO BÌNH THƯỜNG'):
#         package.predict.runAllDataSet(2)

# with col2:
#     if st.button('NÃO CÓ KHỐI U'):
#         package.predict.runAllDataSet(3)
    
# with col3:
#     if st.button('PHỔI BÌNH THƯỜNG'):
#         package.predict.runAllDataSet(0)

# with col4:
#     if st.button('PHỔI CÓ KHỐI U'):
#         package.predict.runAllDataSet(1)