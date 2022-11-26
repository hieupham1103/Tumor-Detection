from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import pandas
from glob import glob
from PIL import Image
from matplotlib import cm
import streamlit as st


modelLung = keras.models.load_model('./package/Model/old.h5')
modelLung.summary()
modelLung.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


def getPredictImg(img):
    img = load_img(img, target_size=(64, 64))
    img = img_to_array(img)
    img = np.array(img)
    # return img
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, [1, 224, 224, 3])
    res = modelLung.predict(img)
    res = res[0]
    resType = "unknown"
    totalRatio = res[0] + res[1] + res[2] + res[3]
    resRatio = 0

    if (round(res[0]) == 1):
        resType = "Phổi bình thường"
        resRatio = res[0]
    if (round(res[1]) == 1):
        resType = "Phổi có khối u"
        resRatio = res[1]
    if (round(res[2]) == 1):
        resType = "Não bình thường"
        resRatio = res[2]
    if (round(res[3]) == 1):
        resType = "Não có khối u"
        resRatio = res[3]
    return resType, resRatio / totalRatio, res


def getPredictPath(path):
    img1 = cv2.imread(path)
    img1 = cv2.resize(img1,(224, 224))
    img1 = np.reshape(img1,[1,224,224,3])

    res = modelLung.predict(img1)
    res = res[0]
    type = 0
    if (round(res[0]) == 1):
        type = 0
    if (round(res[1]) == 1):
        type = 1
    if (round(res[2]) == 1):
        type = 2
    if (round(res[3]) == 1):
        type = 3
    return type

def runAllDataSet(type, num):
    total = 0
    correct = 0
    
    if type == 0:
        x = "1_LUNG NORMAL"
    if type == 1:
        x = "2_LUNG PNEUMONIA"
    if type == 2:
        x = "3_Brain Tumor no"
    if type == 3:
        x = "4_Brain Tumor yes"
        
    url_list = glob(f"package/Datasets/{x}/*")
    
    df = pandas.DataFrame(columns =  ["Tên file", "Chính xác"])
    countRow = 0
    for file in url_list:
        total += 1
        data = [file, "Sai"]
        if getPredictPath(file) == type:
            correct += 1
            data = [file, "Đúng"]
        df.loc[countRow] = data
        countRow += 1
        if countRow >= num:
            break
    
    st.dataframe(df)
    st.metric(label="Tỉ lệ chính xác", value = str(correct / total * 100) + "%")

def DetectTumor(img):
    img = load_img(img, target_size=(255, 255))
    oimg = img
    img = np.array(img)
    # img = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)

    # print(type(img))
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray_img = (gray_img * 255).astype(np.uint8)
    # #
    detector = cv2.CascadeClassifier('./package/cascade-brain.xml')
    rect = detector.detectMultiScale(img, 1.1, 9)
    # st.write(rect)
    for (x, y, w, h) in rect:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # cv2.imshow('Detected faces', img)
    # img = Image.fromarray(np.uint8(cm.gist_earth(img) * 255))
    # oimg = Image.fromarray((oimg * 255).astype(np.uint8))

    return img

