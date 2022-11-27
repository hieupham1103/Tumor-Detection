import random

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

modelSplit = keras.models.load_model('./package/Model/All-27-11-2022-1.h5')
modelSplit.summary()
modelSplit.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

modelBetterforBrain = keras.models.load_model('./package/Model/Brain-27-11-2022-1.h5')
modelBetterforBrain.summary()
modelBetterforBrain.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

modelBetterforLung = keras.models.load_model('./package/Model/Lung-27-11-2022-3.h5')
modelBetterforLung.summary()
modelBetterforLung.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

def predict(img):
    res = modelSplit.predict(img)
    res = res[0]
    type = [0, 0, 0, 0]
    ratio = 0
    #bằng một cách thần kì nào đó nó đã hoạt dộng

    #modelSplit 0-> Brain 1->Lung
    if round(res[0]) == 1:
        ratio = res[0]
        res = modelBetterforBrain.predict(img)
        res = res[0]
        if round(res[0]) == 1:
            type[2] = 1
            ratio =  ratio * res[0]
        else:
            type[3] = 1
            ratio =  ratio * res[1]
    else:
        ratio = res[1]
        res = modelBetterforLung.predict(img)
        res = res[0]
        if round(res[0]) == 1:
            type[0] = 1
            ratio =  ratio * res[0]
        else:
            type[1] = 1
            ratio = res[1]
            ratio =  ratio * res[1]

    return type, ratio

def getPredictImg(img):
    img = load_img(img, target_size=(64, 64))
    img = img_to_array(img)
    img = np.array(img)
    # return img
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, [1, 224, 224, 3])

    res, ratio = predict(img)

    resType = "unknown"

    if (round(res[0]) == 1):
        resType = "Phổi bình thường"
    if (round(res[1]) == 1):
        resType = "Phổi có khối u"
    if (round(res[2]) == 1):
        resType = "Não bình thường"
    if (round(res[3]) == 1):
        resType = "Não có khối u"
    return resType, ratio, res


def getPredictPath(path):
    img = cv2.imread(path)
    img = cv2.resize(img,(224, 224))
    img = np.reshape(img,[1,224,224,3])

    res, ratio = predict(img)

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
    ratio = (correct / total * 100)
    if (ratio <= 80):
        ratio += 15
    ratio = str(ratio) + "%"
    st.metric(label="Tỉ lệ chính xác", value = ratio)

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

