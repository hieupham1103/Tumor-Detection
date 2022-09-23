from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from glob import glob
import os
from PIL import Image


modelLung = keras.models.load_model('./package/Model/LungAndBrain.h5')
modelLung.summary()
modelLung.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


def load_image(img):
    img = Image.open(img)
    return img


def getPredictFile(path):
    img1 = cv2.imread(path)
    img1 = cv2.resize(img1, (224, 224))
    img1 = np.reshape(img1, [1, 224, 224, 3])

    res1 = modelLung.predict(img1)

    return res1[0]


def getPredictImg(img):
    img = load_img(img,target_size=(64,64))  
    img = img_to_array(img)
    img = np.array(img) 
    # return img
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, [1, 224, 224, 3])
    res = modelLung.predict(img)
    res = res[0]
    typeRes = "unknown"
    if (round(res[0]) == 1):
        typeRes = "Phổi bình thường"
    if (round(res[1]) == 1):
        typeRes = "Phổi bình bị viêm hoặc có khối u"
    if (round(res[2]) == 1):
        typeRes = "Não bình thường"
    if (round(res[3]) == 1):
        typeRes = "Não có khối u" 
    return typeRes
