from tensorflow import keras
import numpy as np
import cv2
from glob import glob
import os

modelLB = keras.models.load_model('LungAndBrain.h5')
modelLB.summary()
modelLB.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


def lungBrainCheck(path):
  img1 = cv2.imread(path)
  
  print(type(img1))
  
  img1 = cv2.resize(img1,(224, 224))
  img1 = np.reshape(img1,[1,224,224,3])
  
  
  res1 = modelLB.predict(img1)
  
  return res1[0]
  
t = lungBrainCheck("INPUT/true1.jpeg")

# print(t)