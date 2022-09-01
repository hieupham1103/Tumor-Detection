from itertools import count
import numpy as np
from tensorflow import keras
import cv2
from glob import glob
import os

modelLung = keras.models.load_model('lungcancer.h5')
modelLung.summary()
modelLung.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

modelLB = keras.models.load_model('x.h5')
modelLB.summary()
modelLB.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)



def lungCheck(path):
  img1 = cv2.imread(path)
  img1 = cv2.resize(img1,(224, 224))
  img1 = np.reshape(img1,[1,224,224,3])

  res1 = modelLung.predict(img1)
  
  return res1[0]

def lungBrainCheck(path):
  img1 = cv2.imread(path)
  img1 = cv2.resize(img1,(224, 224))
  img1 = np.reshape(img1,[1,224,224,3])

  res1 = modelLB.predict(img1)
  
  return res1[0]


# listAns = []

# for x in os.listdir("./Datasets/train/"):
#   countNone = 0
#   countHad = 0  
#   url_list = glob(f"./Datasets/test/{x}/*.jpeg")
#   # print(url_list)
#   for file in url_list:
#     res = lungCheck(file);    
#     a = np.around(res[0])
#     b = np.around(res[1])
#     # print(a, b)
#     if a == 1:
#       countNone += 1
#     else:
#       countHad += 1
      
#   listAns.append(f"{countNone} {countHad}")
  
# for ans in listAns:
#   print(ans)
  
# t = lungBrainCheck("D:/SCIENTIFIC RESEARCH/Lung Cancer/Datasets/test/4_Brain Tumor yes/Y33.jpg")

# print(t)