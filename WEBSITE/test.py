import cv2

img = cv2.imread('./package/Datasets/4_Brain Tumor yes/Y23.jpg')

# print(type(img))
# if img.empty():
#         print("Empty")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
detector = cv2.CascadeClassifier('./package/cascade-2.xml')
rect = detector.detectMultiScale(gray_img, 1.1, 9)
for (x, y, w, h) in rect:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

cv2.imshow('Detected faces', img)
cv2.waitKey(0)

