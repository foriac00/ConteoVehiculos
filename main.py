import cv2
import time
import numpy as np

# Create our body classifier
car_classifier = cv2.CascadeClassifier('haarcascade_car.xml')

img = cv2.imread('input_examples/im1.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cars = car_classifier.detectMultiScale(gray, 1.01, 4)

for (x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.imshow('Cars', img)

cv2.waitKey(0)
cv2.destroyAllWindows()