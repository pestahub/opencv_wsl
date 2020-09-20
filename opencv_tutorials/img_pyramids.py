import cv2
import numpy as np  
img = cv2.imread('lena.jpg')
lr = cv2.pyrDown(img)
hr = cv2.pyrUp(img)

cv2.imshow('original', img)
cv2.imshow('down', lr)
cv2.imshow('up', hr)
cv2.waitKey(0)
cv2.destroyAllWindows()