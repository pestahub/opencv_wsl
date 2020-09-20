#!/usr/bin/env python
import numpy as np
import cv2

img1 = np.zeros((128, 128, 3), np.uint8)
img1 = cv2.rectangle(img1, (5,0), (100, 50), (255, 255, 255), -1) 
img2 = cv2.imread('mask.png')

bitAnd = cv2.bitwise_and(img2, img1)
bitOr = cv2.bitwise_or(img2, img1)
bitXor = cv2.bitwise_xor(img2, img1)
bitNot1 = cv2.bitwise_not(img1)
bitNot2 = cv2.bitwise_not(img2)
cv2.imshow('image1', img1)
cv2.imshow('image2', img2)
cv2.imshow('and', bitAnd)
cv2.imshow('or', bitOr)
cv2.imshow('xor', bitXor)
cv2.imshow('not1', bitNot1)
cv2.imshow('not2', bitNot2)
cv2.waitKey(3000)
cv2.destroyAllWindows()
