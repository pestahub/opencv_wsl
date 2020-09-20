#!/usr/bin/env python
import numpy as np
import cv2

img = cv2.imread('messi5.jpg')

print(img.shape)
print(img.size)
print(img.dtype)
b,g,r = cv2.split(img)
t = cv2.merge((b,g,r))



cv2.imshow('image', t)
cv2.waitKey(0)
cv2.destroyAllWindows()