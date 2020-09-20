#!/usr/bin/env python
import numpy as np
import cv2

img = cv2.imread('messi5.jpg')
img2 = cv2.imread('opencv-logo.png')

ball = img[280:340, 330:390]
img[273:333, 100:160] = ball


img = cv2.resize(img, (512, 512))
img2 = cv2.resize(img2, (512, 512))


#dst = cv2.add(img, img2)
dst = cv2.addWeighted(img, .7,  img2, .3, 0)

cv2.imshow('image', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()