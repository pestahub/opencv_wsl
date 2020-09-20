#!/usr/bin/env python
import numpy as np
import cv2

#img = cv2.imread('lena.jpg', 1)
img = np.zeros([512, 512, 3], np.uint8)


img = cv2.line(img, (0,0), (255,255), (255, 0, 255), 10) #line
img = cv2.arrowedLine(img, (0,0), (255, 123), (255, 0, 0), 10) #arrow
img = cv2.rectangle(img, (255,0), (500, 255), (0, 0, 255), -1) #rectangle if -1 full inside elif 10 pt frame
img = cv2.circle(img, (125, 255), 50, (255, 255, 0), 10) #circle if -1 full inside elif 10 pt frame
font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.putText(img, 'HelloWorld', (5, 400), font, 2, (0, 0, 0), 4, cv2.LINE_AA)

cv2.imshow('image', img)

cv2.waitKey(0)

cv2.destroyAllWindows()