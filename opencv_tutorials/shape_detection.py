import cv2 as cv
import numpy as np 

img = cv.imread('pic3.png')
grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, thrash = cv.threshold(grey, 240, 255, cv.THRESH_BINARY)
contours, _ = cv.findContours(thrash, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

for contour in contours:
    approx = cv.approxPolyDP(contour, 0.1*cv.arcLength(contour, True), True, )
    cv.drawContours(img, [approx], 0, (255, 0, 0), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1]
    if len(approx) == 3:
        cv.putText(img, 'triangle', (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    elif len(approx) == 4:
        x, y, w, h = cv.boundingRect(approx)
        aspectRatio = float(w)/h
        if aspectRatio > 0.95 and aspectRatio < 1.05:
            cv.putText(img, 'scqare', (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        else:
            
            cv.putText(img, 'rect', (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
         

cv.imshow('img', img)

cv.waitKey(0)
cv.destroyAllWindows()