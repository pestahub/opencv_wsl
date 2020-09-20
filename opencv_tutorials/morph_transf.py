import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt 

img = cv.imread('smarties.png', cv.IMREAD_GRAYSCALE)
_, mask = cv.threshold(img, 220, 255, cv.THRESH_BINARY_INV)

kernal = np.ones((3,3), np.uint8)

erosion = cv.erode(mask, kernal, iterations=1)
diletion = cv.dilate(erosion, kernal, iterations=1)

opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernal)
closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernal)

titles = ['mask', 'diletion', 'opening', 'closing']
images = [mask, diletion, opening, closing]

for i in range(4):
    plt.subplot(2,2,i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])


#cv.imshow(0)
#cv.destroyAllWindows()
plt.show()