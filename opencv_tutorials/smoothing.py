import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt 

img = cv.imread('lena.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

kernal = np.ones((5, 5), np.float32)/25
dst = cv.filter2D(img, -1, kernal)
blur = cv.blur(img, (5, 5))
gblur = cv.GaussianBlur(img, (5, 5), 0)
median = cv.medianBlur(img, 5)
bilateralFilter = cv.bilateralFilter(img, 9, 75, 75)


titles = ['image', '2D Convolusion', 'blur', 'gblur', 'median', 'bilateralFilter']
images = [img, dst, blur, gblur, median, bilateralFilter]

for i in range(6):
    plt.subplot(2,3,i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])


#cv.imshow(0)
#cv.destroyAllWindows()
plt.show()