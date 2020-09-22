# import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider


img = cv.imread('top_3.png', -1)
# cv.imshow('image', img)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)


# axfreq = plt.axes([0.1,0.1,0.2,0.05])
# sfreq = Slider(axfreq, 'Freq', 0, 30)

titles = ['image', 'slider']
images = [img]
plt.subplot(1,2,1), plt.imshow(images[0], 'gray')
plt.title(titles[0])
plt.xticks([]), plt.yticks([])

plt.subplot(1,2,2), plt.imshow(images[0], 'gray')
plt.title(titles[0])
plt.xticks([]), plt.yticks([])
# # ax = plt.subplot()
# # plt.subplots_adjust(left=0.25, bottom=0.25)
axfreq = plt.axes([0.1,0.1,0.2,0.05])
sfreq = Slider(axfreq, 'Freq', 0, 30)
plt.title(titles[1])
# axfreq = plt.axes([0.1,0.1,0.2,0.05])
# sfreq = Slider(axfreq, 'Freq', 0, 30)
# # samp = Slider(axamp,c 'Amp', 0.1, 10.0, valinit=a0)



# plt.imshow(img)
plt.show()


