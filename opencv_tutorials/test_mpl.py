import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons
import cv2 as cv


img = cv.imread('top_3.png', -1)
# cv.imshow('image', img)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)


fig = plt.figure()
fig.subplots_adjust(bottom=0.5)
ax = fig.add_subplot(1,4,1)
plt.title(1)
plt.xticks([]), plt.yticks([])
bx = fig.add_subplot(1,4,2)
plt.title(2)
plt.xticks([]), plt.yticks([])
cx = fig.add_subplot(1,4,3)
plt.title(3)
plt.xticks([]), plt.yticks([])
dx = fig.add_subplot(1,4,4)
plt.title(4)
plt.xticks([]), plt.yticks([])


# fig.subplots_adjust(bottom=0.5)


ax.imshow(img)
bx.imshow(img)
cx.imshow(img)
dx.imshow(img)




axmin = plt.axes([0.03,0.1,0.2,0.05])
axmax  = plt.axes([0.35,0.1,0.2,0.05])

smin = Slider(axmin, 'Min', 0, 10, valstep=1)
smax = Slider(axmax, 'Max', 0, 10, valstep=1)

axmin1 = plt.axes([0.1,0.03,0.1,0.03])
axmax1  = plt.axes([0.5,0.05,0.2,0.05])

smin1 = Slider(axmin1, 'Min1', 0, 10, valstep=1)
smax1 = Slider(axmax1, 'Max1', 0, 10, valstep=1)


axmin2 = plt.axes([0.1,0.17,0.2,0.05])
axmax2  = plt.axes([0.5,0.2,0.2,0.05])

smin2 = Slider(axmin2, 'Min2', 0, 10, valstep=1)
smax2 = Slider(axmax2, 'Max2', 0, 10, valstep=1)


def update(val):
    
    img1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ax.imshow(img1)
    
smin.on_changed(update)
smax.on_changed(update)    
smin1.on_changed(update)
smax1.on_changed(update)    
smin2.on_changed(update)
smax2.on_changed(update)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()