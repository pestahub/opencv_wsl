import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons
import cv2 as cv


img = cv.imread('top_3.png', -1)
# cv.imshow('image', img)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)


fig = plt.figure()
ax = fig.add_subplot(1,2,1)
bx = fig.add_subplot(2,2,1)
cx = fig.add_subplot(3,2,1)
dx = fig.add_subplot(4,2,1)
fig.subplots_adjust(bottom=0.5)
min0 = 0
max0 = 25000

im1 = ax.imshow(img)
im2 = bx.imshow(img)
im2 = cx.imshow(img)
im2 = dx.imshow(img)
fig.colorbar(im1)


axmin = plt.axes([0.1,0.1,0.2,0.05])
axmax  = plt.axes([0.5,0.1,0.2,0.05])

smin = Slider(axmin, 'Min', 0, 30000)
smax = Slider(axmax, 'Max', 0, 30000)

def update(val):
    im1.set_clim([smin.val,smax.val])
    fig.canvas.draw()
smin.on_changed(update)
smax.on_changed(update)

plt.show()