import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons
import cv2 as cv


img = cv.imread('top_3.png', -1)
# cv.imshow('image', img)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)


fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(left=0.25, bottom=0.25)
min0 = 0
max0 = 25000

im = max0 * 1
im1 = ax.imshow(img)
fig.colorbar(im1)

axcolor = 'lightgoldenrodyellow'
axmin = fig.add_axes([0.25, 0.1, 0.65, 0.03])
axmax  = fig.add_axes([0.25, 0.15, 0.65, 0.03])

smin = Slider(axmin, 'Min', 0, 30000, valinit=min0)
smax = Slider(axmax, 'Max', 0, 30000, valinit=max0)

def update(val):
    im1.set_clim([smin.val,smax.val])
    fig.canvas.draw()
smin.on_changed(update)
smax.on_changed(update)

plt.show()