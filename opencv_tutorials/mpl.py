# import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('top_3.png', -1)
# cv.imshow('image', img)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0, valstep=delta_f)
samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)

plt.imshow(img)
plt.show()


cv.waitKey(0)
cv.destroyAllWindows()
