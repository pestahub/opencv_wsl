import cv2

im = cv2.imread('img.jpg')

cv2.imshow('image', im)

while(1):
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
        break