import cv2 as cv
import numpy as np

drawing = False
img = None
dick = None
# Обработка клика
def click_event(event, x, y, flags, param):
    global ix, iy, mode, drawing, img, dick

    # поднимаем флаг при нажатии на лкм
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True

    # если флаг поднят если мышка в серой зоне и в придела изображения то объединяеи
    # созданный белый квадрат с кусочком на котором курсор и внедряем его в наше изображение
    elif event == cv.EVENT_MOUSEMOVE:
        if (x > 40) & (y > 40) & (x < img.shape[1]) & (y < img.shape[0]):
            if drawing == True:  
                v1 = abs(int(img[y, x, 0]) - int(img[y, x, 1]))
                v2 = abs(int(img[y, x, 1]) - int(img[y, x, 2]))
                v3 = abs(int(img[y, x, 0]) - int(img[y, x, 2]))
                v = v1 + v2 + v3
                if v < 40: 
                    curnel = img[y-10:y+10, x-10:x+10]
                    black_dick = dick[y-10:y+10, x-10:x+10]
                    new = cv.addWeighted(curnel, 0.9, black_dick, 0.1, 0)
                    img[y-10:y+10, x-10:x+10] = new
                    cv.imshow('image', img)
            
    # при отпускание лкм флаг отпускается            
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        
        


img = cv.imread('lot_bil.png', 1)
dick = cv.imread('chlen.jpg', 1)
cv.namedWindow('image')
cv.setMouseCallback('image', click_event)
print(img.shape)
while(1):       
    cv.imshow('image', img)
    k = cv.waitKey(20) & 0xFF
    if  k == 27:
        break

cv.destroyAllWindows()
