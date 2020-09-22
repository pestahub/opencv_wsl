import cv2 as cv
import numpy as np 
import chess
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def empty(a):
    pass
 

def board_detec(img):
    flag = False
    coor_board = None
    # threshold1 = cv.getTrackbarPos("Threshold1","Parameters")
    # threshold2 = cv.getTrackbarPos("Threshold2","Parameters")
    # kernal_s = cv.getTrackbarPos("kernal","Parameters")
    # blur = cv.getTrackbarPos("blur","Parameters")
    # itdel = cv.getTrackbarPos("itdel","Parameters")
    # itero = cv.getTrackbarPos("itero","Parameters")
    # min_a = cv.getTrackbarPos("min_a","Size")
    # max_a = cv.getTrackbarPos("max_a","Size")
    threshold1 = 216
    threshold2 = 111
    kernal_s = 2
    blur = 1
    itdel = 1
    itero = 1
    min_a = 43
    max_a = 239
    kernal = np.ones((kernal_s,kernal_s), np.uint8)
    grey = cv.GaussianBlur(img, (blur*2+1,blur*2+1), 3)
    grey = cv.cvtColor(grey, cv.COLOR_BGR2GRAY)
    imgCanny = cv.Canny(grey, threshold1, threshold2)
    imgdel = cv.dilate(imgCanny, kernal, iterations=itdel)
    evosion = cv.erode(imgdel, kernal, iterations=itero)
    thrash = cv.adaptiveThreshold(evosion, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv.THRESH_BINARY, 11, 2)
    
    _, contours, _ = cv.findContours(thrash, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    
    for contour in contours:
        approx = cv.approxPolyDP(contour, 0.1*cv.arcLength(contour, True), True)
        # area = cv.contourArea(approx)
        # if area > 2000*min_a:
        #     print(len(approx), '   ', area)


        
        if len(approx) == 4:
            flag = True
            _, _, w, h = cv.boundingRect(approx)
            aspectRatio = float(w)/h
            im_size = img.shape[0]*img.shape[1]
            if aspectRatio > 0.5 and aspectRatio < 1.5:
                if (w*h > int(min_a*im_size/614)) and (w*h < (max_a*im_size/409)):
                    cv.drawContours(img, [approx], 0, (255, 0, 0), 5)
                    coor_board = approx
                    

    return coor_board, flag


def correct_perspective(img, approx):                                                      
    pts = approx.reshape(4, 2)      
    tl, tr, br, bl =  (
        pts[np.argmin(np.sum(pts, axis=1))],
        pts[np.argmin(np.diff(pts, axis=1))],   
        pts[np.argmax(np.sum(pts, axis=1))],                           
        pts[np.argmax(np.diff(pts, axis=1))]
    )
    
    w = max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl))
    h = max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl))

    src = np.array([tl, tr, br, bl], dtype='float32')
    dst = np.array([[0, 0],[w, 0],[w, h],[0, h]],   
                    dtype='float32')                                        

    M = cv.getPerspectiveTransform(src, dst)
    img = cv.warpPerspective(img, M, (int(w), int(h)))                           

    return cv.resize(img, (400,400))


def get_square(img, row, col):
    width = img.shape[0]
    square = width // 8
    x1,y1 = row*square, col*square
    x2,y2 = x1+square, y1+square
    return img[x1:x2,y1:y2]


def detect_piece(img):
    threshold1 = thr1.val
    threshold2 = thr2.val
    
    blur = blr.val
    
    grey = cv.GaussianBlur(img, (blur*2+1,blur*2+1), 3)
    grey = cv.cvtColor(grey, cv.COLOR_BGR2GRAY)
    imgCanny = cv.Canny(grey, threshold1, threshold2)
    thrash_sq = cv.adaptiveThreshold(imgCanny, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                   cv.THRESH_BINARY, 11, 2)
    
    return thrash_sq


def update(val):
    
    frame = cv.imread('top_3.png')
    ## Работа со всей доской - вырез с перспектвой        
    coor_board, _ = board_detec(frame)
   
    if not(coor_board is None):
        imboard = correct_perspective(frame, coor_board)
        
    if imboard.shape[0] > 100 and imboard.shape[1] > 100:
        imboard = imboard[15:imboard.shape[0] - 15,15:imboard.shape[1] - 15]

    ## Вырезаем клетку
    square_00 = get_square(imboard, 0, 0)
    square_75 = get_square(imboard, 7, 5)
    square_43 = get_square(imboard, 4, 3)
    ## Обрабатываем клетку  
    thrash_sq_1 = detect_piece(square_00)
    thrash_sq_2 = detect_piece(square_75)
    thrash_sq_3 = detect_piece(square_43)

    ## Настриваем отображение результата в plt

    square_00 = cv.cvtColor(square_00, cv.COLOR_BGR2RGB)
    square_75 = cv.cvtColor(square_75, cv.COLOR_BGR2RGB)
    square_43 = cv.cvtColor(square_43, cv.COLOR_BGR2RGB)
    
    
    images = [square_00, square_75, square_43, thrash_sq_1, thrash_sq_2,
             thrash_sq_3, imboard, frame]
    for i in range(n_imgs):
        ax[i].imshow(images[i])
        


## input img
frame = cv.imread('top_3.png')
## number of output imgs
n_imgs = 8

## Set up matplotlib
fig = plt.figure()
fig.subplots_adjust(bottom=0.25)
ax = []
titles = ['square1', 'square2',  'square3', 'thrash_sq1', 
         'thrash_sq2', 'thrash_sq3',
            'imboard', 'frame']
for i in range(n_imgs):
    ax.append(fig.add_subplot(2,n_imgs//2,i+1))
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])


# Создаем трекбары
## Создаание Трекбаров для регулировки трещхолда и блюра в клетке
thr1ax = plt.axes([0.07,0.17,0.1,0.03])
thr2ax  = plt.axes([0.07,0.1,0.1,0.03])
blrax  = plt.axes([0.07,0.03,0.1,0.03])

thr1 = Slider(thr1ax, 'Threshold1', 0, 255, valstep=1, valinit=160)
thr2 = Slider(thr2ax, 'Threshold2', 0, 255, valstep=1, valinit=111)
blr = Slider(blrax, 'Blur', 0, 40, valstep=1, valinit=0)

thr1.on_changed(update)
thr2.on_changed(update)
blr.on_changed(update)

## Трекбары для обработки доски
thr1bx = plt.axes([0.25,0.12,0.1,0.03])
thr2bx = plt.axes([0.45,0.12,0.1,0.03])
kernbx = plt.axes([0.65,0.12,0.1,0.03])
blrbx = plt.axes([0.85,0.12,0.1,0.03])
itdelbx = plt.axes([0.25,0.05,0.1,0.03])
iterobx = plt.axes([0.45,0.05,0.1,0.03])
min_abx = plt.axes([0.65,0.05,0.1,0.03])
max_abx = plt.axes([0.85,0.05,0.1,0.03])

thr1S = Slider(thr1bx, 'Threshold1', 0, 255, valstep=1, valinit=216)
thr2S = Slider(thr2bx, 'Threshold2', 0, 255, valstep=1, valinit=111)
kern = Slider(kernbx, 'Blur', 0, 40, valstep=1, valinit=2)
blrS = Slider(blrbx, 'Threshold1', 0, 40, valstep=1, valinit=1)
itdel = Slider(itdelbx, 'itdel', 0, 2, valstep=1, valinit=1)
itero = Slider(iterobx, 'itero', 0, 2, valstep=1, valinit=1)
min_a = Slider(min_abx, 'min_a', 0, 255, valstep=1, valinit=43)
max_a = Slider(max_abx, 'max_a', 0, 255, valstep=1, valinit=239)

thr1S.on_changed(update)
thr2S.on_changed(update)
blrS.on_changed(update)
kern.on_changed(update)
itdel.on_changed(update)
itero.on_changed(update)
min_a.on_changed(update)
max_a.on_changed(update)

## Прогоняем все 1(аргуемнт "1" просто так) раз для отображения
update(1)

## Делаем окно мпл на весь экран и запускаем
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()
