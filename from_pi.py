import cv2 as cv
import numpy as np 
import chess


def empty(a):
    pass
 

def board_detec(img):
    flag = False
    coor_board = None
    threshold1 = 50
    threshold2 = 220
    kernal_s = 1
    blur = 1
    itdel = 3
    itero = 1
    min_a = 51
    max_a = 169
    kernal = np.ones((kernal_s,kernal_s), np.uint8)
    grey = cv.GaussianBlur(img, (blur*2+1,blur*2+1), 3)
    grey = cv.cvtColor(grey, cv.COLOR_BGR2GRAY)
    imgCanny = cv.Canny(grey, threshold1, threshold2)
    imgdel = cv.dilate(imgCanny, kernal, iterations=itdel)
    evosion = cv.erode(imgdel, kernal, iterations=itero)
    thrash = cv.adaptiveThreshold(evosion, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv.THRESH_BINARY, 11, 2)
    cv.imshow('trash', thrash)
    _, contours, _ = cv.findContours(thrash, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    cv.imshow('thrash', evosion)
    for contour in contours:
        approx = cv.approxPolyDP(contour, 0.1*cv.arcLength(contour, True), True)
        # area = cv.contourArea(approx)
        # if area > 2000*min_a:
        #     print(len(approx), '   ', area)


        
        if len(approx) == 4:
            flag = True
            x, y, w, h = cv.boundingRect(approx)
            aspectRatio = float(w)/h
            im_size = img.shape[0]*img.shape[1]
            if aspectRatio > 0.5 and aspectRatio < 1.5:
                if (w*h > int(min_a*im_size/614)) and (w*h < (max_a*im_size/409)):
                    cv.drawContours(img, [approx], 0, (255, 0, 0), 5)
                    coor_board = approx
                    

    return coor_board, flag


def board_pos(img):
    None


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


#cap = cv.VideoCapture(0)
board = chess.Board()
imboard = cv.imread('chess_top_1.jpg')

cv.namedWindow("Parameters")
cv.resizeWindow("Parameters",640,240)
cv.namedWindow("Size")
cv.resizeWindow("Size",640,240)
cv.createTrackbar("Threshold1","Parameters",50,255,empty)
cv.createTrackbar("Threshold2","Parameters",220,255,empty)
cv.createTrackbar("kernal","Parameters",1,255,empty)
cv.createTrackbar("blur","Parameters",1,100,empty)
cv.createTrackbar("itdel","Parameters",1,5,empty)
cv.createTrackbar("itero","Parameters",1,5, empty)
cv.createTrackbar("min_a","Size",51,255, empty)
cv.createTrackbar("max_a","Size",169,255, empty)

#while(cap.isOpened()):
while True:
    frame = cv.imread('chess_top_1.jpg')
        #ret, frame = cap.read() 
            
    coor_board, flag = board_detec(frame)
    print(coor_board)
    if not(coor_board is None):
        imboard = correct_perspective(frame, coor_board)
        
    if imboard.shape[0] > 100 and imboard.shape[1] > 100:
        imboard = imboard[15:imboard.shape[0] - 15,15:imboard.shape[1] - 15]
        # cv.imshow('thrash', thrash)
        cv.imshow('board', imboard)
        cv.imshow('img', frame)
        print(imboard.shape)
    square = get_square(imboard, 0, 0)
        #cv.imshow('square', square)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


# cap.release()
cv.destroyAllWindows()