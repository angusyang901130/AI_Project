import cv2
import numpy as np
import pickle

def build_squares(image):
    x, y, w, h = 420, 140, 10, 10
    d = 10

    image_crop = None
    crop = None

    # set up the squares for histogram, draw green squares for the range for hand
    for i in range(10):
        for j in range(5):
            if np.any(image_crop == None):
                image_crop = image[y:y+h, x:x+w]
            else:
                image_crop = np.hstack((image_crop, image_crop[y:y+h, x:x+w]))
            
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
            x += w + d
        
        if np.any(crop == None):
            crop = image_crop
        else:
            crop = np.vstack((crop, image_crop))
        
        x = 420
        y += h + d

    return crop

def get_hand_hist():
    cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        cam.open()
    
    x, y, w, h = 300, 100, 300, 300
    press_c, press_s = False, False

    image_crop = None
    while True:
        image = cam.read()[1]  # cam.read()[0] is a bool, indicates if it is read
        image = cv2.flip(image, 1)  # left right opposite
        image = cv2.resize(image, (640, 480))  
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # hsv 

        press_key = cv2.waitkey(1)

        if not press_s:
            image_crop = build_squares(image)

        if press_key == ord('c'):
            hsv_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
            press_c = True
            hist = cv2.calcHist([hsv_crop], [0, 1], None, [180, 256], [0, 180, 0, 256])  # make historgram, decide color, and number of bins
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)  # normalize and make black and white more clear
        elif press_key == ord('d'):
            press_s = True
            break
    
        if press_c:
            back_proj = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)      # histogram back projection
            print(back_proj)