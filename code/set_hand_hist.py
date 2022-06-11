import cv2
import numpy as np
import pickle

def build_squares(img):
    x, y, w, h = 420, 140, 10, 10
    d = 10

    img_crop = None
    crop = None

    # set up the squares for histogram, draw green squares for the range for hand
    for i in range(10):
        for j in range(5):
            if np.any(img_crop == None):
                img_crop = img[y:y+h, x:x+w]
            else:
                img_crop = np.hstack((img_crop, img[y:y+h, x:x+w]))
            
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            x += w + d
        
        if np.any(crop == None):
            crop = img_crop
        else:
            crop = np.vstack((crop, img_crop))

        img_crop = None
        
        x = 420
        y += h + d

    return crop

def set_hand_hist():
    cam = cv2.VideoCapture(1)
    cam = cv2.VideoCapture(1)
    if cam.read()[0] == False:
        cam = cv2.VideoCapture(0)
    
    x, y, w, h = 300, 100, 300, 300
    press_c, press_s = False, False

    img_crop = None
    while True:
        img = cam.read()[1]  # cam.read()[0] is a bool, indicates if it is read    
          
        if img is None:
            continue
             
        img = cv2.flip(img, 1)  # left right opposite
        img = cv2.resize(img, (640, 480)) 
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # hsv 

        press_key = cv2.waitKey(1)

        if not press_s:
            img_crop = build_squares(img)

        if press_key == ord('c'):
            hsv_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
            press_c = True
            hist = cv2.calcHist([hsv_crop], [0, 1], None, [180, 256], [0, 180, 0, 256])  # make historgram, decide color, and number of bins
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)  # normalize and make black and white more clear
        elif press_key == ord('s'):
            press_s = True
            break
    
        if press_c:
            back_proj = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)  # histogram back projection
            #print(back_proj)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))  # kernel

            cv2.filter2D(back_proj, -1, kernel, back_proj)  # create convolution between image 

            blur = cv2.GaussianBlur(back_proj, (11, 11), 0) 
            blur = cv2.medianBlur(blur, 17)

            ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            thresh = cv2.merge((thresh, thresh, thresh))

            cv2.imshow('Threshold', thresh)

        cv2.imshow("Set hand histogram", img)

    cam.release()
    cv2.destroyAllWindows()
    with open("hist", "wb") as f:
        pickle.dump(hist, f)
        