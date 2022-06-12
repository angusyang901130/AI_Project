import cv2, pickle
import numpy as np
from sqlalchemy import modifier
from cnn import get_image_size
import sqlite3
import pyttsx3
import os

engine = pyttsx3.init()

# set the speaking rate
engine.setProperty('rate', 100)

def process_img(img, img_x, img_y):
    img = cv2.resize(img, (img_x, img_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (1, img_x, img_y, 1))
    return img

def model_predict(model, img):
    img_x, img_y = get_image_size()
    processed = process_img(img, img_x, img_y)
    pred_prob = model.predict(processed)[0]
    pred_class = list(pred_prob).index(max(pred_prob))
    return max(pred_prob), pred_class

def get_pred_text(pred_class):
    connect_db = sqlite3.connect("gesture_db.db")
    cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
    cursor = connect_db.execute(cmd)
    for row in cursor:
        return row[0]

def split_sentence(text, num_of_words):
    '''
	Splits a text into group of num_of_words
	'''
    list_words = text.split(" ")
    length = len(list_words)
    splitted_sentence = []
    begin_ind = 0
    end_ind = num_of_words
    while length > 0:
        part = ""
        for word in list_words[begin_ind:end_ind]:
            part = part + " " + word
        splitted_sentence.append(part)
        begin_ind += num_of_words
        end_ind += num_of_words
        length -= num_of_words
    return splitted_sentence

def put_text_in_blackboard(blackboard, splitted_text):
    # cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
    y = 200
    for text in splitted_text:
        cv2.putText(blackboard, "Predicted text :" + text, (8, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
        cv2.putText(blackboard, text, (4, y), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
        y += 50

def get_hand_hist():
    with open("hist", "rb") as f:
        hist = pickle.load(f)
    return hist

def say_sentence(sentence):
    while engine._inLoop:
        pass
    engine.say(sentence)
    engine.runAndWait()

def recognize(model):
    cap = cv2.VideoCapture(1)
    if cap.read()[0] == False:
        cap = cv2.VideoCapture(0)

    hist = get_hand_hist()
    x, y, w, h = 300, 100, 300, 300

    times = 0
    sentence = ""
    text = ""
    while True:
        img = cap.read()[1]
        
        if img is None:
            continue

        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        # imgCrop = img[y:y+h, x:x+w]
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # img projection
        dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        # Convolution
        cv2.filter2D(dst, -1, disc, dst)

        blur = cv2.GaussianBlur(dst, (11, 11), 0)
        blur = cv2.medianBlur(blur, 17)
        thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        thresh = cv2.merge((thresh, thresh, thresh))
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        thresh = thresh[y:y+h, x:x+w]

        # check openCV version
        (openCV_ver, _, _) = cv2.__version__.split(".")
        if openCV_ver == '3':
            contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
        elif openCV_ver == '4':
            contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        if len(contours) > 0:
            contour = max(contours, key = cv2.contourArea)
            if cv2.contourArea(contour) > 10000:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                save_img = thresh[y1:y1+h1, x1:x1+w1]

                if w1 > h1:
                    save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2), int((w1-h1)/2), 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                elif h1 > w1:
                    save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
				
                pred_prob, pred_class = model_predict(model, save_img)

                if pred_prob*100 > 80:
                    old_text = text
                    text = get_pred_text(pred_class)
                    print(old_text ,text)
                    if old_text == text:
                        times += 1
                    else: 
                        times = 0
                    if times >= 20:
                        sentence += text
                        times = 0
                        if sentence.startswith('I/Me'):
                            sentence = sentence.replace('I/Me', 'I')
                        elif sentence.endswith('I/Me'):
                            sentence.replace('I/Me', 'me')
            else:
                sentence = ""
                text = ""
            blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blackboard, " ", (180, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0,0))
            cv2.putText(blackboard, "Predicted text- " + text, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
            cv2.putText(blackboard, sentence, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255)) 
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            res = np.hstack((img, blackboard))
            cv2.imshow("Recognizing gesture", res)
            cv2.imshow("thresh", thresh)
            keyboard_input = cv2.waitKey(1)
            if keyboard_input == 32:
                sentence += " "
            if keyboard_input == ord('m'):
                while True:
                    modify = cv2.waitKey(1)
                    if modify == ord('d'):
                        sentence = sentence[:-1]
                    if modify == ord('q'):
                        break
                    blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blackboard, " ", (180, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0,0))
                    cv2.putText(blackboard, "Press d to delete ", (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
                    cv2.putText(blackboard, sentence, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255)) 
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    res = np.hstack((img, blackboard))
                    cv2.imshow("Recognizing gesture", res)
                    cv2.imshow("thresh", thresh)
            if keyboard_input == ord('s'):
                say_sentence(sentence)
            if keyboard_input == ord('q'):
                print(sentence)
                break
    # print("Ans: ", sentence)


""" model_predict(model, np.zero((50, 50), dtype=np.uint8))
recognize() """
