import cv2
from cvzone.HandTrackingModule import HandDetector
from time import sleep
import numpy as np
from pynput.keyboard import Controller
import time
from button import Button
import keyboardConfig
from config import *

## Global variables
last_click_time = 0
indexLm = 8
clickLm = 12
buttonList = []

keyboard = Controller()
detector = HandDetector(detectionCon=0.8)
cap = cv2.VideoCapture(camaraIdx)
finalText = ""

# Draw all buttons and text
def drawAll(img, buttonList):
    imgNew = np.zeros_like(img, np.uint8)
    for button in buttonList:
        button.draw(imgNew, buttonColor=(255, 0, 255), textColor=(255, 255, 255), fontScale=2, thickness=3)
    
    cv2.rectangle(imgNew, (50, 550), (700, 650), (210, 146, 192), cv2.FILLED)
    cv2.putText(imgNew, finalText, (60, 625), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
    return out

# Check if the button is clicked
def isClicked(lmList, indexLm, clickLm, clickThreshold=45, click_interval=click_interval):
    x1, y1 = lmList[indexLm][0], lmList[indexLm][1]
    x2, y2 = lmList[clickLm][0], lmList[clickLm][1]
    l, _, _ = detector.findDistance((x1, y1), (x2, y2), img)
    if debugMode:
        print(l)
    return l < clickThreshold and time.time() - last_click_time >= click_interval

# Initialization
def init():
    global cap
    cap.set(3, videoWidth)
    cap.set(4, videoHeight)
    global indexLm
    global clickLm
    if click_mode == 0:
        indexLm = 8
        clickLm = 12
    elif click_mode == 1:
        indexLm = 8
        clickLm = 4
    
    keyboardConfig.init_keyboard(keyboard_start_x, keyboard_start_y, buttonList, button_size)



init()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, draw=True, flipType=False)
    lmList, bboxInfo = [], []
    if hands:
        lmList, bboxInfo = hands[0]["lmList"], hands[0]["bbox"]
    img = drawAll(img, buttonList)

    if lmList:
        for button in buttonList:
            x, y = button.pos
            w, h = button.size
            
            if x < lmList[indexLm][0] < x + w and y < lmList[indexLm][1] < y + h:
                button.draw(img, buttonColor=(175, 0, 175), textColor=(255, 255, 255), fontScale=4, thickness=4)

                ## When clicked
                if isClicked(lmList, indexLm, clickLm):
                    last_click_time = time.time()
                    keyboard.press(button.text)
                    button.draw(img, buttonColor=(0, 255, 0), textColor=(255, 255, 255), fontScale=4, thickness=4)
                    finalText += button.text
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)

