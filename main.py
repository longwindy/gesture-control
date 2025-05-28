import cv2
from HandTrackingModule import HandDetector
from time import sleep
import numpy as np
from pynput.keyboard import Controller
import time
from button import Button
import keyboardConfig
from config import *

## Global variables
last_state_time = 0
last_click_time = 0
indexLm = 8
clickLm = 4
buttonList = []
fingers_up = []  

# record the state of the mode, 0: normal mode, 1: keyboard mode
current_mode = 0
# record the state of the finger, 0: closed, 1: open
finger_state = 0
# record the number of fingers up
finger_count = 0 

# Draw all buttons and text
def drawAll(img, buttonList):
    imgNew = np.zeros_like(img, np.uint8)
    for button in buttonList:
        button.draw(imgNew, buttonColor=buttonColor, textColor=textColor, fontScale=2, thickness=3)
    
    cv2.rectangle(imgNew, (50, 550), (700, 650), (210, 146, 192), cv2.FILLED)
    cv2.putText(imgNew, finalText, (60, 625), cv2.FONT_HERSHEY_PLAIN, 5, textColor, 5)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
    return out

# Check if the button is clicked
def isClicked(lmList, indexLm, clickLm, bboxInfo, img, dynamic_scale=0.08, click_interval=1):
    x1, y1 = lmList[indexLm][0], lmList[indexLm][1]
    x2, y2 = lmList[clickLm][0], lmList[clickLm][1]

    clickThreshold = 5 + (bboxInfo[2] + bboxInfo[3]) * dynamic_scale
    
    l, _, _ = detector.findDistance((x1, y1), (x2, y2), img)
    # if debugMode:
    #     print(f"distance: {l:.2f} ")
    return l < clickThreshold and time.time() - last_click_time >= click_interval

# Check if the mode has been switched
def check_finger_mode_switch():
    global finger_state, last_state_time, current_mode
    if finger_count == 5:
        finger_state = 1
        last_state_time = time.time()
    elif finger_count < 5 and finger_state == 1 and finger_count > 0 and time.time() - last_state_time > switch_delay:
        finger_state = 0
    elif finger_count == 0 and finger_state == 1:
        current_mode = 1 - current_mode  # switch mode
        finger_state = 0
        print(f"mode has switched to {current_mode}")
    return current_mode

# Initialization
def init(cap):
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

if __name__ == "__main__":
    # Initialize the hand detector and keyboard controller
    keyboard = Controller()
    detector = HandDetector(detectionCon=0.8)
    cap = cv2.VideoCapture(camaraIdx)
    finalText = ""
    
    init(cap)

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img, draw=True, flipType=False)
        lmList, bboxInfo = [], []
        if hands:
            lmList, bboxInfo = hands[0]["lmList"], hands[0]["bbox"]
            # detect the fingers up
            fingers_up = detector.fingersUp(hands[0])
            # detect the number of fingers up  
            finger_count = fingers_up.count(1)
        
        # check the mode switch
        current_mode = check_finger_mode_switch()

        # when in normal mode, draw the bbox and the landmarks
        if bboxInfo and debugMode:
            l, _, _ = detector.findDistance((lmList[indexLm][0], lmList[indexLm][1]), (lmList[clickLm][0], lmList[clickLm][1]), img)
            bboxWidth = bboxInfo[2]
            bboxHeight = bboxInfo[3]
            sum = bboxInfo[2]+bboxInfo[3]
            print(f"boxWidth: {bboxWidth} boxHeight: {bboxHeight} sum: {sum} distance: {l:.2f} rate: {l/sum:.3f}")
      
        # when in keyboard mode, draw the buttons and text
        if current_mode == 1:
            img = drawAll(img, buttonList)
      
            # when in keyboard mode, detect the click
            if lmList and len(fingers_up) >= 5:
                midPoint = ((lmList[indexLm][0]+lmList[clickLm][0])//2, (lmList[indexLm][1]+lmList[clickLm][1])//2)
                
                for button in buttonList:
                    x, y = button.pos
                    w, h = button.size
                    
                    if x < midPoint[0] < x + w and y < midPoint[1] < y + h:
                        button.draw(img, buttonColor=buttonHoverColor, textColor=textColor, fontScale=4, thickness=4)

                        # only when the index finger and middle finger are up, the button can be clicked
                        if isClicked(lmList, indexLm, clickLm, bboxInfo, img) and fingers_up[0] and fingers_up[1]:
                            last_click_time = time.time()
                            keyboard.press(button.text)
                            button.draw(img, buttonColor=buttonClickColor, textColor=textColor, fontScale=4, thickness=4)
                            finalText += button.text

                cv2.circle(img, midPoint, 8, (255, 255, 255), 2)
        
        w,h = img.shape[:2]
        display_w, display_h = int(w * display_scale), int(h * display_scale)
        x1, y1 = (w - display_w) // 2, (h - display_h) // 2
        img = img[x1:x1+display_w, y1:y1+display_h]
        cv2.imshow("Image", img)
        cv2.waitKey(1)