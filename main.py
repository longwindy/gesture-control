import cv2
from HandTrackingModule import HandDetector
import numpy as np
from pynput.keyboard import Controller
import time
from button import Button
import keyboardConfig
from config import *
import autopy
import ctypes
from algorithm_setting import *

# Global variables
last_state_time = 0
last_click_time = 0
indexLm = 8
clickLm = 4
buttonList = []
fingers_up = []
wScr, hScr = autopy.screen.size()   # Need repair
# wScr, hScr = ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
pLocx, pLocy = 0, 0
cLocx, cLocy = 0, 0
prev_x1, prev_y1 = 0, 0
finalText = ""

# Record the state of the mode, 0: normal mode, 1: keyboard mode
current_mode = 0
# Record the state of the finger, 0: closed, 1: open
finger_state = 0
# Record the number of fingers up
finger_count = 0

# Draw all buttons and text on the image
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

    clickThreshold = 25 + (bboxInfo[2] + bboxInfo[3]) * dynamic_scale
    l, _, _ = detector.findDistance((x1, y1), (x2, y2), img)
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
        current_mode = 1 - current_mode  # Switch mode
        finger_state = 0
        print(f"Mode has switched to {current_mode}")
    return current_mode

# Initialization function
def init(cap, videoWidth=videoWidth, videoHeight=videoHeight):
    cap.set(3, videoWidth)
    cap.set(4, videoHeight)
    global indexLm, clickLm
    if click_mode == 0:
        indexLm = 8
        clickLm = 12
    elif click_mode == 1:
        indexLm = 8
        clickLm = 4
    
    keyboardConfig.init_keyboard(keyboard_start_x, keyboard_start_y, buttonList, button_size)

# Function to apply different filters and record metrics
def apply_filters_and_record(evaluator, kf, ma_history_x, ma_history_y, x3, y3, pLocx, pLocy):
    # Low pass filter
    x_data = np.array([pLocx, x3])
    y_data = np.array([pLocy, y3])
    lp_x, lp_y = lowpass_filter(x_data)[-1], lowpass_filter(y_data)[-1]
    lp_error = np.sqrt((lp_x - x3)**2 + (lp_y - y3)**2)
    evaluator.record('lowpass', (lp_x, lp_y), lp_error)

    # Kalman filter
    z = np.array([[x3], [y3]])
    kf.predict()
    kf.update(z)
    kf_x, kf_y = kf.x[0], kf.x[1]
    kf_error = np.sqrt((kf_x - x3)**2 + (kf_y - y3)**2)
    evaluator.record('kalman', (kf_x, kf_y), kf_error)

    # Moving average
    ma_x = moving_average(ma_history_x, x3)
    ma_y = moving_average(ma_history_y, y3)
    ma_error = np.sqrt((ma_x - x3)**2 + (ma_y - y3)**2)
    evaluator.record('moving_avg', (ma_x, ma_y), ma_error)

    return kf_x, kf_y

# Function to display performance metrics
def display_metrics(evaluator, img):
    metrics = evaluator.get_metrics()
    y_pos = 30
    for algo in metrics:
        text = f"{algo}: Avg Err {metrics[algo]['avg_error']:.2f} | Max Err {metrics[algo]['max_error']:.2f} | Avg Jitter {metrics[algo]['avg_jitter']:.2f} | Max Jitter {metrics[algo]['max_jitter']:.2f}"
        cv2.putText(img, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30

if __name__ == "__main__":
    # Initialize the hand detector and keyboard controller
    keyboard = Controller()
    detector = HandDetector(detectionCon=0.8)
    cap = cv2.VideoCapture(camaraIdx)
    finalText = ""
    
    init(cap, videoWidth=videoWidth, videoHeight=videoHeight)

    # Initialize Kalman filter, moving average history, and performance evaluator
    kf = init_kalman_filter()
    ma_history_x = []
    ma_history_y = []
    evaluator = PerformanceEvaluator()

    while True:
        success, img = cap.read()
        if not success:
            continue
        img = cv2.flip(img, 1)
        # Get the actual width and height of the video
        actual_width = int(cap.get(3))
        actual_height = int(cap.get(4))
        # Calculate the coordinates of the rectangle
        pt1, pt2 = (int(0.2 * actual_width), int(0.2 * actual_height)), (int(0.8 * actual_width), int(0.8 * actual_height))
        
        hands, img = detector.findHands(img, draw=True, flipType=False)
        lmList, bboxInfo = [], []
        if hands:
            lmList, bboxInfo = hands[0]["lmList"], hands[0]["bbox"]
            # Detect the number of fingers up
            fingers_up = detector.fingersUp(hands[0])
            finger_count = fingers_up.count(1)
        
            # Check the mode switch
            current_mode = check_finger_mode_switch()

            # When in normal mode, draw the bounding box and landmarks
            if bboxInfo and debugMode:
                l, _, _ = detector.findDistance((lmList[indexLm][0], lmList[indexLm][1]), (lmList[clickLm][0], lmList[clickLm][1]), img)
                bboxWidth = bboxInfo[2]
                bboxHeight = bboxInfo[3]
                sum_val = bboxInfo[2] + bboxInfo[3]
                # print(f"boxWidth: {bboxWidth} boxHeight: {bboxHeight} sum: {sum_val} distance: {l:.2f} rate: {l/sum_val:.3f} threshold: {(5+sum_val*0.08):.2f}")

            if current_mode == 0:
                # Draw the rectangle on the screen
                cv2.rectangle(img, pt1, pt2, (0, 255, 255), 5)
                # Get the coordinates of the index finger tip and thumb tip
                x1, y1 = lmList[8][:2]
                x2, y2 = lmList[4][:2]

                # Calculate the movement distance
                movement = np.sqrt((x1 - prev_x1) ** 2 + (y1 - prev_y1) ** 2)

                # If the index finger is up and the thumb is down, move the mouse
                if fingers_up[1] and not fingers_up[0]:
                    if pt1[0] - 10 <= x1 <= pt2[0] + 10 and pt1[1] - 10 <= y1 <= pt2[1] + 10:
                        # Draw a circle at the index finger tip
                        cv2.circle(img, (x1, y1), 15, (255, 255, 0), cv2.FILLED)
                
                        # Original coordinates
                        x3 = np.interp(x1, (pt1[0], pt2[0]), (0, wScr))
                        y3 = np.interp(y1, (pt1[1], pt2[1]), (0, hScr))

                        # Apply filters and record metrics
                        cLocx, cLocy = apply_filters_and_record(evaluator, kf, ma_history_x, ma_history_y, x3, y3, pLocx, pLocy)
                        
                        # Display performance metrics
                        display_metrics(evaluator, img)

                        # Ensure the coordinates are within the valid range
                        cLocx = max(0, min(cLocx, wScr - 1))
                        cLocy = max(0, min(cLocy, hScr - 1))
                        # Move the mouse
                        if movement > STATIC_THRESHOLD:
                            autopy.mouse.move(cLocx, cLocy)
                        
                        # Update the previous mouse position
                        pLocx, pLocy = cLocx, cLocy

                    prev_x1, prev_y1 = x1, y1

                # Left-click if the right hand's thumb and index finger are closed
                if hands[0]['type'] == 'Right':
                    if isClicked(lmList, indexLm, clickLm, bboxInfo, img, dynamic_scale=dynamic_scale, click_interval=click_interval) and fingers_up[0] and fingers_up[1]:
                        autopy.mouse.click()
                # Right-click if the left hand's thumb and index finger are closed
                elif hands[0]['type'] == 'Left':
                    if isClicked(lmList, indexLm, clickLm, bboxInfo, img, dynamic_scale=dynamic_scale, click_interval=click_interval) and fingers_up[0] and fingers_up[1]:
                        autopy.mouse.click(button=autopy.mouse.Button.RIGHT)

        # When in keyboard mode, draw buttons and text
        if current_mode == 1:
            img = drawAll(img, buttonList)
    
            # Detect button clicks in keyboard mode
            if lmList and len(fingers_up) >= 5:
                midPoint = ((lmList[indexLm][0] + lmList[clickLm][0]) // 2, (lmList[indexLm][1] + lmList[clickLm][1]) // 2)
                
                for button in buttonList:
                    x, y = button.pos
                    w, h = button.size
                    
                    if x < midPoint[0] < x + w and y < midPoint[1] < y + h:
                        button.draw(img, buttonColor=buttonHoverColor, textColor=textColor, fontScale=4, thickness=4)

                        # Click the button only when the index finger and middle finger are up
                        if isClicked(lmList, indexLm, clickLm, bboxInfo, img) and fingers_up[0] and fingers_up[1]:
                            if debugMode:
                                print("Clicked:", button.text)
                            last_click_time = time.time()
                            if button.action:
                                for func in button.action:
                                    func()
                            button.draw(img, buttonColor=buttonClickColor, textColor=textColor, fontScale=4, thickness=4)
                            finalText += button.text

                cv2.circle(img, midPoint, 8, (255, 255, 255), 2)
      
        cv2.imshow(f"Image", img)
        cv2.waitKey(1)