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
from algorithm_setting import LowPassFilter, MovingAverageFilter, ExtendedKalmanFilterWrapper, KalmanFilterWrapper, PerformanceEvaluator

class GestureControlApp:
    def __init__(self):
        # Initialize keyboard controller and hand detector
        self.keyboard = Controller()
        self.detector = HandDetector(detectionCon=0.8)
        self.cap = cv2.VideoCapture(camaraIdx)
        self.finalText = ""

        # Initialize global variables
        self.last_state_time = 0
        self.last_click_time = 0
        self.indexLm = 8
        self.clickLm = 4
        self.buttonList = []
        self.fingers_up = []
        self.wScr, self.hScr = autopy.screen.size()
        self.pLocx, self.pLocy = 0, 0
        self.cLocx, self.cLocy = 0, 0
        self.prev_x1, self.prev_y1 = 0, 0
        self.current_mode = 0
        self.finger_state = 0
        self.finger_count = 0

        # Initialize filters and performance evaluator
        self.lowpass_filter_x = LowPassFilter()
        self.lowpass_filter_y = LowPassFilter()
        self.ekf = ExtendedKalmanFilterWrapper(dt=1/30.0)
        self.moving_avg_filter_x = MovingAverageFilter()
        self.moving_avg_filter_y = MovingAverageFilter()
        self.kf = KalmanFilterWrapper(dt=1/30.0)
        self.evaluator = PerformanceEvaluator()

    def init(self, videoWidth=videoWidth, videoHeight=videoHeight):
        """Initialize camera and keyboard buttons"""
        self.cap.set(3, videoWidth)
        self.cap.set(4, videoHeight)
        if click_mode == 0:
            self.indexLm = 8
            self.clickLm = 12
        elif click_mode == 1:
            self.indexLm = 8
            self.clickLm = 4
        keyboardConfig.init_keyboard(keyboard_start_x, keyboard_start_y, self.buttonList, button_size)

    def draw_all(self, img):
        """Draw all buttons and text on the image"""
        imgNew = np.zeros_like(img, np.uint8)
        for button in self.buttonList:
            button.draw(imgNew, buttonColor=buttonColor, textColor=textColor, fontScale=2, thickness=3)
        
        cv2.rectangle(imgNew, (50, 550), (700, 650), (210, 146, 192), cv2.FILLED)
        cv2.putText(imgNew, self.finalText, (60, 625), cv2.FONT_HERSHEY_PLAIN, 5, textColor, 5)

        out = img.copy()
        alpha = 0.5
        mask = imgNew.astype(bool)
        out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
        return out

    def is_clicked(self, lmList, bboxInfo, img, dynamic_scale=0.08, click_interval=1):
        """Check if a button is clicked"""
        x1, y1 = lmList[self.indexLm][0], lmList[self.indexLm][1]
        x2, y2 = lmList[self.clickLm][0], lmList[self.clickLm][1]

        clickThreshold = 25 + (bboxInfo[2] + bboxInfo[3]) * dynamic_scale
        l, _, _ = self.detector.findDistance((x1, y1), (x2, y2), img)
        return l < clickThreshold and time.time() - self.last_click_time >= click_interval

    def check_finger_mode_switch(self):
        """Check for mode switching based on finger count"""
        if self.finger_count == 5:
            self.finger_state = 1
            self.last_state_time = time.time()
        elif self.finger_count < 5 and self.finger_state == 1 and self.finger_count > 0 and time.time() - self.last_state_time > switch_delay:
            self.finger_state = 0
        elif self.finger_count == 0 and self.finger_state == 1:
            self.current_mode = 1 - self.current_mode  # Toggle mode
            self.finger_state = 0
            print(f"Mode has switched to {self.current_mode}")
        return self.current_mode

    def apply_filters_and_record(self, x3, y3):
        """Apply different filters and record metrics"""
        # Low-pass filter
        lp_x = self.lowpass_filter_x.filter(x3)
        lp_y = self.lowpass_filter_y.filter(y3)
        lp_error = np.sqrt((lp_x - x3)**2 + (lp_y - y3)**2)
        self.evaluator.record('lowpass', (lp_x, lp_y), lp_error)

        # Extended Kalman Filter
        z = np.array([x3, y3])
        self.ekf.predict()
        self.ekf.update(z)
        ekf_x, ekf_y = self.ekf.get_state()[0], self.ekf.get_state()[1]
        ekf_error = np.sqrt((ekf_x - x3)**2 + (ekf_y - y3)**2)
        self.evaluator.record('ekf', (ekf_x, ekf_y), ekf_error)

        # Moving Average filter
        ma_x = self.moving_avg_filter_x.filter(x3)
        ma_y = self.moving_avg_filter_y.filter(y3)
        ma_error = np.sqrt((ma_x - x3)**2 + (ma_y - y3)**2)
        self.evaluator.record('moving_avg', (ma_x, ma_y), ma_error)

        # Kalman Filter
        self.kf.predict()
        self.kf.update(z)
        kf_x, kf_y = self.kf.get_state()[0], self.kf.get_state()[1]
        kf_error = np.sqrt((kf_x - x3)**2 + (kf_y - y3)**2)
        self.evaluator.record('kf', (kf_x, kf_y), kf_error)

        return ekf_x, ekf_y

    def display_metrics(self, img):
        """Display performance metrics on screen"""
        metrics = self.evaluator.get_metrics()
        y_pos = 30
        for algo in metrics:
            text = f"{algo}: Avg Err {metrics[algo]['avg_error']:.2f} | Max Err {metrics[algo]['max_error']:.2f} | Avg Jitter {metrics[algo]['avg_jitter']:.2f} | Max Jitter {metrics[algo]['max_jitter']:.2f}"
            cv2.putText(img, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += 30

    def run(self):
        """Run the gesture control application"""
        self.init()

        while True:
            success, img = self.cap.read()
            if not success:
                continue
            img = cv2.flip(img, 1)
            # Get actual video dimensions
            actual_width = int(self.cap.get(3))
            actual_height = int(self.cap.get(4))
            # Calculate rectangle coordinates
            pt1, pt2 = (int(0.2 * actual_width), int(0.2 * actual_height)), (int(0.8 * actual_width), int(0.8 * actual_height))
            
            hands, img = self.detector.findHands(img, draw=True, flipType=False)
            lmList, bboxInfo = [], []
            if hands:
                lmList, bboxInfo = hands[0]["lmList"], hands[0]["bbox"]
                # Detect number of raised fingers
                self.fingers_up = self.detector.fingersUp(hands[0])
                self.finger_count = self.fingers_up.count(1)
            
                # Check for mode switching
                self.current_mode = self.check_finger_mode_switch()

                # Draw bounding box and keypoints in normal mode
                if bboxInfo and debugMode:
                    l, _, _ = self.detector.findDistance((lmList[self.indexLm][0], lmList[self.indexLm][1]), (lmList[self.clickLm][0], lmList[self.clickLm][1]), img)
                    bboxWidth = bboxInfo[2]
                    bboxHeight = bboxInfo[3]
                    sum_val = bboxInfo[2] + bboxInfo[3]
                    # Debug output: (commented out in production)
                    # print(f"boxWidth: {bboxWidth} boxHeight: {bboxHeight} sum: {sum_val} distance: {l:.2f} rate: {l/sum_val:.3f} threshold: {(5+sum_val*0.08):.2f}")

                if self.current_mode == 0:
                    # Draw rectangle on screen
                    cv2.rectangle(img, pt1, pt2, (0, 255, 255), 5)
                    # Get coordinates of index and thumb tips
                    x1, y1 = lmList[8][:2]
                    x2, y2 = lmList[4][:2]

                    # Calculate movement distance
                    movement = np.sqrt((x1 - self.prev_x1) ** 2 + (y1 - self.prev_y1) ** 2)

                    # Move mouse if index finger raised and thumb down
                    if self.fingers_up[1] and not self.fingers_up[0]:
                        if pt1[0] - 10 <= x1 <= pt2[0] + 10 and pt1[1] - 10 <= y1 <= pt2[1] + 10:
                            # Draw circle at index fingertip
                            cv2.circle(img, (x1, y1), 15, (255, 255, 0), cv2.FILLED)
                    
                            # Raw coordinates
                            x3 = np.interp(x1, (pt1[0], pt2[0]), (0, self.wScr))
                            y3 = np.interp(y1, (pt1[1], pt2[1]), (0, self.hScr))

                            # Apply filters and record metrics
                            self.cLocx, self.cLocy = self.apply_filters_and_record(x3, y3)
                            
                            # Display performance metrics
                            self.display_metrics(img)

                            # Ensure coordinates are within valid range
                            self.cLocx = max(0, min(self.cLocx, self.wScr - 1))
                            self.cLocy = max(0, min(self.cLocy, self.hScr - 1))
                            # Move mouse
                            if movement > STATIC_THRESHOLD:
                                autopy.mouse.move(self.cLocx, self.cLocy)
                            
                            # Update previous mouse position
                            self.pLocx, self.pLocy = self.cLocx, self.cLocy

                        self.prev_x1, self.prev_y1 = x1, y1

                    # Left-click if right hand thumb and index close
                    if hands[0]['type'] == 'Right':
                        if self.is_clicked(lmList, bboxInfo, img, dynamic_scale=dynamic_scale, click_interval=click_interval) and self.fingers_up[0] and self.fingers_up[1]:
                            autopy.mouse.click()
                    # Right-click if left hand thumb and index close
                    elif hands[0]['type'] == 'Left':
                        if self.is_clicked(lmList, bboxInfo, img, dynamic_scale=dynamic_scale, click_interval=click_interval) and self.fingers_up[0] and self.fingers_up[1]:
                            autopy.mouse.click(button=autopy.mouse.Button.RIGHT)

            # Draw buttons and text in keyboard mode
            if self.current_mode == 1:
                img = self.draw_all(img)
        
                # Detect button clicks in keyboard mode
                if lmList and len(self.fingers_up) >= 5:
                    midPoint = ((lmList[self.indexLm][0] + lmList[self.clickLm][0]) // 2, (lmList[self.indexLm][1] + lmList[self.clickLm][1]) // 2)
                    
                    for button in self.buttonList:
                        x, y = button.pos
                        w, h = button.size
                        
                        if x < midPoint[0] < x + w and y < midPoint[1] < y + h:
                            button.draw(img, buttonColor=buttonHoverColor, textColor=textColor, fontScale=4, thickness=4)

                            # Click button only when index and middle fingers raised
                            if self.is_clicked(lmList, bboxInfo, img) and self.fingers_up[0] and self.fingers_up[1]:
                                if debugMode:
                                    print("Clicked:", button.text)
                                self.last_click_time = time.time()
                                if button.action:
                                    for func in button.action:
                                        func()
                                button.draw(img, buttonColor=buttonClickColor, textColor=textColor, fontScale=4, thickness=4)
                                self.finalText += button.text

                    cv2.circle(img, midPoint, 8, (255, 255, 255), 2)
          
            cv2.imshow(f"Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = GestureControlApp()
    app.run()