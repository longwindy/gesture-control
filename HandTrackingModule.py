import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, 
               static_image_mode=False,
               max_num_hands=2,
               model_complexity=1,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
        
        self.results = None
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            self.static_image_mode,
            self.max_num_hands,
            self.model_complexity,
            self.min_detection_confidence,
            self.min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img, hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
        return img

    def find_position(self, img, hand_no=0, draw=True):
        landmark_list = []
        hand_type = ""
        if self.results.multi_hand_landmarks:
            hand_no = min(hand_no, len(self.results.multi_hand_landmarks)-1)
            selected_hand = self.results.multi_hand_landmarks[hand_no]
            
            # 新增手部类型检测
            if self.results.multi_handedness:
                hand_type = self.results.multi_handedness[hand_no].classification[0].label
                
            # 原有坐标转换逻辑保持不变...
            for idx, landmark in enumerate(selected_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                landmark_list.append([idx, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                
        return landmark_list, hand_type  # 返回类型信息

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        landmark_list = detector.find_position(img)
        if len(landmark_list) != 0:
            print(landmark_list[4])
        cv2.imshow("Image", img)
        cv2.waitKey(1)
    
    