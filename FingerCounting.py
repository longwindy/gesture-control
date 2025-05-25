import cv2
import time
import os
import HandTrackingModule as htm

wcam, hcam = 1280, 960
cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)
cap.set(cv2.CAP_PROP_FPS, 60)  # 新增帧率设置，建议值30/60

folderPath = "FingerImages"
myList = os.listdir(folderPath)
# print(myList)
overlayList = []
pTime = 0

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # 将图像统一缩放到128x128尺寸
    image = cv2.resize(image, (150, 150))  # 新增缩放操作
    overlayList.append(image)

detector = htm.HandDetector(min_detection_confidence=0.75)
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # 新增水平翻转
    img = detector.find_hands(img)
    all_fingers = []  # 存储多只手的手指状态
    
    # 获取检测到的手部数量
    num_hands = len(detector.results.multi_hand_landmarks) if detector.results.multi_hand_landmarks else 0
    
    for hand_idx in range(num_hands):
        lmList, hand_type = detector.find_position(img, hand_no=hand_idx, draw=False)
        if len(lmList) != 0:
            fingers = []
            # 大拇指判断（根据左右手类型）
            if hand_type == "Right":
                fingers.append(1 if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1] else 0)
            else:
                fingers.append(1 if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1] else 0)
                
            # 其他手指判断（保持原有逻辑）
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            # 计算手指数量 # 存储当前手的手指状态
            all_fingers.append(fingers)
            # 计算当前手的手指数量
            total_fingers = sum(fingers)

            # 新增标注逻辑（显示在手腕位置）
            wrist_x, wrist_y = lmList[0][1], lmList[0][2]  # 0号关键点是手腕
            # 添加在putText之前
            text_size = cv2.getTextSize(hand_type, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            cv2.rectangle(img, 
                         (wrist_x - 55, wrist_y - 5),
                         (wrist_x - 55 + text_size[0], wrist_y + 25 + text_size[1]),
                         (0,0,0), -1)  # 黑色背景框
            cv2.putText(img, f'{hand_type}({total_fingers})', 
                       (wrist_x - 50, wrist_y + 20),  # 偏移量避免遮挡
                       cv2.FONT_HERSHEY_SIMPLEX, 1, 
                       (0, 255, 0) if hand_type == "Right" else (255, 0, 0), 2)  # 右手绿色/左手红色
            

    # h,w,c = overlayList[0].shape
    # img[0:h, 0:w] = overlayList[0]

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (1100, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)