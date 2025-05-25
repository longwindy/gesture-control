import cv2
import cvzone
import numpy as np

class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.text = text
        self.size = size

    def draw(self, img, buttonColor=(210, 146, 192), textColor=(255, 255, 255), fontScale=4, thickness=4):
        x, y = self.pos
        w, h = self.size
        cvzone.cornerRect(img, (x, y, w, h), 20, rt=3)
        cv2.rectangle(img, self.pos, (x + w, y + h), buttonColor, cv2.FILLED)
        # cv2.putText(img, self.text, (x + 25, y + 75), cv2.FONT_HERSHEY_PLAIN, fontScale, textColor, thickness)
        cv2.putText(img, self.text, (np.int32(x + w/3), np.int32(y + h - h/4)), cv2.FONT_HERSHEY_PLAIN, fontScale, textColor, thickness)
        return img