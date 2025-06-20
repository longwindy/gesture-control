import cv2
import cvzone
import numpy as np
from config import buttonColor, textColor, button_size, caps_lock

class Button():
    def __init__(self, pos, text, size=button_size, action=None):
        self.pos = pos
        self.text = text
        self.size = size
        self.action = action

    def draw(self, img, buttonColor=buttonColor, textColor=textColor, fontScale=4, thickness=4):
        x, y = self.pos
        w, h = self.size
        cvzone.cornerRect(img, (x, y, w, h), 20, rt=3)
        display_text = self.text
        if caps_lock and len(self.text) == 1 and self.text.isaplha():
            print(1)
            display_text = self.text.upper()
        cv2.rectangle(img, self.pos, (x + w, y + h), buttonColor, cv2.FILLED)
        cv2.putText(img, display_text, (np.int32(x + w//3), np.int32(y + h - h//4)), 
                    cv2.FONT_HERSHEY_PLAIN, fontScale, textColor, thickness)
        return img
    
