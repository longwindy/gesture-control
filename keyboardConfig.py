import cv2
import numpy as np
from button import Button
from config import button_interval, button_size

keys = [
    ["~", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "="],
    ["Tab", "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "[", "]"],
    ["Caps", "A", "S", "D", "F", "G", "H", "J", "K", "L", ";", "'"],
    ["Shift", "Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
]

custom_keys = {
    "Tab" : {"key" : "Tab", "size": [np.int32(button_size[0]*1.5), button_size[1]]},
    "Caps" : {"key" : "Caps", "size": [np.int32(button_size[0]*1.75), button_size[1]]},
    "Shift" : {"key" : "Shift", "size": [np.int32(button_size[0]*2.25), button_size[1]]},
}

def init_keyboard(start_x, start_y, buttonList, size=button_size, keys = keys, custom_keys = custom_keys):
    current_x, current_y = start_x, start_y
    for i in range(len(keys)):
        current_y = start_y + i * (size[1] + button_interval)
        current_x = start_x
        for j, key in enumerate(keys[i]):
            if key in custom_keys:
                buttonList.append(Button([current_x, current_y], custom_keys[key]["key"], custom_keys[key]["size"]))
                current_x += custom_keys[key]["size"][0] + button_interval
            else:
                buttonList.append(Button([current_x, current_y], key, size))
                current_x += size[0] + button_interval
