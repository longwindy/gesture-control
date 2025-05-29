import cv2
import numpy as np
from button import Button
from pynput.keyboard import Controller, Key
from config import button_interval, button_size, caps_lock

keyboard = Controller()

keys = [
    ["~", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "="],
    ["Tab", "q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "[", "]"],
    ["Caps", "a", "s", "d", "f", "g", "h", "j", "k", "l", ";", "'"],
    ["Shift", "z", "x", "c", "v", "b", "n", "m", ",", ".", "/"],
    ["Ctrl", "Alt", "Cmd", "Space"]
]

custom_keys = {
    "Tab" : {
        "key" : "Tab", 
        "size": [np.int32(button_size[0]*1.5), button_size[1]],
        "action": [lambda: keyboard.press(Key.tab), lambda: keyboard.release(Key.tab)]},
    "Caps" : {
        "key" : "Caps", 
        "size": [np.int32(button_size[0]*1.75), button_size[1]],
        "action": [lambda: keyboard.press(Key.caps_lock), lambda: keyboard.release(Key.caps_lock), lambda: globals().update(caps_lock = not caps_lock)]},
    "Shift" : {
        "key" : "Shift", 
        "size": [np.int32(button_size[0]*2.25), button_size[1]],
        "action": [lambda: keyboard.press(Key.shift), lambda: keyboard.release(Key.shift)]
        },
    "Ctrl" : {
        "key" : "Ctrl",
        "size": [np.int32(button_size[0]*1.25), button_size[1]],
        "action": [lambda: keyboard.press(Key.ctrl), lambda: keyboard.release(Key.ctrl)]
        },
    "Alt" : {
        "key" : "Alt",
        "size": [np.int32(button_size[0]*1.25), button_size[1]],
        "action": [lambda: keyboard.press(Key.alt), lambda: keyboard.release(Key.alt)]
        },
    "Cmd" : {
        "key" : "Cmd",
        "size": [np.int32(button_size[0]*1.25), button_size[1]],
        "action": [lambda: keyboard.press(Key.cmd), lambda: keyboard.release(Key.cmd)]
        },
    "Space" : {
        "key" : "Space",
        "size": [np.int32(button_size[0]*6.25), button_size[1]],
        "action": [lambda: keyboard.press(Key.space), lambda: keyboard.release(Key.space)]
        }
}

def init_keyboard(start_x, start_y, buttonList, size=button_size, keys = keys, custom_keys = custom_keys):
    current_x, current_y = start_x, start_y
    for i in range(len(keys)):
        current_y = start_y + i * (size[1] + button_interval)
        current_x = start_x
        for j, key in enumerate(keys[i]):
            if key in custom_keys:
                buttonList.append(Button(
                    [current_x, current_y], 
                    custom_keys[key]["key"], 
                    custom_keys[key]["size"],
                    custom_keys[key]["action"]))
                current_x += custom_keys[key]["size"][0] + button_interval
            else:
                current_key = key
                action = [lambda k=current_key: keyboard.press(k), lambda k=current_key: keyboard.release(k)]
                buttonList.append(Button([current_x, current_y], key, size, action=action))
                current_x += size[0] + button_interval
